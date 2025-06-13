"""
Prompt Library Utilities

This module provides a robust parsing and formatting system for a custom prompt
DSL (Domain-Specific Language). It's designed to handle complex, multi-part
prompt structures that include not just conversational turns, but also metadata
for training and inference processes.

=== PROMPT DSL SYNTAX ===

The DSL uses a tag-based system with the `<|...|>` delimiter.

- Role Tags: Define the speaker in a conversation.
  - `<|SYS|>`: Sets role to system.
  - `<|USER|>`: Sets role to user.
  - `<|HOLO|>`: Sets role to assistant, and begin inference.
  - `<|HOLO TAG=tag|>`: inference inside a <tag>...</tag> known as a cognitive fence in prompt engineering.
  - `<|HOLO GOAL=...|>`: set a bingo reward curriculum by decomposing the hand-written high-taste goal prompt. Encourage the model to rediscover the human's highest standards through granular reconstitution. Can be combined with TAG.
  - `<|DATA=text|>`: A placeholder for a data variable that will be injected from environment context or dataset sample and its columns

The parser converts a prompt file into a structured `PromptTemplate` object,
which contains a sequence of nodes representing the parsed DSL.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


# --- Data Structures for Parsed Prompt Nodes ---

@dataclass
class RoleNode:
    """Sets the role tag by printing the special token."""
    role: str

@dataclass
class FixedNode:
    """Represents a block of plain text content."""
    text: str

@dataclass
class DataNode:
    """Represents a data variable placeholder, e.g., <|DATA=sample|>."""
    variable_name: str

@dataclass
class InferNode:
    """Represents a metadata goal for the training loop, e.g., <|HOLO GOAL=...|>, <|HOLO TAG=think|>, ...."""
    goal: str = ""
    tag: str = ""

@dataclass
class RolloutNode:
    """Represents the result of a generation call during rollout."""
    text: str

PromptNode = Union[RoleNode, FixedNode, DataNode, InferNode, RolloutNode]

@dataclass
class PromptTemplate:
    """
    Structured representation of a parsed prompt template from the DSL.
    Contains a sequence of nodes that define the entire prompt and its metadata.
    """
    nodes: List[PromptNode] = field(default_factory=list)

class PromptLibrary:
    """
    Utility class for loading, parsing, and formatting prompt templates.
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache = {}

    def load_prompt(self, filename: str) -> Union[PromptTemplate, str]:
        """
        Load a prompt from file and parse it if it uses the DSL.
        """
        if filename in self._cache:
            return self._cache[filename]

        prompt_path = os.path.join(self.prompts_dir, filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # First, filter out comment lines
            content = self._filter_comments(content)

            if content.strip().startswith('<|'):
                result = self._parse_dsl_prompt(content)
            else:
                result = content.strip()  # Legacy format

            self._cache[filename] = result
            return result

        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_path}: {e}")
            raise

    def _filter_comments(self, content: str) -> str:
        """
        Remove comment lines (starting with #) from prompt content.
        """
        lines = content.split('\n')
        # Rebuild the content string, ignoring any line that starts with '#'
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        return '\n'.join(filtered_lines)

    def _parse_dsl_prompt(self, content: str) -> "PromptTemplate":
        """
        Parse a string containing the prompt DSL into a PromptTemplate object.
        """
        template = PromptTemplate()

        # Regex to find all DSL tags, keeping them as delimiters
        tag_pattern = re.compile(
            r'(<\|(?:SYS|USER.*?|HOLO.*?|DATA=.*?|INFER:.*?|INFER|![A-Z_]+)\|>)',
            re.IGNORECASE | re.DOTALL
        )

        parts = tag_pattern.split(content)

        for part in parts:
            if not part:
                continue

            stripped_part = part.strip()
            if tag_pattern.fullmatch(stripped_part):
                # It's a DSL tag
                tag = stripped_part[2:-2].strip()
                t = tag.upper()

                role_to_set = None
                if t.startswith('USER'):
                    role_to_set = 'user'
                elif t.startswith('HOLO'):
                    role_to_set = 'assistant'

                if role_to_set:
                    template.nodes.append(RoleNode(role_to_set))

                    goal_val = ""
                    tag_val = ""

                    goal_match = re.search(r'GOAL=', tag, re.IGNORECASE)
                    tag_match = re.search(r'TAG=', tag, re.IGNORECASE)

                    if goal_match:
                        val = re.split(r'GOAL=', tag, maxsplit=1, flags=re.IGNORECASE)[1]
                        if tag_match:
                            val = re.split(r'TAG=', val, maxsplit=1, flags=re.IGNORECASE)[0]
                        goal_val = val.strip()

                    if tag_match:
                        val = re.split(r'TAG=', tag, maxsplit=1, flags=re.IGNORECASE)[1]
                        if goal_match:
                            val = re.split(r'GOAL=', val, maxsplit=1, flags=re.IGNORECASE)[0]
                        tag_val = val.strip()

                    if goal_val or tag_val:
                        template.nodes.append(InferNode(goal=goal_val, tag=tag_val))

                elif t == 'SYS':
                    template.nodes.append(RoleNode('system'))
                elif t.startswith('DATA='):
                    var_name = tag[len('DATA='):].strip()
                    template.nodes.append(DataNode(variable_name=var_name))
                elif t == 'INFER':
                    template.nodes.append(InferNode())

            else:
                # It's plain text content
                template.nodes.append(FixedNode(text=part))

        return template

    def format_prompt(self, template: Union[PromptTemplate, str, List[Dict[str, str]]], **kwargs) -> Union[str, List[Dict[str, str]]]:
        """
        Format a template with given variables into a message list or string.
        """
        if isinstance(template, PromptTemplate):
            messages = []
            msg = None

            for node in template.nodes:
                if isinstance(node, RoleNode):
                    if msg and msg['role'] != node.role:
                        msg['content'] = msg['content'].strip()
                        # msg["content"] += f"<|im_end|>"
                        messages.append(msg)
                    else:
                        msg = {"role": node.role, "content": ""}
                    msg["content"] += f"<|im_start|>"
                    continue

                if msg is None:
                    raise ValueError("wtf!! set role first")

                # If the assistant turn has a wrapper, add the opening tag
                if isinstance(node, FixedNode):
                    msg["content"] += node.text
                    continue

                if isinstance(node, RolloutNode):
                    msg["content"] += node.text
                    continue

                if isinstance(node, InferNode):
                    msg["content"] += f"{{INFERENCE-RESULT}}"
                    continue

                if isinstance(node, DataNode):
                    msg["content"] += f"{{{node.variable_name}}}"
                    continue

            if msg:
                msg['content'] = msg['content'].strip()
                messages.append(msg)

            return messages

        elif isinstance(template, list):  # Legacy multi-turn
            formatted_messages = []
            for message in template:
                formatted_content = message["content"].format(**kwargs)
                formatted_messages.append({
                    "role":    message["role"],
                    "content": formatted_content
                })
            return formatted_messages

        else:  # Legacy single-turn
            return template.format(**kwargs)

    def rollout(self, template: PromptTemplate, generate_fn, data) -> list[Any]:
        if not isinstance(template, PromptTemplate):
            raise TypeError("Rollout is only supported for PromptTemplate objects.")

        messages = []
        current_msg = None

        for node in template.nodes:
            if isinstance(node, RoleNode):
                if current_msg:
                    if node.role == current_msg['role']:
                        continue
                    else:
                        messages.append(current_msg)
                current_msg = {"role": node.role, "content": f"<|im_start|>"}
                continue

            if current_msg is None:
                raise ValueError("A role must be set before content can be added.")

            if isinstance(node, FixedNode):
                current_msg["content"] += node.text
                continue

            if isinstance(node, DataNode):
                current_msg["content"] += f"<|DATA:{data.get(node.variable_name, 'NOT-FOUND')}|>"
                continue

            if isinstance(node, InferNode):
                # We have our context up to this point. Now, generate.
                context_for_generation = [msg.copy() for msg in messages]
                if current_msg:
                    context_for_generation.append(current_msg.copy())

                # Substitute variables in the context needed for generation.
                for msg in context_for_generation:
                    # Replace placeholders like {input} with actual data
                    content = msg['content']
                    for key, value in data.items():
                        placeholder = f"{{{key}}}"
                        if placeholder in content:
                            content = content.replace(placeholder, str(value))
                    msg['content'] = content

                if node.tag:
                    current_msg["content"] += f'<{node.tag}>'

                generated_text = generate_fn(context_for_generation)
                if generated_text == "<|INFERENCE|>" and node.goal:
                    generated_text = f"<|INFERENCE:{node.goal}|>"

                current_msg["content"] += generated_text

                if node.tag:
                    current_msg["content"] += f'</{node.tag}>'


        if current_msg:
            messages.append(current_msg)

        return messages

    def clear_cache(self):
        self._cache.clear()

    def list_prompts(self) -> List[str]:
        try:
            return [f for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        except OSError:
            logger.warning(f"Could not list prompts directory: {self.prompts_dir}")
            return []

# --- Convenience Functions ---

_default_library = None

def get_default_library(prompts_dir: str = "prompts") -> PromptLibrary:
    """Get or create the default prompt library instance."""
    global _default_library
    if _default_library is None or _default_library.prompts_dir != prompts_dir:
        _default_library = PromptLibrary(prompts_dir)
    return _default_library

def load_prompt(filename: str, prompts_dir: str = "prompts") -> Union[PromptTemplate, str]:
    """Convenience function to load a prompt using the default library."""
    return get_default_library(prompts_dir).load_prompt(filename)

def format_prompt(template: Union[PromptTemplate, str, List[Dict[str, str]]], **kwargs) -> Union[str, List[Dict[str, str]]]:
    """Convenience function to format a prompt."""
    # The template could be a legacy list, so we handle that case.
    if isinstance(template, PromptTemplate):
        # The new formatter expects the data variables to be in the kwargs directly
        messages = get_default_library().format_prompt(template, **kwargs)

        if isinstance(messages, str):
            return messages.format_map(kwargs)

        # Now, perform the f-string style substitution
        formatted_messages = []
        for message in messages:
            try:
                # The format_map is safer than format as it doesn't fail on extra keys
                formatted_content = message["content"].format_map(kwargs)
                formatted_messages.append({
                    "role":    message["role"],
                    "content": formatted_content
                })
            except KeyError as e:
                logger.warning(f"Missing variable for formatting prompt: {e}")
                # Add the message with the placeholder to allow debugging
                formatted_messages.append(message)
        return formatted_messages

    # Handle legacy formats
    if isinstance(template, list):
        formatted_messages = []
        for message in template:
            formatted_content = message["content"].format(**kwargs)
            formatted_messages.append({
                "role":    message["role"],
                "content": formatted_content
            })
        return formatted_messages
    else:
        return str(template).format(**kwargs)

def rollout_prompt(template: PromptTemplate, generate_fn, data) -> list[Any]:
    """Convenience function to perform a rollout on a prompt template."""
    return get_default_library().rollout(template, generate_fn, data)
