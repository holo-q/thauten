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
  - `<|USER:id|>`: Sets role to user with an ID for reference.
  - `<|USER GOAL=...|>`: Sets a bingo reward curriculum by decomposing the hand-written high-taste goal prompt.
  - `<|HOLO|>`: Sets role to assistant.
  - `<|HOLO:id|>`: Sets role to assistant with an ID for reference.
  - `<|HOLO TAG=...|>`: Sets a cognitive fence tag for the assistant's response.
  - `<|HOLO GOAL=...|>`: Sets a goal for the assistant's response.

- Object Tags: Define data variables and their attributes.
  - `<|OBJ:id|>`: A placeholder for a data variable.
  - `<|OBJ:id <>name=val|>`: A placeholder with attributes.
  - `<|id|>`: Shorthand for a data variable reference.

- Attribute Syntax:
  - `<>name=val`: Sets an attribute on the tag.
  - `<>tag`: Shorthand for setting a tag attribute.

- Special Tags:
  - `---`: Clears the context to simulate multiple conversations.
  - `---`: Like --- but marks the next conext for training.

The parser converts a prompt file into a structured `PromptTemplate` object,
which contains a sequence of nodes representing the parsed DSL.
"""
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Group
from rich.rule import Rule
from rich.text import Text

logger = logging.getLogger(__name__)

# TODO type alias the openai conversation dict format
# TODO all union etc should use the type hinting syntax

# --- Data Structures for Parsed Prompt Nodes ---

@dataclass
class Node:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ContextResetNode(Node):
    """Reset the context."""
    train = True

@dataclass
class RoleNode(Node):
    """Sets the role tag by printing the special token."""
    role: str = ""

@dataclass
class InferNode(Node):
    """Run inference"""
    goal: str = ""
    fence: str = ""

    @property
    def display_goal(self):
        return self.goal[:30].replace(chr(10), '\\n')

@dataclass
class TextNode(Node):
    """Represents a block of plain text content."""
    text: str = ""

    @property
    def display_text(self):
        return self.text[:30].replace(chr(10), '\\n')


@dataclass
class ObjNode(Node):
    """Represents a data variable placeholder, e.g., <|OBJ=sample|>."""
    var_id: str = ""
    var_attributes: dict[str, str] = field(default_factory=dict)


PromptNode = Union[ContextResetNode, RoleNode, TextNode, ObjNode, InferNode]

@dataclass
class PromptTemplate:
    """
    Structured representation of a parsed prompt template from the DSL.
    Contains a sequence of nodes that define the entire prompt and its metadata.
    """
    nodes: list[PromptNode] = field(default_factory=list)

    @property
    def trained_contexts(self) -> list[int]:
        """Returns a list of indices where context resets occur."""
        contexts = []
        current = 0
        for i, node in enumerate(self.nodes):
            if isinstance(node, ContextResetNode):
                if i > 0: # if theres no context reset on the first line, then it's implicit
                    current += 1
                if node.train:
                    contexts.append(current)

        return contexts

    def to_rich_debug(self):
        """Format the prompt template nodes for rich debug display."""
        text = Text()
        for node in self.nodes:
            if isinstance(node, RoleNode):
                text.append("RoleNode (", style="bold cyan")
                text.append(f"role={node.role}", style="cyan")
                text.append(")\n", style="cyan")
            elif isinstance(node, InferNode):
                text.append("InferNode (", style="bold cyan")
                parts = []
                if node.goal:
                    parts.append(f"goal={node.display_goal}...")
                if node.fence:
                    parts.append(f"tag={node.fence}")
                text.append(", ".join(parts), style="yellow")
                text.append(")\n", style="cyan")
            elif isinstance(node, TextNode):
                text.append("TextNode", style="bold white")
                text.append(f" (text={node.display_text}...)\n", style="white")
            elif isinstance(node, ObjNode):
                text.append("ObjNode", style="bold green")
                text.append(f" (var={node.var_id}", style="green")
                if node.var_attributes:
                    text.append(f", attrs={node.var_attributes}", style="yellow")
                text.append(")\n", style="green")
            elif isinstance(node, ContextResetNode):
                text.append("ContextResetNode\n", style="bold red")
        return text

@dataclass
class PromptContext:
    messages: list[Any] = field(default_factory=list)

    @property
    def text_rich(self) -> Text:
        """Helper to format a conversation for rich display."""
        text = Text()
        for msg in self.messages:
            role = msg.get("role", "none")
            content = msg.get("content", "")
            c = "cyan"
            if role == "system": c = "yellow"
            if role == "user": c = "green"
            if role == "assistant": c = "magenta"
            text.append(f"<|{role.upper()}|>\n", style=f"bold {c}")
            text.append(content + "\n")
        return text


@dataclass
class PromptInstance:
    contexts: list[PromptContext] = field(default_factory=list)

    def extract_fence(self, tag, role="assistant") -> Optional[str]:
        for c in self.contexts:
            ret = extract_fence(c.messages, tag, role)
            if ret:
                return ret
        return None

    def to_rich(self):
        renderables = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                renderables.append(Rule(style="yellow"))
            renderables.append(ctx.text_rich)
        return Group(*renderables)


class PromptLibrary:
    """
    Utility class for loading, parsing, and formatting prompt templates.
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache = {}

    def load_holoware(self, filename: str) -> PromptTemplate:
        """
        Load a prompt from file and parse it if it uses the DSL.
        """
        if filename in self._cache:
            return self._cache[filename]

        prompt_path = os.path.join(self.prompts_dir, filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # First, filter out comment lines
            text = self._filter_comments(text)  # TODO do this as part of _parse_prompt
            tpl = self._parse_prompt(text)

            self._cache[filename] = tpl
            return tpl

        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_path}: {e}")
            raise

    def _parse_prompt(self, content: str) -> "PromptTemplate":
        """
        Parse a string containing the prompt DSL into a PromptTemplate object.
        Uses manual string parsing instead of regex for better maintainability.
        """
        ret = PromptTemplate()
        pos = 0
        content_len = len(content)

        def skip_whitespace() -> None:
            nonlocal pos
            while pos < content_len and content[pos].isspace():
                pos += 1

        def read_until_tag_end() -> str:
            nonlocal pos
            start = pos
            while pos < content_len:
                if content[pos:pos + 2] == '|>':
                    pos += 2
                    return content[start:pos - 2]
                pos += 1
            raise ValueError(f"Unclosed tag at position {start}")

        def parse_attributes(tag: str) -> Tuple[Dict[str, str], Dict[str, str], str]:
            """Parse attributes in the format <>name=val, <>tag, or space-separated name=val.
            Returns a tuple of (node_attrs, fence_attrs, base_tag) where:
            - node_attrs: attributes from space-separated format (e.g. "USER GOAL=text")
            - fence_attrs: attributes from <> prefix format (e.g. "<>name=val")
            - base_tag: the base tag without attributes
            All attribute names are automatically lowercased.
            """
            node_attrs: Dict[str, str] = {}
            fence_attrs: Dict[str, str] = {}

            # First check for space-separated attributes (e.g. "USER GOAL=some text")
            parts = tag.split(' ', 1)
            if len(parts) > 1:
                attr_parts = parts[1:]
                for part in attr_parts:
                    if "=" in part:
                        name, val = part.split('=', 1)
                        name = name.strip().lower()
                        val = val.strip()
                        if name.startswith("<>"):
                            fence_attrs[name[2:]] = val
                        else:
                            node_attrs[name] = val
                    else:
                        if part.startswith("<>"):
                            node_attrs['fence'] = part[2:]
                        else:
                            raise ValueError("Unexpected non-keyed symbol in holotag after tag name.")

            return node_attrs, fence_attrs, parts[0]

        def parse_tag(tag: str, out: list) -> None:
            tag = tag.strip()
            tag_upper = tag.upper()

            node_attrs, fence_attrs, tag = parse_attributes(tag)
            node_id = ""
            if ':' in tag:
                node_id = tag.split(':', 1)[1].strip()

            if tag_upper == "SYS":
                out.append(RoleNode(role='system'))
                return

            if tag_upper.startswith("USER") or tag_upper.startswith("HOLO"):
                role = 'user' if tag_upper.startswith('USER') else 'assistant'

                out.append(RoleNode(role=role, id=node_id))

                if len(node_attrs) > 0:
                    out.append(InferNode(
                        id=node_id,
                        goal=node_attrs.get("goal", ''),
                        fence=node_attrs.get("fence", '')
                    ))
                return

            if tag_upper.startswith('INFER'):
                out.append(InferNode(
                    id=node_id,
                    goal=node_attrs.get('goal', ''),
                    fence=node_attrs.get('fence', '')
                ))
                return

            if tag_upper.startswith('OBJ:'):
                var_name = tag[4:].strip()
                out.append(ObjNode(var_id=var_name, var_attributes=fence_attrs))
                return

            if tag == '---':
                out.append(ContextResetNode())
                return

            if tag == '===':
                out.append(ContextResetNode())
                return

            if ':' not in tag and not tag_upper.startswith(('USER', 'HOLO', 'SYS')):
                out.append(ObjNode(var_id=tag, var_attributes=fence_attrs))
                return

            logger.warning(f"Unknown tag type: {tag}")

        def read_text_content() -> str:
            nonlocal pos
            start = pos
            while pos < content_len:
                if content[pos:pos + 2] == '<|':
                    break
                pos += 1
            return content[start:pos]

        # Main parsing loop ----------------------------------------
        role = None
        out = []

        while pos < content_len:
            if pos >= content_len:
                break

            if content[pos:pos + 2] == '<|':
                pos += 2
                tag_content = read_until_tag_end()
                parse_tag(tag_content, out)
            else:
                text = read_text_content()
                last_is_role = len(ret.nodes) and isinstance(ret.nodes[-1], RoleNode)

                # Skip intermediate whitespace between role & first text
                skip = last_is_role and not text.strip()

                # Skip leading whitespace same as intermediate whitespace
                if last_is_role:
                    text = text.lstrip()

                if text and not skip:
                    out.append(TextNode(text=text))

            # Validate & commit
            for n in out:
                if isinstance(n, RoleNode):
                    if role != n.role:
                        role = n.role
                    else:
                        continue
                if isinstance(n, ContextResetNode):
                    if not role:
                        continue  # discard redundant context reset (there can be no existing context without a role)
                    role = None
                if isinstance(n, TextNode) and not role:
                    if n.text.strip():
                        raise ValueError(f"Cannot have text node before a role. (text: {n.display_text})")
                    continue
                if isinstance(n, (InferNode, ObjNode)) and not role:
                    raise ValueError("Cannot have infer or obj node before a role.")

                ret.nodes.append(n)

            out.clear()

        return ret

    def _filter_comments(self, content: str) -> str:
        """
        Remove comment lines (starting with #) from prompt content.
        """
        lines = content.split('\n')
        # Rebuild the content string, ignoring any line that starts with '#'
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        return '\n'.join(filtered_lines)

    def rollout(self, template: PromptTemplate, fn_inference, input_env) -> PromptInstance:
        if not isinstance(template, PromptTemplate):
            raise TypeError("Rollout is only supported for PromptTemplate objects.")

        msg = None
        inst = PromptInstance(contexts=[PromptContext()])
        env = dict(input_env)

        for node in template.nodes:
            text = ""

            if isinstance(node, RoleNode):
                if msg:
                    if node.role == msg['role']:
                        continue
                    else:
                        inst.contexts[-1].messages.append(msg)
                msg = {"role": node.role, "content": ""}

            if isinstance(node, ContextResetNode):
                if msg:
                    inst.contexts[-1].messages.append(msg)
                    msg = None
                inst.contexts.append(PromptContext())

            if isinstance(node, TextNode):
                text = node.text

            if isinstance(node, ObjNode):
                text += f"\n<obj id=\'{node.var_id}\'>"
                text += env.get(node.var_id, 'NOT_FOUND')
                text += f"</obj>"

            if isinstance(node, InferNode):
                # We have our context up to this point. Now, generate.
                current_ctx = [msg.copy() for msg in inst.contexts[-1].messages]
                if msg:
                    current_ctx.append(msg.copy())

                if node.fence:
                    text = f'<{node.fence}>'

                # TODO this should be dry run only
                if node.goal:
                    output = node.goal
                else:
                    output = fn_inference(current_ctx)

                # TODO this should be dry run only
                if output == '<|INFER|>' and node.goal:
                    output = f'<|INFER:{node.goal}|>'

                text += output

                if node.fence:
                    text += f'</{node.fence}>'

                # Store node text to env
                if node.id:
                    env[node.id] = output

            if text:
                if not msg:
                    raise ValueError("TODO")
                msg['content'] += text

        if msg:
            inst.contexts[-1].messages.append(msg)

        return inst

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
                        messages.append(msg)
                    else:
                        msg = {"role": node.role, "content": ""}
                    msg["content"] += f"<|im_start|>"
                    continue

                if msg is None:
                    raise ValueError("wtf!! set role first")

                # If the assistant turn has a wrapper, add the opening tag
                if isinstance(node, TextNode):
                    msg["content"] += node.text
                    continue

                if isinstance(node, InferNode) and (node.goal or node.fence):
                    msg["content"] += f"{{INFERENCE-RESULT}}"
                    continue

                if isinstance(node, ObjNode):
                    msg["content"] += f"{{{node.var_id}}}"
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

    def clear_cache(self):
        self._cache.clear()

    def list_prompts(self) -> List[str]:
        try:
            return [f for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        except OSError:
            logger.warning(f"Could not list prompts directory: {self.prompts_dir}")
            return []

_default_library = None

def get_default_library(prompts_dir: str = "prompts") -> PromptLibrary:
    """Get or create the default prompt library instance."""
    global _default_library
    if _default_library is None or _default_library.prompts_dir != prompts_dir:
        _default_library = PromptLibrary(prompts_dir)
    return _default_library

def load_prompt(filename: str, prompts_dir: str = "prompts") -> Union[PromptTemplate, str]:
    """Convenience function to load a prompt using the default library."""
    return get_default_library(prompts_dir).load_holoware(filename)

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

def rollout_prompt(template: PromptTemplate, generate_fn, env) -> PromptInstance:
    """Convenience function to perform a rollout on a prompt template."""
    return get_default_library().rollout(template, generate_fn, env)

def format_conversation(conversation: list[Any]):
    text = ""
    for msg in conversation:
        text += msg['content']
    return text

def extract_fence(messages: List[Dict[str, Any]], wrapper_tag: Optional[str], role='assistant') -> Optional[str]:
    """Extract content from a dynamic wrapper tag, e.g., <compress>...</compress>"""
    if not wrapper_tag:
        return messages.strip() if isinstance(messages, str) else messages[-1]["content"].strip()

    tag = wrapper_tag.lower()

    for msg in reversed(messages):
        if role is not None and msg["role"] != role:
            continue
        content = msg["content"]
        matches = list(re.finditer(fr'<{tag}>\s*(.*?)\s*(?:</{tag}>|$)', content, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()

    return None

def extract_json(conversation: str) -> Optional[str]:
    # TODO take str|list[Any]
    # Extract JSON from anywhere in the response
    # Look for ```json blocks first
    block_match = re.search(r'```json\s*(.*?)\s*```', conversation, re.DOTALL)
    if block_match:
        ret = block_match.group(1).strip()
    else:
        # Look for standalone JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*}[^{}]*)*}', conversation, re.DOTALL)
        if match:
            ret = match.group(0)
        else:
            logger.warning("No JSON found in evaluation")
            return None

    return ret
