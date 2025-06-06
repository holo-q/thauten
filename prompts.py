"""
Prompt Library Utilities

Reusable utilities for working with prompt templates that support:
- Multi-turn conversation format with <|im_start|>role...<|im_end|> structure
- Comment filtering (# lines and inline comments)
- Variable substitution with {tag} replacement
- Fallback to legacy single-turn string templates

=== PROMPT FORMATS ===

Multi-turn conversation format:
```
# Comments are filtered out automatically
<|im_start|>system
System message content<|im_end|>
<|im_start|>user  
User message with {variable} substitution<|im_end|>
<|im_start|>assistant
Assistant response template
```

Legacy single-turn format:
```
Simple template with {variable} substitution
```

The system auto-detects format based on "# Multi-turn conversation format" header.
"""

import os
import re
import logging
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)

class PromptLibrary:
    """
    Utility class for loading and managing prompt templates with multi-turn conversation support.
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize prompt library.
        
        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = prompts_dir
        self._cache = {}  # Cache loaded prompts
    
    def load_prompt(self, filename: str) -> Union[str, List[Dict[str, str]]]:
        """
        Load a prompt template from file.
        
        Args:
            filename: Name of prompt file (e.g., "compression.txt")
            
        Returns:
            Either string template (legacy) or list of conversation messages (multi-turn)
        """
        if filename in self._cache:
            return self._cache[filename]
            
        prompt_path = os.path.join(self.prompts_dir, filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Check if it's multi-turn format
            if content.startswith("# Multi-turn conversation format"):
                result = self._parse_multiturn_prompt(content)
            else:
                # Legacy single-turn format
                result = content
                
            self._cache[filename] = result
            return result
                
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_path}: {e}")
            raise
    
    def _parse_multiturn_prompt(self, content: str) -> List[Dict[str, str]]:
        """
        Parse multi-turn conversation format with standard special tokens.
        
        Args:
            content: Raw prompt file content
            
        Returns:
            List of conversation messages with role and content
        """
        # Filter out comments before parsing
        content = self._filter_comments(content)
        
        # Split on <|im_start|> to get individual messages
        message_blocks = content.split('<|im_start|>')[1:]  # Skip the first empty part
        messages = []
        
        for block in message_blocks:
            # Remove <|im_end|> and split role from content
            block = block.replace('<|im_end|>', '').strip()
            if not block:
                continue
                
            # Split at first newline to separate role from content
            lines = block.split('\n', 1)
            if len(lines) < 2:
                continue
                
            role = lines[0].strip()
            content_text = lines[1].strip()
            
            # Validate role
            if role in ['system', 'user', 'assistant']:
                messages.append({
                    "role": role,
                    "content": content_text
                })
        
        return messages

    def _filter_comments(self, content: str) -> str:
        """
        Remove comment lines and inline comments from prompt content.
        
        Args:
            content: Raw prompt content
            
        Returns:
            Filtered content with comments removed
        """
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Remove entire lines that start with #
            if line.strip().startswith('#'):
                continue
            
            # Remove inline comments (everything from # to end of line)
            if '#' in line:
                line = line[:line.index('#')]
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def format_prompt(self, template: Union[str, List[Dict[str, str]]], **kwargs) -> Union[str, List[Dict[str, str]]]:
        """
        Format a prompt template with given variables.
        
        Args:
            template: Either string template or list of conversation messages
            **kwargs: Variables to substitute (e.g., content="hello")
            
        Returns:
            Formatted prompt in same format as input
        """
        if isinstance(template, list):
            # Multi-turn conversation format
            formatted_messages = []
            for message in template:
                formatted_content = message["content"].format(**kwargs)
                formatted_messages.append({
                    "role": message["role"],
                    "content": formatted_content
                })
            return formatted_messages
        else:
            # Single-turn format
            return template.format(**kwargs)
    
    def clear_cache(self):
        """Clear the prompt cache (useful for development/testing)."""
        self._cache.clear()
    
    def list_prompts(self) -> List[str]:
        """
        List available prompt files in the prompts directory.
        
        Returns:
            List of prompt filenames
        """
        try:
            return [f for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        except OSError:
            logger.warning(f"Could not list prompts directory: {self.prompts_dir}")
            return []

# Convenience functions for direct usage
_default_library = None

def get_default_library(prompts_dir: str = "prompts") -> PromptLibrary:
    """Get or create the default prompt library instance."""
    global _default_library
    if _default_library is None:
        _default_library = PromptLibrary(prompts_dir)
    return _default_library

def load_prompt(filename: str, prompts_dir: str = "prompts") -> Union[str, List[Dict[str, str]]]:
    """Convenience function to load a prompt using the default library."""
    return get_default_library(prompts_dir).load_prompt(filename)

def format_prompt(template: Union[str, List[Dict[str, str]]], **kwargs) -> Union[str, List[Dict[str, str]]]:
    """Convenience function to format a prompt using the default library."""
    return get_default_library().format_prompt(template, **kwargs) 