# prompt/PromptBuilderMain.py

from typing import Any, Tuple
from common import PromptBuilderComponent

class PromptBuilderMain(PromptBuilderComponent):
    """
    Main prompt builder for the core conversational flow.

    By default this will look for:
        prompt_main.yml
    in your templates directory. If you instead named your file
    prompt_chat.yml, you can override:

        template_key = "chat"

    to point at prompt_chat.yml.
    """
    # If your YAML file is named prompt_chat.yml, uncomment:
    # template_key = "chat"

    def build_user(self, **context: Any) -> str:
        """
        Render the main user_prompt body.
        Expects context keys:
          - new_message: str
          - context: str
          - memory_snippets: str  (semantic memory reads)
          - memory_writes: str    (new facts to write)
        """
        new_msg        = context.get("new_message", "")
        convo          = context.get("context", "")
        memory_reads   = context.get("memory_snippets", "")
        memory_writes  = context.get("memory_writes", "")

        parts = []
        parts.append(f"User says: {new_msg}")
        if convo:
            parts.append(f"\nConversation history:\n{convo}")
        if memory_reads:
            parts.append(f"\nRelevant memories:\n{memory_reads}")
        if memory_writes:
            parts.append(f"\nMemory to write:\n{memory_writes}")

        return "\n\n".join(parts)
