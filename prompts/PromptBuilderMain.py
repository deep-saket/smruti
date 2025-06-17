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
        Render the user_prompt from the YAML template.
        Expects context keys matching placeholders defined in user_prompt.
        """
        template = self._tpl["user_prompt"]
        return template.format(**context)

    def build_assistant(self, **context: Any) -> str:
        """
        Render the assistant_prompt body.
        Expects context keys matching those used in your YAML's
        assistant_prompt template (e.g. new_message, context, etc.).
        """
        # Format the assistant prompt section
        assistant_body = self._tpl["assistant_prompt"].format(**context)
        # Optionally append the postfix if desired by your prompt design
        postfix = self.build_postfix(**context)
        return f"{assistant_body}{postfix}"