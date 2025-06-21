import os
import importlib
from common.PromptBuilderComponent import PromptBuilderComponent

class PromptBuilderTool(PromptBuilderComponent):
    """
    Builds the 'prompt_tool' template (prompt_tool.yml) and
    injects the ToolSelection schema.
    """
    # matches prompt_tool.yml → filename prompt_tool.yml
    template_key = "tool"

    def build_user(self, **context) -> str:
        return self._tpl["user_prompt"].format(**context)

    def build_assistant(self, **context) -> str:
        assistant_body = self._tpl["assistant_prompt"].format(**context)
        postfix = self.build_postfix(**context)
        return f"{assistant_body}{postfix}"