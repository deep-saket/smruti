import importlib
from typing import List, Dict, Any
from common import CallableComponent
from .ModelManager import ModelManager
from prompts.PromptBuilderTool import PromptBuilderTool

class ToolsDecider(CallableComponent):
    """
    Decides which external tool(s) to invoke based on user input.
    Builds a prompt via PromptBuilderTool, queries an LLM, parses the
    decision, and maps tool names to actual tool classes.
    """
    def __init__(self, lm=None):
        super().__init__()
        # Ensure models are loaded
        self.model_manager = ModelManager
        # choose the LLM for tool‐deciding
        if lm is not None:
            self.lm = lm
        else:
            self.lm = getattr(self.model_manager, "tool_decider_lm", None) \
                      or self.model_manager.llm
        # prompt builder for tool selection
        self.prompt_builder = PromptBuilderTool()
        # load name→class mapping from the prompt template
        tpl = self.prompt_builder._tpl
        mapping = tpl.get("tool_mapping")
        if not mapping:
            raise KeyError("prompt_tool.yml missing required field: 'tool_mapping'")
        self.tool_map = {}
        for name, cls_path in mapping.items():
            module_name, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            self.tool_map[name] = getattr(mod, cls_name)
        # parser for the tool‐selection JSON
        self.parser = self.prompt_builder.parser

    def __call__(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts, each with:
          - tool           : the mapped tool class
          - requires_media : "image"|"video"|None
          - parameters     : dict of parameters
        """
        prompts = self.prompt_builder(user_input=user_input)
        raw = self.lm.infer(
            system_prompt=prompts.get("system"),
            prompt=prompts.get("user"),
            assistant_message=prompts.get("assistant"),
        )
        decision = self.parser(raw)
        results = []
        for sel in decision.tools:
            if sel.tool == "none":
                continue
            cls = self.tool_map.get(sel.tool)
            if cls is None:
                raise ValueError(f"No tool class mapped for name: {sel.tool}")
            results.append({
                "tool": cls,
                "requires_media": sel.requires_media,
                "parameters": sel.parameters
            })
        return results

