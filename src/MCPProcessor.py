import importlib
from common import CallableComponent
from config.loader import settings, agent


class MCPProcessor(CallableComponent):
    """
    Chains together a list of MCP server processors.
    Reads class names from settings["agent"]["mcp_processor"],
    imports them from mcp_util.clients, instantiates each, and calls them in sequence.
    """

    def __init__(self):
        super().__init__()
        # List of class names to load, e.g. ["FooServer", "BarServer"]
        class_names = agent["mcp_processor"]
        self.processors = {}
        module = importlib.import_module("mcp_util.clients")
        for cls_name in class_names:
            try:
                cls = getattr(module, cls_name)
            except AttributeError:
                raise ImportError(f"Cannot import '{cls_name}' from mcp_util.clients")
            self.processors[cls_name] = cls()

    def __call__(self, mcp_details, prompt: str) -> list:
        """
        Passes the prompt through each processor in turn.
        """
        results = []
        for mcp_detail in mcp_details:
            if self.processors.get(mcp_detail["tool"]):
                results.append(self.processors.get(mcp_detail["tool"]))(prompt)
        return results