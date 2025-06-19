import importlib
from common import CallableComponent
from config.loader import settings

class MCPProcessor(CallableComponent):
    """
    Chains together a list of MCP server processors.
    Reads class names from settings["agent"]["mcp_processor"],
    imports them from mcp_util.servers, instantiates each, and calls them in sequence.
    """

    def __init__(self):
        super().__init__()
        # List of class names to load, e.g. ["FooServer", "BarServer"]
        class_names = settings["agent"]["mcp_processor"]
        self.processors = []
        module = importlib.import_module("mcp_util.servers")
        for cls_name in class_names:
            try:
                cls = getattr(module, cls_name)
            except AttributeError:
                raise ImportError(f"Cannot import '{cls_name}' from mcp_util.servers")
            self.processors.append(cls())

    def __call__(self, prompt: str) -> str:
        """
        Passes the prompt through each processor in turn.
        """
        result = prompt
        for proc in self.processors:
            result = proc(result)
        return result