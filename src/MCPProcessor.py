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
        Expects `mcp_details` to be an iterable of dict-like entries with a "tool" key.
        Returns a list of processor outputs (in the same order as `mcp_details`).
        """
        results = []
        for mcp_detail in mcp_details:
            proc = self.processors.get(mcp_detail.get("tool")) if isinstance(mcp_detail, dict) else self.processors.get(mcp_detail)
            if proc:
                # Call the processor with the prompt and append its result
                try:
                    res = proc(prompt)
                except TypeError:
                    # Some processors may expect (prompt, **kwargs) or be callables that
                    # expose a `__call__` with different signature — try calling without
                    # positional args as a fallback.
                    res = proc()
                results.append(res)
        return results