import re
from abc import abstractmethod
from fastmcp import Client
import json
from common import CallableComponent
from config.loader import settings

class MCPClientComponent(CallableComponent):
    """
    Base for all MCP‐based HTTP clients. The subclass name must end with 'Client';
    we strip that off, snake-case the remainder, and look up:
        settings["mcp_util"][<snake_name>]["server_url"]
    """

    def __init__(self):
        super().__init__()
        cls_name = self.__class__.__name__
        if not cls_name.endswith("Client"):
            raise ValueError(f"{cls_name!r} must end with 'Client'")
        base = cls_name[:-6]  # drop 'Client'
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', base).lower()
        self.server_url = settings["mcp"][snake]["server_url"]
        self.tool_name = settings["mcp"][snake]["tool_name"]

    async def __call__(self, prompt, *args, **kwargs):
        return await self._process_prompt(prompt, *args, **kwargs)

    async def request(self, **kwargs):
        """
        Makes the actual HTTP call using the processed prompt.
        """
        async with Client(self.server_url) as client:
            # call the tool; returns a list of Content objects  
            contents = await client.call_tool(self.tool_name, kwargs)

            # grab the first TextContent and parse its `.text`
            text = contents[0].text
            weather: dict = json.loads(text)
            return weather

    @abstractmethod
    async def _process_prompt(self, prompt, *args, **kwargs):
        """
        Must be implemented by subclass to process input arguments into a prompt.
        Must be implemented as an async method.
        """
        raise NotImplementedError 