from fastmcp import Client
from common import MCPClientComponent
import json

class WeatherClient(MCPClientComponent):
    """
    Client for the /weather endpoint.
    """

    async def _process_prompt(self, prompt, *args, **kwargs):
        pass
