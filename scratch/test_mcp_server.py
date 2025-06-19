import asyncio
import json
from fastmcp import Client


async def main():
    url = "http://localhost:8000/mcp"  # no trailing slash
    async with Client(url) as client:
        # call the tool; returns a list of Content objects
        contents = await client.call_tool("get_weather", {"city": "Bhubaneswar"})

        # grab the first TextContent and parse its `.text`
        text = contents[0].text
        weather: dict = json.loads(text)

        print("As dict:", weather)
        print("Temperature (°C):", weather["temperature_c"])


if __name__ == "__main__":
    asyncio.run(main())