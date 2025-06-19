from fastmcp import FastMCP
from config.loader import settings
from modules.services.weather import WeatherComponent

# 1) Make HTTP stateless + return direct JSON
mcp = FastMCP(
    "WeatherService",
    stateless_http=True,
    json_response=True
)

weather_comp = WeatherComponent()

@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Returns weather info as a dict.
    """
    info = weather_comp(city)
    return info.dict()

if __name__ == "__main__":
    cfg  = settings["mcp_util"]["weather"]
    port = cfg.get("port", 8000)
    print(f"🚀 Starting WeatherService (stateless) on port {port}")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )