from fastmcp import FastMCP
from config.loader import settings, project_root
import datetime

mcp = FastMCP(
    "TimeService",
    stateless_http=True,
    json_response=True
)

@mcp.tool()
def get_time(tz: str | None = None) -> dict:
    """
    Return current server time in ISO format. Optional tz (ignored in this simple implementation).
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return {"iso": now, "tz": tz}

if __name__ == "__main__":
    cfg = settings.get("mcp", {}).get("time", {})
    port = cfg.get("port", 8001)
    print(f"🚀 Starting TimeService on port {port}")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )

