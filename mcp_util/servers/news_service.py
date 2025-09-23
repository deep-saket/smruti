from fastmcp import FastMCP
from config.loader import settings

# Simple NewsService stub. Replace with an actual news API integration if required.
# Tool: get_headlines(topic: str = 'general', limit: int = 5) -> dict

mcp = FastMCP("NewsService", stateless_http=True, json_response=True)

SAMPLE_HEADLINES = {
    'general': [
        "Local AI lab releases new lightweight model for edge devices",
        "City council approves new transit plan",
        "Breakthrough in battery tech promises faster charging",
        "Major retailer announces sustainability initiative",
        "New study shows benefits of short naps"
    ],
    'tech': [
        "Open-source community adopts faster compilation pipeline",
        "Edge inference becomes mainstream with new runtimes",
        "GPU vendor releases optimized kernels for mobile"
    ]
}

@mcp.tool()
def get_headlines(topic: str = 'general', limit: int = 5) -> dict:
    topic = topic.lower() if topic else 'general'
    headlines = SAMPLE_HEADLINES.get(topic, SAMPLE_HEADLINES['general'])
    return {"topic": topic, "headlines": headlines[:limit]}

if __name__ == "__main__":
    cfg = settings.get('mcp', {}).get('news', {})
    port = cfg.get('port', 8005)
    print(f"🚀 Starting NewsService on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)

