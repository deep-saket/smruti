from fastmcp import FastMCP
from config.loader import settings

# This service wraps a trivial translation helper using a free API or local stub.
# For now we implement a local stub that pretends to translate by returning the input
# and language code; you can wire this to a real API (DeepL, Google Translate, etc.).

mcp = FastMCP("TranslateService", stateless_http=True, json_response=True)

@mcp.tool()
def translate(text: str, target_lang: str = "en") -> dict:
    """
    Translate text to target language. Currently a stub (returns input and target).
    Replace implementation with a real translation API if available.
    """
    # stub: echo back with target language
    return {"translated_text": text, "target_lang": target_lang}

if __name__ == "__main__":
    cfg = settings.get('mcp', {}).get('translate', {})
    port = cfg.get('port', 8003)
    print(f"🚀 Starting TranslateService on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)

