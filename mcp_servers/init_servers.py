import importlib
from config.loader import settings

# Iterate over each MCP service defined under the top-level 'mcp_servers' key in settings.yaml
for name, cfg in settings["mcp_servers"].items():
    module = cfg.get("module")
    port   = cfg.get("port")
    if not module or not port:
        print(f"Skipping '{name}': module or port missing in settings.")
        continue

    full_mod = f"mcp_servers.server.{module}"
    try:
        mod = importlib.import_module(full_mod)
        mcp = getattr(mod, "mcp_servers", None)
        if not mcp:
            print(f"Module {full_mod} has no 'mcp_servers' instance.")
            continue

        print(f"Starting MCP server '{name}' on port {port}...")
        mcp.serve(port=port)
    except Exception as e:
        print(f"Failed to start {full_mod}: {e}")