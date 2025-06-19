import importlib
from config.loader import settings

# Iterate over each MCP service defined under the top-level 'mcp_util' key in settings.yaml
for name, cfg in settings["mcp"].items():
    module = cfg.get("module")
    port   = cfg.get("port")
    if not module or not port:
        print(f"Skipping '{name}': module or port missing in settings.")
        continue

    full_mod = f"mcp_util.servers.{module}"
    # try:
    if True:
        mod = importlib.import_module(full_mod)
        mcp = getattr(mod, "mcp", None)
        if not mcp:
            print(f"Module {full_mod} has no 'mcp_util' instance.")
            continue

        print(f"Starting MCP server '{name}' on port {port}...")
        mcp.run(
                transport="streamable-http",
                host="0.0.0.0",
                port=port
            )
   # except Exception as e:
    #    print(f"Failed to start {full_mod}: {e}")