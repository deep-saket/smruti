MCP servers

This directory contains small FastMCP-based microservices (MCP servers) that expose simple tools the agent can call.

Available services

- weather_service - existing: returns weather information using OpenWeatherMap (configured in `config/files/settings.yml`).
- time_service - returns server time in ISO format.
- calc_service - safe arithmetic evaluator for simple expressions.
- translate_service - stub translation service (replace with real API if needed).
- notes_service - simple local notes storage (add/list notes persisted to `.cache/notes.json`).
- news_service - stub news/headlines provider.

Each service has a short README with usage examples. To start services for local development, set `PROJECT_ROOT` in your environment (the config loader requires it), then run the desired service module directly, for example:

```bash
export PROJECT_ROOT=$(pwd)
python mcp_util/servers/time_service.py
```

Or use `mcp_util/servers/init_servers.py` to start all services listed in `config/files/settings.yml`:

```bash
export PROJECT_ROOT=$(pwd)
python mcp_util/servers/init_servers.py
```

Notes

- The translate and news services are stubs: they return pre-defined or echo responses. Replace their implementations with real APIs if you require production data.
- The `calc_service` uses a restricted AST-based evaluator to avoid executing arbitrary code. It only supports basic arithmetic operators.

