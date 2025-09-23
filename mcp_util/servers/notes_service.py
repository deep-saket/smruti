from fastmcp import FastMCP
from config.loader import settings, project_root
import json
import os
from typing import Dict, Any

mcp = FastMCP("NotesService", stateless_http=True, json_response=True)

NOTES_PATH = os.path.join(project_root, ".cache", "notes.json")

def _load_notes() -> Dict[str, Any]:
    if not os.path.exists(NOTES_PATH):
        return {}
    try:
        with open(NOTES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_notes(notes: Dict[str, Any]):
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=2)

@mcp.tool()
def add_note(title: str, content: str, user: str = "default") -> dict:
    notes = _load_notes()
    user_notes = notes.setdefault(user, [])
    entry = {"title": title, "content": content}
    user_notes.append(entry)
    _save_notes(notes)
    return {"status": "ok", "note": entry}

@mcp.tool()
def list_notes(user: str = "default") -> dict:
    notes = _load_notes()
    return {"notes": notes.get(user, [])}

if __name__ == "__main__":
    cfg = settings.get('mcp', {}).get('notes', {})
    port = cfg.get('port', 8004)
    print(f"🚀 Starting NotesService on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)

