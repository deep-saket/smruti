import os
import json
import threading
from common import MemoryComponent
from datetime import datetime

class EpisodicMemory(MemoryComponent):
    """Stores events with timestamps and contextual metadata, persisted to JSON asynchronously."""

    def __init__(self, memory_dir: str, filename: str = "episodic_memory.json"):
        os.makedirs(memory_dir, exist_ok=True)
        self.meta_path = os.path.join(memory_dir, filename)
        self._lock = threading.Lock()
        # Load existing events if present
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.events = json.load(f)
        else:
            self.events = []  # list of dicts: {event, timestamp, user_id}

    def add(self, event: str, user_id: str = None):
        timestamp = datetime.now()
        self.events.append({"event": event, "timestamp": timestamp, "user_id": user_id})
        threading.Thread(target=self._save, daemon=True).start()

    def get(self, user_id: str = None):
        if user_id:
            return [e for e in self.events if e.get("user_id") == user_id]
        return self.events

    def clear(self):
        with self._lock:
            self.events.clear()
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        """Persist events to JSON file."""
        with self._lock:
            with open(self.meta_path, 'w') as f:
                json.dump(self.events, f, indent=2)
