import os
import json
import threading
from datetime import datetime
from common.MemoryComponent import MemoryComponent

class ProceduralMemory(MemoryComponent):
    """Stores procedural knowledge (how-to instructions), persisted to JSON asynchronously."""

    def __init__(self, memory_dir: str = "memory", filename: str = "procedural_memory.json"):
        os.makedirs(memory_dir, exist_ok=True)
        self.meta_path = os.path.join(memory_dir, filename)
        self._lock = threading.Lock()
        # Load existing procedures if present
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.procedures = json.load(f)
        else:
            self.procedures = {}

    def add(self, task_name: str, steps: list):
        """Add or update a procedure and persist asynchronously."""
        with self._lock:
            self.procedures[task_name] = {"value": steps, "timestamp": datetime.now()}
        threading.Thread(target=self._save, daemon=True).start()

    def get(self, task_name: str):
        """Retrieve steps for the given task."""
        return self.procedures.get(task_name)

    def clear(self):
        """Clear all stored procedures and persist asynchronously."""
        with self._lock:
            self.procedures.clear()
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        """Persist all procedures to JSON file."""
        with self._lock:
            with open(self.meta_path, 'w') as f:
                json.dump(self.procedures, f, indent=2)
