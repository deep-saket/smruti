import os
import json
import threading
from datetime import datetime
from common.MemoryComponent import MemoryComponent

class SemanticMemory(MemoryComponent):
    """Stores general factual knowledge, persisted to JSON asynchronously."""

    def __init__(self, memory_dir: str = "memory", filename: str = "semantic_memory.json"):
        os.makedirs(memory_dir, exist_ok=True)
        self.meta_path = os.path.join(memory_dir, filename)
        self._lock = threading.Lock()
        # Load existing facts if present
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.facts = json.load(f)
        else:
            self.facts = {}

    def add(self, key: str, fact: str):
        """Add or update a fact and persist asynchronously."""
        with self._lock:
            self.facts[key] = {"value": fact, "timestamp": datetime.now()}
        threading.Thread(target=self._save, daemon=True).start()

    def get(self, key: str):
        """Retrieve the fact for the given key."""
        return self.facts.get(key)

    def clear(self):
        """Clear all stored facts and persist asynchronously."""
        with self._lock:
            self.facts.clear()
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        """Persist all facts to JSON file."""
        with self._lock:
            with open(self.meta_path, 'w') as f:
                json.dump(self.facts, f, indent=2)
