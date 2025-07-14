import json
from pathlib import Path
from common.MemoryComponent import MemoryComponent
from datetime import datetime
import os

class PersonMemory(MemoryComponent):
    """Stores and retrieves user-specific details."""

    def __init__(self, memory_dir, db_path="person_memory.json"):
        os.makedirs(memory_dir, exist_ok=True)
        db_path = os.path.join(memory_dir, db_path)
        self.db_path = Path(db_path)
        self.users = self._load()

    def add(self, user_id, details):
        """Add or update user details."""
        if user_id not in self.users:
            self.users[user_id] = details
        else:
            self.users[user_id].update(details)
        self.users[user_id]['last_interaction'] = datetime.now().isoformat()
        self._save()

    def get(self, user_id):
        """Retrieve user details."""
        return self.users.get(user_id)

    def clear(self):
        """Clear all user profiles."""
        self.users.clear()
        self._save()

    def _load(self):
        if self.db_path.exists():
            return json.loads(self.db_path.read_text())
        return {}

    def _save(self):
        self.db_path.write_text(json.dumps(self.users, indent=4))