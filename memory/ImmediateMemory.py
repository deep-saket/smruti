from collections import deque
from common import MemoryComponent

class ImmediateMemory(MemoryComponent):
    """
    Rolling buffer of the last N (speaker, text) tuples for immediate context.
    """
    def __init__(self, capacity: int = 6):
        self.buffer = deque(maxlen=capacity)

    def add(self, speaker: str, text: str):
        """Append a new turn (user or assistant) to the chat buffer."""
        self.buffer.append((speaker, text))

    def get(self):
        """Return the list of (speaker, text) in chronological order."""
        return list(self.buffer)

    def clear(self):
        """Empty the buffer."""
        self.buffer.clear()