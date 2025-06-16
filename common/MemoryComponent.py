from common.BaseComponent import BaseComponent
from abc import abstractmethod

class MemoryComponent(BaseComponent):
    """
    Base class for all memory modules.

    Defines a consistent interface:
      - add(...)
      - get(...)
      - clear()
    """
    @abstractmethod
    def add(self, *args, **kwargs):
        """Add an entry to memory."""
        raise NotImplementedError()

    @abstractmethod
    def get(self, *args, **kwargs):
        """Retrieve entries from memory."""
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        """Clear all stored entries."""
        raise NotImplementedError()