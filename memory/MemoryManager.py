from common import BaseComponent
from memory.SemanticMemory import SemanticMemory
from memory.ShortTermMemory import ShortTermMemory
from memory.EpisodicMemory import EpisodicMemory
from memory.ProceduralMemory import ProceduralMemory
from memory.PersonMemory import PersonMemory
from models import ModelManager
from config.loader import settings


class MemoryManager(BaseComponent):
    """Orchestrates various memory components."""

    def __init__(self):
        self.short_term = ShortTermMemory(getattr(ModelManager, 'embedder'), settings['memory']['dir'])
        self.episodic = EpisodicMemory(settings['memory']['dir'])
        self.semantic = SemanticMemory(settings['memory']['dir'])
        self.procedural = ProceduralMemory(settings['memory']['dir'])
        self.person_memory = PersonMemory(settings['memory']['dir'])

    def clear_all(self):
        self.short_term.clear()
        self.episodic.clear()
        self.semantic.clear()
        self.procedural.clear()
        self.person_memory.clear()

    def fetch(self, memory_type: str, **kwargs):
        """
        Centralized method to fetch memories from various components.

        Args:
            memory_type (str): Type of memory to fetch from.
                               Options: 'short_term', 'episodic', 'semantic', 'procedural', 'person'
            kwargs: Arguments specific to each memory's `get()` method.

        Returns:
            Results from the specified memory's get() method.

        Raises:
            ValueError: If unsupported memory_type is given.
        """
        if memory_type == 'short_term':
            return self.short_term.get(**kwargs)
        elif memory_type == 'episodic':
            return self.episodic.get(**kwargs)
        elif memory_type == 'semantic':
            return self.semantic.get(**kwargs)
        elif memory_type == 'procedural':
            return self.procedural.get(**kwargs)
        elif memory_type == 'person':
            return self.person_memory.get(**kwargs)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # inside your MemoryManager class

    def hierarchical_fetch(self, user_id: str, query_text: str = None):
        """
        Try memories in this order:
          1. recent Short-Term
          2. last Episodic event
          3. Person profile
          4. Semantic fact (by key=query_text)
          5. Procedural steps (task_name=query_text)

        Returns:
            dict with keys 'memory_type' and 'data'
        """
        # 1) Short-Term
        recent = self.short_term.get(k=3)
        if recent:
            return {"memory_type": "short_term", "data": recent}

        # 2) Episodic
        episodes = self.episodic.get(user_id=user_id)
        if episodes:
            last_event = episodes[-1]["event"]
            return {"memory_type": "episodic", "data": last_event}

        # 3) Person profile
        profile = self.person_memory.get(user_id=user_id)
        if profile:
            return {"memory_type": "person", "data": profile}

        # 4) Semantic fact
        if query_text:
            fact = self.semantic.get(key=query_text)
            if fact:
                return {"memory_type": "semantic", "data": fact}

        # 5) Procedural
        if query_text:
            steps = self.procedural.get(task_name=query_text)
            if steps:
                return {"memory_type": "procedural", "data": steps}

        # Nothing found
        return {"memory_type": "none", "data": None}

    def put(self, memory_type: str, **kwargs):
        """
        Add entries to various memory modules.

        Args:
            memory_type (str): The memory type to put entries into.
                               Options: 'short_term', 'episodic', 'semantic', 'procedural', 'person'
            kwargs: Specific arguments required by each memory module's `add()` method.
        """
        if memory_type == 'short_term':
            self.short_term.add(kwargs.get('entry'))
        elif memory_type == 'episodic':
            self.episodic.add(event=kwargs.get('event'),
                              timestamp=kwargs.get('timestamp'),
                              user_id=kwargs.get('user_id'))
        elif memory_type == 'semantic':
            self.semantic.add(key=kwargs.get('key'),
                              fact=kwargs.get('fact'))
        elif memory_type == 'procedural':
            self.procedural.add(task_name=kwargs.get('task_name'),
                                steps=kwargs.get('steps'))
        elif memory_type == 'person':
            self.person_memory.add(user_id=kwargs.get('user_id'),
                                   details=kwargs.get('details'))
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")