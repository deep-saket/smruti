from common.BaseComponent import BaseComponent
from abc import abstractmethod
import numpy as np

class InferenceTextEmbeddingComponent(BaseComponent):
    """
    Base class for all text embedding components.

    Subclasses must implement:
        - embed(text: str) -> np.ndarray
    """
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Produce a fixed‐size embedding for the given text.
        """
        raise NotImplementedError("Subclasses must implement embed()")