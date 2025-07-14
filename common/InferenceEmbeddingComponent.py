from common.BaseComponent import BaseComponent
from abc import abstractmethod
import numpy as np

class InferenceEmbeddingComponent(BaseComponent):
    """
    Base class for all text embedding components.

    Subclasses must implement:
        - embed(text) -> np.ndarray
    """
    @abstractmethod
    def embed(self, model_input) -> np.ndarray:
        """
        Produce a fixed‐size embedding for the given text.
        """
        raise NotImplementedError("Subclasses must implement embed()")