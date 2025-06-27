from common.BaseComponent import BaseComponent
from abc import abstractmethod

class InferenceNERComponent(BaseComponent):
    """
    Base class for all NER inference components.

    Subclasses must implement:
        - infer(text: str, labels: list[str]) -> list[dict]

    Where:
      - text: str, the input text to extract entities from.
      - labels: list of str, the entity types to extract (e.g., ["Person", "Location"]).
    """

    @abstractmethod
    def infer(self, text: str, labels: list[str]) -> list[dict]:
        """
        Extract named entities from the input text.

        Args:
            text: The input string.
            labels: A list of entity type labels.

        Returns:
            A list of dictionaries with keys like {'entity': ..., 'type': ...}.
        """
        raise NotImplementedError("Subclasses must implement infer()")