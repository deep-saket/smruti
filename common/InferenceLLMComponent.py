# common/InferenceLLMComponent.py

from common.BaseComponent import BaseComponent
from abc import abstractmethod

class InferenceLLMComponent(BaseComponent):
    """
    Base class for all LLM inference components.

    Subclasses must implement:
        - infer(prompt: str, **kwargs) -> str

    Where:
      - prompt: str, the input text or instruction.
      - kwargs: generation parameters (e.g. max_length, temperature).
    """

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> str:
        """
        Generate text given a prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Optional generation parameters.

        Returns:
            str: The generated output from the model.
        """
        raise NotImplementedError("Subclasses must implement infer()")