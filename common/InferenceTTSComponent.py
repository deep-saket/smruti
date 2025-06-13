# common/InferenceTTSComponent.py

from common.BaseComponent import BaseComponent
from abc import abstractmethod
import numpy as np

class InferenceTTSComponent(BaseComponent):
    """
    Base class for all TTS inference components.

    Subclasses must implement:
        - infer(text: str, **kwargs) -> np.ndarray

    Where:
      - text: str, the input text to synthesize.
      - kwargs: synthesis parameters (e.g., speaker, speed).
    """
    @abstractmethod
    def infer(self, text: str, **kwargs) -> np.ndarray:
        """
        Generate an audio waveform from the given text.

        Args:
            text: The input text to convert to speech.
            **kwargs: Optional synthesis parameters.

        Returns:
            np.ndarray: The generated audio samples as a 1-D float32 array.
        """
        raise NotImplementedError("Subclasses must implement infer()")