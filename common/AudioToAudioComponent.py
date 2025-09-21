from common.BaseComponent import BaseComponent
from abc import abstractmethod
import numpy as np


class AudioToAudioComponent(BaseComponent):
    """
    Base class for all audio-to-audio processing components.

    Subclasses must implement:
        - process(audio: np.ndarray, **kwargs) -> np.ndarray

    Where:
      - audio: np.ndarray, the input audio waveform.
      - kwargs: additional parameters for processing.
    """

    @abstractmethod
    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process audio and return enhanced audio.

        Args:
            audio (np.ndarray): Input audio array (1D float32, typically normalized).
            **kwargs: Optional processing parameters.

        Returns:
            np.ndarray: The processed (denoised) audio array.
        """
        raise NotImplementedError("Subclasses must implement process()")