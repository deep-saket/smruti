import numpy as np
from common.BaseComponent import BaseComponent
from abc import abstractmethod

class InferenceAudioComponent(BaseComponent):
    """
    Base class for all audio inference components.

    Subclasses must implement:
        - infer(audio_data: np.ndarray) -> str
    """
    @abstractmethod
    def infer(self, audio_data: np.ndarray) -> str:
        """
        Run inference on the provided audio data.

        Args:
            audio_data: 1D NumPy array of audio samples.

        Returns:
            str: The model’s generated text (transcription).
        """
        raise NotImplementedError("Subclasses must implement infer()")
