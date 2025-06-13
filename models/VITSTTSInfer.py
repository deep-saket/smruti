import numpy as np
from TTS.api import TTS
from common import InferenceTTSComponent

class VITSTTSInfer(InferenceTTSComponent):
    """
    Inference component for the VITS TTS model (LJSpeech).
    """
    def __init__(self,
                 model_name: str = "tts_models/en/ljspeech/vits",
                 speaker: str = None,
                 gpu: bool = False):
        """
        Args:
            model_name: HF model ID for the TTS model.
            speaker: Optional speaker ID for multi-speaker models.
            gpu: Whether to use GPU (if supported).
        """
        # Initialize Coqui TTS engine
        self.tts = TTS(model_name=model_name,
                       progress_bar=False,
                       gpu=gpu)
        self.speaker = speaker

    def infer(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize the given text to audio.

        Args:
            text: The text to speak.
            **kwargs: Optional synthesis params (e.g., speaker).

        Returns:
            np.ndarray: Synthesized audio waveform (float32).
        """
        # Generate waveform; TTS.tts returns numpy array
        wav = self.tts.tts(text,
                           speaker=self.speaker
                           )[0]
        # Ensure float32 output
        return np.array(wav, dtype=np.float32)
