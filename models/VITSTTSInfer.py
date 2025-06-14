import numpy as np
import torch
from TTS.api import TTS
from common.InferenceTTSComponent import InferenceTTSComponent

class VITSTTSInfer(InferenceTTSComponent):
    """
    Inference component for the VITS TTS model (LJSpeech), with device support.
    """
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/vits",
        speaker: str = None,
        device='cpu'
    ):
        """
        Args:
            model_name: HF model ID for the TTS model.
            speaker: Optional speaker ID for multi-speaker models.
            device: 'cuda', 'cpu', 'mps', or torch.device.
        """
        # Resolve device and GPU flag
        if isinstance(device, torch.device):
            dev = device
        else:
            d = device.lower()
            if d == 'mps' and torch.backends.mps.is_available():
                self.logger.warning("MPS not supported by Coqui TTS; falling back to CPU.")
                dev = torch.device('cpu')
                use_gpu = False
            elif d == 'cuda' and torch.cuda.is_available():
                dev = torch.device('cuda')
                use_gpu = True
            else:
                dev = torch.device('cpu')
                use_gpu = False

        self.device = dev
        self.tts = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=use_gpu,
        )
        self.speaker = speaker

    def infer(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize the given text to audio.

        Args:
            text: The text to speak.
            **kwargs: Optional synthesis parameters.

        Returns:
            np.ndarray: Synthesized audio waveform (float32).
        """
        # Coqui TTS may return (waveform, sample_rate) or just waveform array
        out = self.tts.tts(text, speaker=self.speaker)
        if isinstance(out, tuple):
            wav, _sr = out  # unpack waveform, ignore sample rate
        else:
            wav = out       # already the full waveform

        # Convert to float32 numpy array
        return np.array(wav, dtype=np.float32)