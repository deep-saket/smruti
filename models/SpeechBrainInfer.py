from common import AudioToAudioComponent
import numpy as np
import torch
from speechbrain.pretrained import SpectralMaskEnhancement


class SpeechBrainInfer(AudioToAudioComponent):
    """
    Audio-to-audio component using SpeechBrain's SpectralMaskEnhancement models for speech denoising.

    Available pretrained models (Hugging Face):
    -------------------------------------------------------------------------
    1. 'speechbrain/metricgan-plus-voicebank'
       - Model Type: MetricGAN+
       - Parameters: ~2-3 Million
       - PESQ Improvement: ~1.8 -> ~3.0
       - STOI: Typically improves from ~0.7 (noisy) to ~0.92+
       - Purpose: General speech enhancement targeting PESQ.
       - Sampling Rate: 16kHz recommended.

    2. 'speechbrain/sepformer-whamr-enhancement'
       - Model Type: SepFormer (Dual-path Transformer)
       - Parameters: ~26 Million
       - PESQ Improvement: ~1.9 -> ~3.5+
       - STOI: Typically improves to ~0.95+
       - Purpose: High-quality speech separation and enhancement in extreme noise.
       - Sampling Rate: 8kHz or 16kHz (depends on dataset).

    3. 'speechbrain/denoise-tasnet'
       - Model Type: Conv-TasNet
       - Parameters: ~5-10 Million
       - PESQ Improvement: ~2.0 -> ~3.2
       - STOI: ~0.90+
       - Purpose: Real-time friendly, clean denoising.
       - Sampling Rate: 16kHz.

Usage:
------
    >>> denoiser = SpeechBrainInfer(model_source="speechbrain/metricgan-plus-voicebank")
    >>> denoised_audio = denoiser.process(audio_array)

Notes:
------
- Input audio should be **mono, 16kHz, float32, 1D NumPy array**.
- Output is **denoised 1D NumPy array**, same shape and dtype.

Typical Applications:
----------------------
- Speech enhancement for ASR pipelines.
- Denoising recorded speech.
- Preprocessing noisy datasets.
- Improving intelligibility in low-SNR recordings.
    """

    def __init__(self, model_name="speechbrain/metricgan-plus-voicebank"):
        """
        Args:
            model_name (str): Hugging Face repo ID for SpeechBrain model.
        """
        self.model = SpectralMaskEnhancement.from_hparams(source=model_name, savedir="tmpdir_speechbrain_denoise")
        self.model.device = "cpu"  # Explicitly use CPU for macOS

    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply SpeechBrain MetricGAN+ enhancement.

        Args:
            audio (np.ndarray): 1D float32 array, mono, 16kHz, any amplitude range.

        Returns:
            np.ndarray: 1D float32 array of denoised audio, normalized [-1, 1].
        """
        # Ensure shape (samples,)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze()

        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize dynamically to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        assert audio.ndim == 1, "Audio must be 1D mono."
        assert np.max(np.abs(audio)) <= 1.0, "Audio must be normalized to [-1, 1]"

        # Convert to torch tensor with batch dimension
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # Denoise
        lengths = torch.tensor([1.0], dtype=torch.float32)
        enhanced_audio = self.model.enhance_batch(audio_tensor, lengths=lengths)

        # Convert back to numpy
        denoised = enhanced_audio.squeeze(0).cpu().numpy()
        return denoised.astype(np.float32)