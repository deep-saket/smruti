"""
Embedding class for speaker verification using SpeechBrain's ECAPA-TDNN.

This class wraps the SpeechBrain ``SpeakerRecognition`` model based on
the ECAPA-TDNN architecture.  It exposes an ``embed`` method that
produces a fixed-size, L2-normalised speaker embedding from a mono
waveform at 16 kHz.  The embedding dimension is inferred on
initialisation and stored in ``embedding_dim``.

Dependencies:
    pip install speechbrain torch
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import normalize as l2_normalize
from common import InferenceEmbeddingComponent

try:
    # SpeechBrain provides the ECAPA-TDNN model via SpeakerRecognition
    from speechbrain.pretrained import SpeakerRecognition  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ECAPAEmbeddingInfer requires the speechbrain package. "
        "Install it with `pip install speechbrain`."
    ) from e


class ECAPAEmbeddingInfer(InferenceEmbeddingComponent):
    """Compute speaker embeddings using the ECAPA-TDNN model from SpeechBrain.

    Args:
        model_name: HuggingFace identifier for the pretrained ECAPA model.
        sample_rate: Expected audio sample rate (Hz).
        device: Torch device identifier (e.g. "cpu" or "cuda").  Defaults
            to auto-detection.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        sample_rate: int = 16_000,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the pretrained ECAPA speaker recognition model
        self._model = SpeakerRecognition.from_hparams(
            source=model_name,
            run_opts={"device": self.device},
        )
        # Warm up to determine embedding dimension
        dummy = torch.zeros(int(sample_rate * 1.0), dtype=torch.float32)
        emb = self.embed(dummy.numpy())
        self.embedding_dim = emb.shape[0]

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Return a unit-normalised embedding for the given waveform.

        Args:
            wav: 1-D numpy array of floating point audio samples.

        Returns:
            A numpy array of shape (embedding_dim,) containing the L2-normalised
            speaker embedding.
        """
        if wav.ndim != 1:
            raise ValueError("Expected a mono 1-D waveform for embedding")
        # SpeechBrain expects torch tensors in shape (batch, time)
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        # Compute embedding; SpeakerRecognition.encode_batch returns shape (batch, dim)
        with torch.no_grad():
            emb_tensor = self._model.encode_batch(wav_tensor)
        # Remove batch dimension and normalise
        emb_tensor = emb_tensor.squeeze(0)
        emb_tensor = l2_normalize(emb_tensor, p=2, dim=-1)
        return emb_tensor.cpu().numpy().astype(np.float32)