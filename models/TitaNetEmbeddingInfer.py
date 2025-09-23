# models/TitaNetEmbeddingInfer.py
"""
Embedding class for speaker verification using NVIDIA's TitaNet model.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn.functional import normalize
from common import InferenceEmbeddingComponent
import nemo.collections.asr as nemo_asr

class TitaNetEmbeddingInfer(InferenceEmbeddingComponent):
    """
    Speaker embedding inference using NVIDIA TitaNet.
    """

    def __init__(
        self,
        model_name: str = "nvidia/speakerverification_en_titanet_large",
        sample_rate: int = 16_000,
        sample_duration: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

        # Load the pretrained TitaNet model
        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name
        )

        # Warm up the model and determine embedding dimension
        dummy_waveform = torch.zeros(
            int(sample_rate * sample_duration), dtype=torch.float32
        )
        with torch.no_grad():
            emb = self._model.get_embedding(dummy_waveform, sr=sample_rate)
        self.embedding_dim = int(emb.shape[0])

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """
        Compute a speaker embedding for the given waveform.
        Args:
            wav: 1‑D NumPy array of float32 samples at sample_rate.
        Returns:
            A L2‑normalized embedding as a 1‑D NumPy array of length embedding_dim.
        """
        tensor = torch.from_numpy(wav.astype(np.float32))
        with torch.no_grad():
            emb = self._model.get_embedding(tensor, sr=self.sample_rate)
        emb_tensor = torch.from_numpy(emb)
        emb_tensor = normalize(emb_tensor, p=2, dim=-1)
        return emb_tensor.cpu().numpy()