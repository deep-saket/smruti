import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from common import InferenceEmbeddingComponent

class SentenceEmbedderInfer(InferenceEmbeddingComponent):
    """
    Wraps a SentenceTransformer model to produce text embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | torch.device = "cpu"):
        if isinstance(device, str):
            dev = device.lower()
            if dev not in ["cpu", "cuda", "mps"]:
                raise ValueError(f"Unsupported device string: {device}")
            if dev == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS device requested but not available")
            if dev == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available")
            self.device = torch.device(dev)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError("`device` must be a str or torch.device")

        self.model = SentenceTransformer(model_name, device=str(self.device))


    def embed(self, model_input: str) -> np.ndarray:
            """
            Encode the input text into a numpy float32 embedding.
            """
            emb = self.model.encode([model_input])[0]
            return emb.astype(np.float32)