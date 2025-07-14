from common import InferenceEmbeddingComponent
import torch
from torch.nn.functional import normalize
from speechbrain.pretrained import SpeakerRecognition
import numpy as np

class SpeakerEmbeddingInfer(InferenceEmbeddingComponent):
    """
    Embedding for speaker verification via ECAPA-TDNN.
    Captures `embd_dim` so you can initialize FAISSIndexManager(dim=embd_dim).
    """
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        sample_rate: int = 16000,
        sample_duration: float = 1.0,
    ):
        # load the pretrained ECAPA-TDNN
        self.recognizer = SpeakerRecognition.from_hparams(source=model_name)
        # determine embedding dimension by running a dummy batch
        dummy_wav = torch.zeros(1, int(sample_rate * sample_duration))
        with torch.no_grad():
            emb = self.recognizer.encode_batch(dummy_wav)  # [1, D]
        self.embedding_dim = emb.size(-1)

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """
        wav: 1D float32 numpy array @16kHz
        returns: L2-normalized 1D embedding of length `self.embd_dim`
        """
        tensor = torch.from_numpy(wav).unsqueeze(0)           # [1, time]
        emb = self.recognizer.encode_batch(tensor)            # [1, D]
        emb = normalize(emb, p=2, dim=-1).squeeze(0)          # [D]
        return emb.cpu().numpy()