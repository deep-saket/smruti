# python
import faiss
import numpy as np
from common import BaseComponent

class FAISSIndexManager(BaseComponent):
    """
    Simple wrapper around a FAISS IndexFlatIP (inner-product) index,
    storing string IDs in parallel.
    """
    def __init__(self, dim: int):
        self.dim = dim
        # use inner-product for cosine if vectors are normalized
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, vec: np.ndarray, label: str):
        """
        vecs: (N, dim) float32, normalized
        labels: length-N list of unique IDs
        """
        assert vec.dtype == np.float32 and vec.shape[-1] == self.dim
        self.index.add(vec)
        self.ids.append(label)

    def search(self, query: np.ndarray, top_k: int = 1):
        """
        query: (1, dim) float32, normalized
        returns: list of (label, score)
        """
        assert query.dtype == np.float32 and query.shape[1] == self.dim
        D, I = self.index.search(query, top_k)  # D: scores, I: indices
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.ids[idx], float(score)))
        return results

    def reset(self):
        """Clear the index and labels."""
        self.index.reset()
        self.ids.clear()