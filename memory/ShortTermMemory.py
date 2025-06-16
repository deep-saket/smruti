import os
import time
import pickle
import threading
import numpy as np
import faiss
from dateutil import parser as date_parser
from datetime import datetime, timedelta
from common.MemoryComponent import MemoryComponent
from models import SentenceEmbedderInfer

class ShortTermMemory(MemoryComponent):
    """
    Persistent in-memory semantic memory using text embeddings + FAISS.

    - Embeddings & FAISS index saved asynchronously to disk.
    - Metadata (text + timestamp) pickled alongside the index.
    """
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        capacity: int = 100,
        index_path: str = "short_term.index",
        meta_path: str = "short_term_meta.pkl"
    ):
        self.embedder   = SentenceEmbedderInfer(embed_model_name)
        self.capacity   = capacity
        self.index_path = index_path
        self.meta_path  = meta_path
        self._lock      = threading.Lock()

        dim = self.embedder.model.get_sentence_embedding_dimension()
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index    = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index    = faiss.IndexFlatL2(dim)
            self.metadata = []  # list of (text: str, ts: float)

    def add(self, text: str, ts: float = None):
        """
        Embed text and add to FAISS index, evicting oldest if over capacity.
        ts defaults to current time.
        """
        ts = ts or time.time()
        emb = self.embedder.embed(text)
        with self._lock:
            self.index.add(np.expand_dims(emb, 0))
            self.metadata.append((text, ts))
            if len(self.metadata) > self.capacity:
                self.metadata.pop(0)
                embs = np.stack([self.embedder.embed(t) for t, _ in self.metadata])
                self.index.reset()
                self.index.add(embs)
        threading.Thread(target=self._save, daemon=True).start()

    def get(self, query_text: str = None, date_str: str = None, k: int = 5) -> list[str]:
        """
        Retrieve entries by:
          - semantic similarity if query_text is provided,
          - or by date if date_str is provided (e.g. "yesterday", "2025-06-14"),
          - or all entries if neither is provided.

        If both query_text and date_str are given, date_str takes precedence.
        """
        with self._lock:
            # Date-based query
            if date_str:
                dt = date_parser.parse(date_str, fuzzy=True)
                # handle "yesterday" etc.
                if "yesterday" in date_str.lower():
                    dt = datetime.now() - timedelta(days=1)
                start = datetime(dt.year, dt.month, dt.day).timestamp()
                end   = start + 86400
                return [text for text, ts in self.metadata if start <= ts < end]

            # Semantic query
            if query_text:
                q_emb = self.embedder.embed(query_text)
                _, I  = self.index.search(np.expand_dims(q_emb, 0), k)
                return [self.metadata[idx][0] for idx in I[0] if idx < len(self.metadata)]

            # Return all
            return [text for text, _ in self.metadata]

    def clear(self):
        """Reset the FAISS index and clear metadata, then persist asynchronously."""
        dim = self.embedder.model.get_sentence_embedding_dimension()
        with self._lock:
            self.index    = faiss.IndexFlatL2(dim)
            self.metadata.clear()
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        """Persist FAISS index and metadata to disk."""
        with self._lock:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)