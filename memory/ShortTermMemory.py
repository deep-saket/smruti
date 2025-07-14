import os
import time
from datetime import datetime
import json
import threading
import numpy as np
import faiss
from dateutil import parser as date_parser
from datetime import datetime, timedelta
from common.MemoryComponent import MemoryComponent
from common import InferenceEmbeddingComponent

class ShortTermMemory(MemoryComponent):
    """
    Persistent in-memory semantic memory using text embeddings + FAISS,
    with metadata stored as JSON for easy visualization.

    - Embeddings & FAISS index saved asynchronously to disk.
    - Metadata (text + timestamp) saved as JSON alongside the index.
    """
    def __init__(
        self,
        embedder: InferenceEmbeddingComponent,
        memory_dir: str,
        capacity: int = 100,
        index_filename: str = "short_term.index",
        meta_filename: str = "short_term_meta.json"
    ):
        # Ensure memory directory exists
        os.makedirs(memory_dir, exist_ok=True)
        self.index_path = os.path.join(memory_dir, index_filename)
        self.meta_path = os.path.join(memory_dir, meta_filename)

        self.embedder = embedder
        self.capacity = capacity
        self._lock = threading.Lock()

        # Initialize or load FAISS index
        dim = self.embedder.model.get_sentence_embedding_dimension()
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

        # Initialize or load metadata JSON
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []  # list of {"text": str, "timestamp": float}

    def add(self, text: str, ts: float = None):
        """
        Embed text and add to FAISS index, evicting oldest if over capacity.
        """
        ts = datetime.now()
        emb = self.embedder.embed(text)
        with self._lock:
            self.index.add(np.expand_dims(emb, 0))
            self.metadata.append({"text": text, "timestamp": ts})
            # Enforce capacity
            if len(self.metadata) > self.capacity:
                self.metadata.pop(0)
                embs = np.stack([self.embedder.embed(item['text']) for item in self.metadata])
                self.index.reset()
                self.index.add(embs)
        threading.Thread(target=self._save, daemon=True).start()

    def get(self, query_text: str = None, date_str: str = None, k: int = 5) -> list[str]:
        """
        Retrieve entries by:
          - semantic similarity if `query_text` is provided,
          - or by date if `date_str` is provided (e.g. "yesterday"),
          - or all entries if neither is provided.
        """
        with self._lock:
            # Date-based query
            if date_str:
                dt = date_parser.parse(date_str, fuzzy=True)
                if "yesterday" in date_str.lower():
                    dt = datetime.now() - timedelta(days=1)
                start = datetime(dt.year, dt.month, dt.day).timestamp()
                end = start + 86400
                return [item['text'] for item in self.metadata if start <= item['timestamp'] < end]

            # Semantic query
            if query_text:
                q_emb = self.embedder.embed(query_text)
                _, I = self.index.search(np.expand_dims(q_emb, 0), k)
                return [self.metadata[idx]['text'] for idx in I[0] if idx < len(self.metadata)]

            # Return all
            return [item['text'] for item in self.metadata]

    def clear(self):
        """Reset the FAISS index and clear metadata."""
        dim = self.embedder.model.get_sentence_embedding_dimension()
        with self._lock:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata.clear()
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        """Persist FAISS index and metadata JSON to disk."""
        with self._lock:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            # Save metadata as JSON
            with open(self.meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
