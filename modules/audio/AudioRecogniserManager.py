import os
import uuid
import numpy as np
from torch.nn.functional import normalize

from modules.db.FAISSIndexManager import FAISSIndexManager
from common import BaseComponent

class AudioRecogniserManager(BaseComponent):
    """
    Uses a FAISSIndexManager plus an embedding component to enroll,
    verify speakers, and persist/load prototypes and metadata to/from disk.
    """
    def __init__(
        self,
        embedding_component,
        embedding_dim: int,
        db_path: str = "audio_embeddings.npz",
    ):
        self.embedder = embedding_component
        self.embedding_dim = embedding_dim
        self.faiss = FAISSIndexManager(dim=embedding_dim)
        self.db_path = db_path
        # in-memory maps
        self.db: dict[str, np.ndarray] = {}           # speaker_id -> prototype
        self.metadata: dict[str, str] = {}            # speaker_id -> name
        # load on startup
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if os.path.exists(self.db_path):
            print(self.db_path, "##########")
            self._load_database()

    def _load_database(self):
        """Load saved embeddings, labels, names; add to FAISS index."""
        data = np.load(self.db_path, allow_pickle=True)
        labels = data["labels"].tolist()               # list of speaker_ids
        names = data["names"].tolist()                 # parallel list of names
        embs = data["embeddings"]                      # shape (N, dim)
        for sid, name, vec in zip(labels, names, embs):
            self.db[sid] = vec
            self.metadata[sid] = name
            self.faiss.add(vec.astype(np.float32), sid)

    def _save_database(self):
        """Persist current db to a .npz file (labels, names, embeddings)."""
        labels = list(self.db.keys())
        names = [self.metadata[sid] for sid in labels]
        if labels:
            embs = np.stack([self.db[sid] for sid in labels], axis=0)
            embs = embs.astype(np.float32)
        else:
            # empty embeddings array with known embedding_dim
            embs = np.empty((0, int(self.embedding_dim)), dtype=np.float32)

        np.savez(
            self.db_path,
            embeddings=embs,
            labels=np.array(labels, dtype=object),
            names=np.array(names, dtype=object)
        )

    def recreate_database(self, remove_file: bool = True):
        """Delete any existing DB file and recreate an empty FAISS index and metadata.

        Args:
            remove_file: if True, remove the persisted .npz file from disk.
        """
        # reset in-memory structures
        self.faiss.reset()
        self.db.clear()
        self.metadata.clear()
        # optionally remove file
        try:
            if remove_file and os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            self.logger.exception("Failed to remove existing audio DB file")
        # persist an empty DB
        self._save_database()

    def enroll(self, name: str, wavs: list[np.ndarray], speaker_id: str = None) -> str:
        """
        Compute prototype from wavs, normalize, generate or use speaker_id,
        then add to FAISS, store metadata, and persist.
        Returns the speaker_id used.
        """
        if speaker_id is None:
            speaker_id = uuid.uuid4().hex
        # compute per-wav embeddings
        embs = [self.embedder.embed(w) for w in wavs]
        proto = np.mean(np.stack(embs, axis=0), axis=0)
        # ensure float32 dtype and stable normalisation
        proto = proto.astype(np.float32)
        norm = np.linalg.norm(proto).astype(np.float32)
        if norm == 0.0:
            # avoid division by zero; keep as-is
            normalized = proto
        else:
            normalized = proto / (norm + 1e-12)

        # add to FAISS and in-memory structures (pass speaker_id as a string)
        self.faiss.add(normalized.reshape(1, -1), speaker_id)
        self.db[speaker_id] = proto
        self.metadata[speaker_id] = name
        # persist full database
        self._save_database()
        return speaker_id

    def verify(self, wav: np.ndarray, threshold: float = 0.56):
        """
        Embed & normalize test wav, search top-1 in FAISS.
        Return (speaker_id or None, name or None, score).
        """
        emb = self.embedder.embed(wav).astype(np.float32)

        results = self.faiss.search(emb.reshape(1, -1), top_k=1)
        if not results:
            return None, None, 0.0
        sid, score = results[0]
        if score < threshold:
            return None, None, score
        return sid, self.metadata.get(sid), score