#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import sounddevice as sd
import numpy.linalg as LA
from config.loader import settings
from models.ModelManager import ModelManager
from modules.audio.AudioRecogniserManager import AudioRecogniserManager

def record_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    print(f"Recording for {seconds:.1f} seconds… speak now!")
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    x = x.flatten()
    # Normalise to [-1, 1]
    norm = np.max(np.abs(x)) or 1.0
    return (x / norm).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    project_root = os.environ.get("PROJECT_ROOT") or os.getcwd()
    db_path = settings['db']['audio_recogniser'].format(project_root=project_root)

    # Initialise models (loads embedder)
    ModelManager()
    embedder = getattr(ModelManager, "speaker_embedder")
    embedding_dim = embedder.embedding_dim
    sample_rate = getattr(embedder, "sample_rate", 16000)

    # Check existing DB for dimension mismatch
    if os.path.exists(db_path):
        data = np.load(db_path, allow_pickle=True)
        old_embs = data['embeddings']
        if old_embs.shape[1] != embedding_dim:
            print(f"Existing DB dimension {old_embs.shape[1]} != new {embedding_dim}; removing old DB…")
            os.remove(db_path)

    recogniser = AudioRecogniserManager(embedder, embedding_dim, db_path=db_path)
    wav = record_audio(args.duration, sr=sample_rate)
    sid = recogniser.enroll(args.name, [wav]) if args.replace else recogniser.enroll(args.name, [wav], speaker_id=None)
    print(f"Enrolled speaker '{args.name}' with ID {sid}")
    print(f"Embedding database saved at {db_path}")

if __name__ == "__main__":
    main()