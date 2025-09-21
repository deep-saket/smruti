#!/usr/bin/env python3
"""
Compute a speaker embedding suitable for SpeechT5 TTS from an audio file
and save it as a NumPy .npy file.

Usage:
    python helper_apps/get_tts_embedding.py /path/to/input.wav /path/to/output_embedding.npy

If output path is omitted, writes `config/tts_speaker_embedding.npy` inside PROJECT_ROOT.

Dependencies:
    pip install torch torchaudio speechbrain numpy

This script is resilient to multi-channel input and different sample rates.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def load_audio_mono(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    # Convert to mono by averaging channels if necessary
    if wav.ndim > 1 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    # Normalize to -1..1
    wav = wav / (wav.abs().max() + 1e-9)
    # Expected shape for speechbrain encode_batch: (batch, samples) or (batch, channels, samples)
    # We'll return shape (1, samples)
    if wav.ndim == 2 and wav.size(0) == 1:
        return wav.squeeze(0)
    return wav.squeeze(0)


def compute_embedding(wav_tensor: torch.Tensor, device: str | None = None) -> np.ndarray:
    """
    Compute speaker embedding using SpeechBrain's `spkrec-xvect-voxceleb` encoder.

    Args:
        wav_tensor: 1-D torch.Tensor of audio samples (float32, -1..1) or shape (channels, samples)
        device: 'cpu' or 'cuda' or None to auto-detect

    Returns:
        numpy array of shape (1, 512) dtype float32
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_opts = {"device": device}

    # Load encoder (this will download weights the first time)
    spk_enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts=run_opts)

    # speechbrain expects (batch,) list or 2D tensor (batch, samples) or (batch, channels, samples)
    if isinstance(wav_tensor, torch.Tensor):
        tensor = wav_tensor.unsqueeze(0) if wav_tensor.dim() == 1 else wav_tensor.unsqueeze(0)
    else:
        tensor = torch.from_numpy(wav_tensor).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        emb = spk_enc.encode_batch(tensor)

    emb = emb.squeeze()
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)

    emb = emb.cpu().numpy().astype(np.float32)

    # Ensure final shape (1, 512)
    if emb.ndim == 2 and emb.shape[1] != 512:
        emb = emb.reshape(1, -1)
    return emb


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute and save a TTS speaker embedding from an audio file")
    p.add_argument("input", help="Input audio file (wav or other format supported by torchaudio)")
    p.add_argument("output", nargs="?", help="Output .npy path to save embedding")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device to use (default auto)" )
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate (default 16000)")
    args = p.parse_args(argv)

    inp = args.input
    out = args.output
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(inp):
        print(f"ERROR: input file not found: {inp}", file=sys.stderr)
        return 2

    if out is None:
        project_root = os.environ.get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cfg_dir = os.path.join(project_root, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        out = os.path.join(cfg_dir, "tts_speaker_embedding.npy")

    print(f"Loading audio: {inp}")
    wav = load_audio_mono(inp, target_sr=args.sr)

    print(f"Computing embedding on device: {device}")
    emb = compute_embedding(wav, device=device)

    # Ensure parent directory for explicit output exists
    parent = os.path.dirname(out) or '.'
    os.makedirs(parent, exist_ok=True)

    # Save embedding
    print(f"Saving embedding to: {out}")
    np.save(out, emb)
    print("Done. Embedding shape:", emb.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
