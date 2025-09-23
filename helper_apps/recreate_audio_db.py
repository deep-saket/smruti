"""
Recreate audio recogniser DB and optionally enroll a WAV file.

Usage:
  source setup.sh
  python helper_apps/recreate_audio_db.py            # recreate empty DB
  python helper_apps/recreate_audio_db.py --enroll path/to/file.wav --name "Saket"

Notes:
- This script will instantiate the real ModelManager and the configured
  speaker_embedder. Ensure you have run `source setup.sh` and installed
  dependencies (speechbrain, torch) before running.
"""

import os
import argparse
import numpy as np

from config.loader import settings
from models.ModelManager import ModelManager
from modules.audio.AudioRecogniserManager import AudioRecogniserManager


def load_wav(path: str, sr: int = 16000):
    try:
        import soundfile as sf
    except Exception:
        raise RuntimeError("soundfile is required to read audio files. Install with `pip install soundfile`")
    data, rate = sf.read(path)
    if rate != sr:
        try:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=rate, target_sr=sr)
        except Exception:
            raise RuntimeError(f"Audio sample rate {rate} != expected {sr}. Install librosa to resample, or provide 16k WAV.")
    # ensure mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", help="Path to WAV file to enroll after recreating DB")
    parser.add_argument("--name", help="Speaker name for enrollment", default="enrolled_speaker")
    parser.add_argument("--no-remove", action="store_true", help="Do not remove persisted DB file (just reset in-memory)")
    args = parser.parse_args()

    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        print("ERROR: PROJECT_ROOT not set. Run: source setup.sh")
        raise SystemExit(1)

    print("Instantiating ModelManager (will load configured models)...")
    MM = ModelManager()

    if not hasattr(MM, 'speaker_embedder'):
        print("No speaker_embedder configured in ModelManager. Check your config and models.")
        raise SystemExit(1)

    embedder = getattr(MM, 'speaker_embedder')
    emb_dim = getattr(embedder, 'embedding_dim', None)
    if emb_dim is None:
        print("speaker_embedder has no embedding_dim attribute; cannot proceed.")
        raise SystemExit(1)

    db_path = settings['db']['audio_recogniser']
    print(f"Using audio DB path: {db_path}")

    arm = AudioRecogniserManager(embedder, embedding_dim=emb_dim, db_path=db_path)

    print("Recreating audio DB (this will clear existing entries)...")
    arm.recreate_database(remove_file=not args.no_remove)
    print("Audio DB recreated.")

    if args.enroll:
        wav_path = args.enroll
        if not os.path.exists(wav_path):
            print(f"Provided file does not exist: {wav_path}")
            raise SystemExit(1)
        print(f"Loading WAV {wav_path}...")
        wav = load_wav(wav_path, sr=getattr(embedder, 'sample_rate', 16000))
        print("Enrolling speaker...")
        sid = arm.enroll(args.name, [wav], speaker_id=None)
        print(f"Enrolled speaker id: {sid}")


if __name__ == '__main__':
    main()

