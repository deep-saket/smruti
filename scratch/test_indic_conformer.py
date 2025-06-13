# test_indic_conformer.py
#!/usr/bin/env python3
"""
A test harness for ai4bharat/indic-conformer-600m-multilingual to process
either WAV files or microphone input using the AudioRecorder class.
"""
import sys
import argparse
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from modules.audio import AudioRecorder

# Load environment variables from .env
load_dotenv()

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

def load_audio(path, target_sampling_rate=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sampling_rate)
    return waveform.squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Test IndicConformer ASR on WAV or mic.")
    parser.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration (s) for microphone recording")
    parser.add_argument("audio_files", nargs="*", help="Paths to WAV files to transcribe")
    args = parser.parse_args()

    # Load processor and model
    print(f"Loading model and processor '{MODEL_ID}'...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=(5, 5),
        device=0 if torch.cuda.is_available() else -1,
        trust_remote_code=True
    )

    transcripts = []

    if args.mic:
        recorder = AudioRecorder(samplerate=processor.feature_extractor.sampling_rate)
        waveform = recorder.record(args.duration)
        result = asr(waveform, sampling_rate=recorder.samplerate)
        transcripts.append(("mic", result["text"]))
    else:
        if not args.audio_files:
            print("Error: no audio files provided. Use --mic or list WAV files.")
            sys.exit(1)
        for path in args.audio_files:
            print(f"\n📂 Processing '{path}'...")
            waveform = load_audio(path, processor.feature_extractor.sampling_rate)
            result = asr(waveform, sampling_rate=processor.feature_extractor.sampling_rate)
            transcripts.append((path, result["text"]))

    # Print results
    for source, text in transcripts:
        label = "Microphone" if source == "mic" else source
        print(f"\n📝 Transcript ({label}):\n{text}")


if __name__ == "__main__":
    main()
