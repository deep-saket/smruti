# test_tts_playback.py

import numpy as np
import sounddevice as sd
from models import VITSTTSInfer

def main():
    # Initialize TTS on CPU for testing
    tts = VITSTTSInfer(model_name="tts_models/en/ljspeech/vits", speaker=None, device='cpu')
    test_text = "This is a volume test for the text-to-speech system. One two three."

    # Generate waveform
    wav = tts.infer(test_text)

    # Normalize if peak > 1.0
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / peak
    wav = wav.astype('float32')

    # Determine sample rate (fallback to 22050 Hz)
    try:
        sr = tts.tts.synthesizer.output_sample_rate
    except Exception:
        sr = 22050

    sd.default.samplerate = sr
    sd.default.channels = wav.ndim if wav.ndim > 1 else 1

    sd.default.device = (2, None)

    # Play back the waveform directly
    sd.play(wav, sr)
    sd.wait()

if __name__ == "__main__":
    main()