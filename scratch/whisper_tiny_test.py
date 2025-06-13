# whisper_tiny_test.py

from modules.audio import AudioRecorder, AudioPlayer
from models import OpenAIWhisperTinyInfer

def main():
    # 1) Record a short audio clip
    recorder      = AudioRecorder()
    audio_data    = recorder.record(duration=5)  # record for 5 seconds

    # 2) Play it back for verification
    player        = AudioPlayer()
    player.play(audio_data)

    # 3) Transcribe via the Whisper Tiny inference component
    infer         = OpenAIWhisperTinyInfer()
    transcription = infer.infer(audio_data)
    print("\n📝 Transcription:", transcription)

if __name__ == "__main__":
    main()