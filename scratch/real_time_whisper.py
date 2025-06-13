# real_time_whisper.py

from modules.audio import AudioStreamer, RealTimeTranscriber
from models import OpenAIWhisperTinyInfer

def main():
    # 1) Set up continuous audio streamer (2-second blocks)
    streamer = AudioStreamer(samplerate=16000, channels=1, dtype='int16', block_duration=2.0)
    streamer.start()
    print("🔴 Audio stream started.")

    # 2) Initialize Whisper Tiny inference component
    infer = OpenAIWhisperTinyInfer(model_name="openai/whisper-tiny", samplerate=16000)

    # 3) Tie streamer to real-time transcriber
    transcriber = RealTimeTranscriber(infer)

    # 4) Run real-time transcription until interrupted
    transcriber.transcribe_stream(streamer)

    # 5) Clean up
    streamer.stop()
    print("🟢 Audio stream stopped.")

if __name__ == "__main__":
    main()