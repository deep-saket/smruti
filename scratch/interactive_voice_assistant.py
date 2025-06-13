# interactive_voice_assistant.py

from modules.audio import AudioRecorder, AudioPlayer
from models import OpenAIWhisperTinyInfer
from models import Qwen25Infer
from models import VITSTTSInfer
from modules.processing import LLMResponseParser

from dotenv import load_dotenv
load_dotenv()


def main():
    # Initialize components
    recorder = AudioRecorder(samplerate=16000, channels=1, dtype='int16')
    player   = AudioPlayer(samplerate=16000)
    stt      = OpenAIWhisperTinyInfer()  # speech-to-text
    llm      = Qwen25Infer(model_name="Qwen/Qwen2.5-3B")  # LLM
    tts      = VITSTTSInfer(model_name="tts_models/en/ljspeech/vits", device="cpu")  # text-to-speech
    llm_response_parser   = LLMResponseParser()  # LLMResponseParser instance

    print("🤖 Interactive voice assistant (say 'exit' to quit)")

    while True:
        # 1) Capture user speech
        audio_in = recorder.record(duration=5)

        # 2) Transcribe to text
        user_text = stt.infer(audio_in).strip()
        print(f"\nYou said: {user_text}")

        # Exit condition
        if user_text.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # 3) Generate assistant response
        print("💬 Generating response...")
        raw_response = llm.infer(user_text, max_length=256, temperature=0.7)
        print(f"Assistant (raw): {raw_response}")

        # 4) Parse into sentences
        parsed = llm_response_parser(raw_response)
        sentences = parsed.get("sentences", [])

        # 5) Synthesize and play each sentence
        print("🔊 Speaking response...")
        for sent in sentences:
            wav = tts.infer(sent)
            player.play(wav)

    print("🛑 Session ended.")


if __name__ == "__main__":
    main()