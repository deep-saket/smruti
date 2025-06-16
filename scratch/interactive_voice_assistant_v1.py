from modules.audio import AudioRecorder, AudioPlayer
from models import OpenAIWhisperTinyInfer
from models import Qwen25Infer
from models import VITSTTSInfer
from modules.processing import LLMResponseParser
from memory import ImmediateMemory, ShortTermMemory

def main():
    # Initialize components
    recorder = AudioRecorder(16000,1,'int16')
    player   = AudioPlayer(16000)
    stt      = OpenAIWhisperTinyInfer()
    llm      = Qwen25Infer(model_name="Qwen/Qwen2.5-3B")
    tts      = VITSTTSInfer(model_name="tts_models/en/ljspeech/vits", device="cpu")
    parser   = LLMResponseParser()

    # Initialize memory
    chat_mem = ImmediateMemory(capacity=6)
    sem_mem  = ShortTermMemory(capacity=100)

    print("🤖 Interactive voice assistant (say 'exit' to quit)")

    while True:
        audio_in  = recorder.record(5)
        user_text = stt.infer(audio_in).strip()
        print(f"\nYou said: {user_text}")

        # add to memories
        chat_mem.add("user", user_text)
        sem_mem.add(user_text)

        if user_text.lower() in ("exit","quit"):
            print("👋 Goodbye!")
            break

        # simple memory-retrieval command:
        if user_text.lower().startswith("what did i say about"):
            query = user_text[len("what did i say about"):].strip()
            hits  = sem_mem.get(query_text=query, k=5)
            print("📚 You said:")
            for h in hits:
                print("   ", h)
            continue

        # build prompt from last N turns
        history = chat_mem.get()
        prompt  = "\n".join(f"{s}: {t}" for s,t in history) + "\nassistant:"

        print("💬 Generating response...")
        raw = llm.infer(prompt, max_length=256)
        print(f"Assistant (raw): {raw}")

        chat_mem.add("assistant", raw)
        sem_mem.add(raw)

        # parse and speak
        parts = parser(raw).get("sentences", [])
        print("🔊 Speaking response...")
        for sent in parts:
            wav = tts.infer(sent)
            player.play(wav)

    print("🛑 Session ended.")

if __name__=="__main__":
    main()