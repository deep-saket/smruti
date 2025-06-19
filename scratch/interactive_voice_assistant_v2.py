# interactive_voice_assistant.py

from modules.audio import AudioRecorder, AudioPlayer
from models.OpenAIWhisperTinyInfer import OpenAIWhisperTinyInfer
from models import QwenV25Infer
from models.VITSTTSInfer import VITSTTSInfer
from prompts import PromptBuilderMain
from memory import ImmediateMemory
from memory import ShortTermMemory
from modules.services.weather import WeatherComponent

def main():
    # Initialize I/O & models
    recorder = AudioRecorder(samplerate=16000, channels=1, dtype='int16')
    player   = AudioPlayer(samplerate=16000)
    stt      = OpenAIWhisperTinyInfer()
    llm      = QwenV25Infer(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
    tts      = VITSTTSInfer(model_name="tts_models/en/ljspeech/vits", device="cpu")

    # Prompt builder (which also provides parser)
    builder  = PromptBuilderMain()

    # Memories
    chat_mem = ImmediateMemory(capacity=6)
    sem_mem  = ShortTermMemory(capacity=100)

    print("🤖 Interactive voice assistant (say 'exit' to quit)")

    i = 0
    while True:
        # 1) Capture & transcribe
        audio_in  = recorder.record(duration=5)
        user_text = stt.infer(audio_in).strip()
        print(f"\nYou said: {user_text}")

        if i == 0:
            user_text = "Hi"
        if i == 1:
            user_text = "What is the weather like today?"
        i += 1

        # 2) Update memories
        chat_mem.add("user", user_text)
        sem_mem.add(user_text)

        if user_text.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # 3) Ad-hoc retrieval
        if user_text.lower().startswith("what did i say about"):
            query = user_text[len("what did i say about"):].strip()
            hits  = sem_mem.get(query_text=query, k=5)
            print("📚 You said:")
            for h in hits:
                print("   ", h)
            continue

        if user_text.lower().startswith("weather in ") or user_text.lower().startswith("what's the weather in "):
            city = user_text.split("in", 1)[1].strip()
            weather = WeatherComponent()(city)
            print(f"🌤️  {weather.city}: {weather.description}, {weather.temperature_c}°C, humidity {weather.humidity}%")
            continue

        # 4) Build prompts
        history         = "\n".join(f"{role}: {txt}" for role, txt in chat_mem.get())
        memory_snippets = "\n".join(sem_mem.get()[:5])
        memory_writes   = ""  # or populate from your logic
        context = {
            "new_message":     user_text,
            "context":         history,
            "memory_snippets": memory_snippets,
            "memory_writes":   memory_writes
        }
        prompts = builder(**context)

        # 5) Generate LLM response
        print("💬 Generating response...")
        system_prompt = prompts["system"]
        user_prompt   = prompts["user"]

        raw = llm.infer_lang(
            system_prompt=system_prompt,
            prompt=user_prompt,
            max_new_tokens=8000
        )
        print(f"Assistant (raw): {raw}")

        # 6) Parse via builder’s cached parser
        chat_resp      = builder.parser(raw)
        assistant_text = chat_resp.messages[-1].content

        # 7) Update memories with assistant’s reply
        chat_mem.add("assistant", assistant_text)
        sem_mem.add(assistant_text)

        # 8) Play back
        print("🔊 Speaking response...")
        for line in assistant_text.splitlines():
            if line.strip():
                wav = tts.infer(line)
                player.play(wav)

    print("🛑 Session ended.")

if __name__ == "__main__":
    main()