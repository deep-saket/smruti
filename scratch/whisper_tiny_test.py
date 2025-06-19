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


from mcp_util.clients import WeatherClient

async def main2():
    client = WeatherClient()
    city = input("Enter city name: ").strip()
    try:
        # call the client’s .req method to fetch weather
        info = await client.request(city)
        print(f"Weather for {city}:")
        print(f"  City:           {info.city}")
        print(f"  Description:    {info.description}")
        print(f"  Temperature °C: {info.temperature_c}")
        print(f"  Humidity %:     {info.humidity}")
    except Exception as e:
        print(f"Error fetching weather: {e}")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main2())
