import sys
from common.BaseComponent import BaseComponent
from .ModelManager import ModelManager
from .MCPProcessor import MCPProcessor
from prompts import PromptBuilderMain
from modules.audio import AudioRecorder
from modules.audio import AudioPlayer
from memory.ImmediateMemory import ImmediateMemory
from memory.ShortTermMemory import ShortTermMemory
from modules.processing.LLMResponseParser import LLMResponseParser
from config.loader import settings


class AgentSmruti(BaseComponent):
    """
    High-level agent that ties together audio I/O, models, memory, prompt building,
    MCP processing, and response playback in a modular fashion.
    """

    def __init__(self):
        super().__init__()
        # core components
        self.model_manager = ModelManager()
        # alias common models
        self.stt = self.model_manager.OpenAIWhisperTinyInfer
        self.llm = self.model_manager.QwenV25Infer
        self.tts = self.model_manager.VITSTTSInfer
        self.embedder = self.model_manager.SentenceEmbedderInfer

        self.mcp = MCPProcessor()
        self.prompt_builder = PromptBuilderMain(settings["prompt_builder"])
        self.recorder = AudioRecorder(16000, 1, "int16")
        self.player = AudioPlayer(16000)
        self.immediate_memory = ImmediateMemory(capacity=6)
        self.short_term_memory = ShortTermMemory(self.embedder, capacity=100)
        self.parser = LLMResponseParser()

    def record_audio(self, seconds: int = 5):
        """Record raw audio from microphone."""
        return self.recorder.record(seconds)

    def transcribe(self, audio) -> str:
        """Convert audio to text."""
        text = self.stt.infer(audio).strip()
        self.logger.info(f"STT result: {text!r}")
        return text

    def add_to_memory(self, role: str, text: str):
        """Store message in both immediate (dialogue) and semantic memory."""
        self.immediate_memory.add(role, text)
        self.short_term_memory.add(text)

    def check_special_commands(self, user_text: str) -> bool:
        """
        Handle memory recall or exit commands.
        Returns True if the main loop should continue.
        """
        text = user_text.lower()
        if text in ("exit", "quit"):
            self.logger.info("Exit command received.")
            return False

        if text.startswith("what did i say about"):
            query = user_text[len("what did i say about"):].strip()
            hits = self.short_term_memory.get(query_text=query, k=5)
            print("📚 You said:")
            for h in hits:
                print("   ", h)
            return True

        return True

    def build_prompt(self) -> str:
        """Construct the LLM prompt from dialogue history."""
        history = self.immediate_memory.get()
        prompt = self.prompt_builder.build(history)
        return prompt

    def generate_response(self, prompt: str) -> str:
        """Call the LLM model and then MCP processors."""
        raw = self.llm.infer(prompt, max_length=256)
        self.logger.info(f"LLM raw output: {raw!r}")
        # optional post-processing via MCP chain
        processed = self.mcp(raw)
        return processed

    def parse_and_play(self, raw_response: str):
        """Parse the raw LLM text into sentences and play via TTS."""
        parts = self.parser(raw_response).get("sentences", [])
        for sent in parts:
            wav = self.tts.infer(sent)
            self.player.play(wav)

    def run(self):
        """Main interaction loop."""
        print("🤖 AgentSmruti ready (say 'exit' to quit)")
        while True:
            audio = self.record_audio()
            user_text = self.transcribe(audio)
            print(f"You said: {user_text}")

            # store user turn
            self.add_to_memory("user", user_text)
            if not self.check_special_commands(user_text):
                print("👋 Goodbye!")
                break

            # build prompt and get assistant reply
            prompt = self.build_prompt()
            print("💬 Generating response...")
            response = self.generate_response(prompt)

            # store assistant turn
            self.add_to_memory("assistant", response)
            print(f"Assistant: {response}")

            # speak out
            print("🔊 Speaking response...")
            self.parse_and_play(response)