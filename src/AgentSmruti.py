from common.BaseComponent import BaseComponent
from config.loader import agent, settings
from models.ModelManager import ModelManager
from src.MCPProcessor import MCPProcessor
from prompts import PromptBuilderMain
from modules.audio import AudioRecorder, AudioPlayer, AudioRecogniserManager
from memory.ImmediateMemory import ImmediateMemory
from modules.processing import LLMResponseParser, MemoryFetcher
from src.ToolsDecider import ToolsDecider
from modules.processing import NERProcessor
from memory import MemoryManager
import numpy as np
import os


class AgentSmruti(BaseComponent):
    """
    High-level agent that ties together audio I/O, models, memory, prompt building,
    MCP processing, and response playback in a modular fashion.
    """

    def __init__(self):
        super().__init__()
        # Initialize ModelManager here so models are loaded when the agent is created
        self.model_manager = ModelManager()

        # core components
        self.mcp_processor = MCPProcessor()

        self.prompt_builder = PromptBuilderMain()
        self.parser = LLMResponseParser()

        self.recorder = AudioRecorder(16000, 1, "int16")
        self.player = AudioPlayer(16000)

        self.immediate_memory = ImmediateMemory(capacity=6)

        # initialize the tool decider
        self.tools_decider = ToolsDecider()

        # Optional NER processor (only if configured)
        if 'ner' in getattr(ModelManager, '_registry', {}):
            self.ner_processor = NERProcessor(getattr(ModelManager, 'ner'))
        else:
            self.ner_processor = None

        # Memory manager (handles its own missing-model logic)
        self.memory_manager = MemoryManager()
        self.memory_fetcher = MemoryFetcher(self.memory_manager, self.ner_processor)

        self.charactor_details = agent["details"]

        # Optional speaker recogniser (only if embedding model provided)
        if 'speaker_embedder' in getattr(ModelManager, '_registry', {}):
            self.audio_recogniser = AudioRecogniserManager(
                getattr(ModelManager, 'speaker_embedder'),
                embedding_dim=getattr(ModelManager, 'speaker_embedder').embedding_dim,
                db_path=settings['db']['audio_recogniser']
            )
        else:
            self.audio_recogniser = None

        self.saket_id = "ef87ca7a1f974e89beeadfcc15c4597b"

    def record_audio(self, seconds: int = 5):
        """Record raw audio from microphone."""
        return self.recorder.record_until_speech_end()

    def transcribe(self, audio) -> str:
        """Convert audio to text."""
        text = ModelManager.stt.infer(audio).strip()
        self.logger.info(f"STT result: {text!r}")
        return text

    def add_to_memory(self, role: str, text: str):
        """Store message in both immediate (dialogue) and semantic memory."""
        self.immediate_memory.add(role, text)
        # write to short-term memory if available
        if getattr(self.memory_manager, 'short_term', None) is not None:
            try:
                self.memory_manager.short_term.add(text)
            except Exception:
                # best-effort: do not let memory writes block the agent
                self.logger.exception("Failed to write to short-term memory")

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

    def build_prompt(self, **kwargs) -> str:
        """Construct the LLM prompt from dialogue history."""
        history = self.immediate_memory.get()
        prompt = self.prompt_builder(context=history, charactor_details=self.charactor_details, **kwargs)
        return prompt

    def generate_response(self, prompt: str) -> str:
        """Call the LLM model and then MCP processors."""
        raw = ModelManager.llm.infer(
                    system_prompt=prompt.get("system"),
                    prompt=prompt.get("user"),
                    assistant_message=prompt.get("assistant")
        )
        self.logger.info(f"LLM raw output: {raw!r}")
        # optional post-processing via MCP chain
        processed = self.prompt_builder.parser(raw)
        return processed

    def parse_response(self, raw_response: str) -> list:
        """Parse the raw LLM text into sentences."""
        return self.parser(raw_response).get("sentences", [])

    def play_response(self, sentences: list):
        """Play the parsed sentences via TTS."""
        # Resolve configured speaker embedding path (may be None)
        embed_path = None
        try:
            embed_path = settings.get('tts', {}).get('embedding_path')
            if embed_path:
                embed_path = embed_path.format(project_root=self.project_root)
                if not embed_path or not os.path.exists(embed_path):
                    # if file missing, ignore and log
                    self.logger.info(f"TTS embedding not found at {embed_path}; using neutral voice")
                    embed_path = None
        except Exception:
            embed_path = None

        for sent in sentences:
            # Pass speaker_embedding as path or None; SpeechT5TTSInfer handles path/numpy/tensor
            wav = ModelManager.tts.infer(sent, speaker_embedding=embed_path)
            self.player.play(wav)

    def play(self, sentences: str):
        """Parse the raw LLM text into sentences and play via TTS."""
        self.play_response(sentences)

    def is_english(self, text: str) -> bool:
        """Return True if text is made of basic ASCII chars only."""
        return all(ord(ch) < 128 for ch in text)

    def run(self):
        """Main interaction loop."""
        print("AgentSmruti ready (say 'exit' to quit)")
        for turn, audio in enumerate(self.recorder.listen()):
            # Transcribe and log
            user_text = self.transcribe(audio)
            print(f"You said: {user_text}")
            # Speaker verification on first turn
            if turn == 0 and self.audio_recogniser is not None:
                sid, name, score = self.audio_recogniser.verify(
                    audio.astype(np.float32) / np.iinfo(np.int16).max
                )
                if sid == self.saket_id:
                    self.play(["Hello Mr. Saket!! Welcome back."])
            # Handle special commands
            if not self.check_special_commands(user_text):
                print("Goodbye!")
                break
            # Memory fetch and tool selection
            fetched_memory = self.memory_fetcher(
                self.immediate_memory.get(), self.saket_id
            )
            tools = self.tools_decider.decide_tools(user_text)
            # Build prompt, generate response and speak
            prompt = self.build_prompt(
                new_message=user_text,
                memory_snippets="",
                memory_writes="",
                mcp_results=self.mcp_processor(tools, user_text),
            )
            response = self.generate_response(prompt)
            sentences = self.parse_response(response)
            self.add_to_memory("Saket", user_text)
            self.add_to_memory("Smruti", " ".join(sentences))
            self.play(" ".join(sentences))

if __name__ == "__main__":
    ags = AgentSmruti()
    ags.run()
