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
from typing import Any


class AgentSmruti(BaseComponent):
    """
    High-level agent that ties together audio I/O, models, memory, prompt building,
    MCP processing, and response playback in a modular fashion.

    use_voice: when False the agent runs in headless/text mode. The main
    `run` method unifies both audio and text flows: it reads inputs from the
    audio recorder when `use_voice=True`, otherwise it consumes an iterable of
    text inputs passed via the `text_inputs` argument (or falls back to stdin).
    """

    def __init__(self, use_voice: bool = True):
        super().__init__()
        # NOTE: ensure you've sourced setup.sh before running this script so
        # that PROJECT_ROOT and environment are configured and heavy models can load.
        ModelManager.load_models()

        # core components
        self.mcp_processor = MCPProcessor()
        self.prompt_builder = PromptBuilderMain()
        self.parser = LLMResponseParser()

        # audio components: only create if use_voice is True
        self.use_voice = use_voice
        if self.use_voice:
            self.recorder = AudioRecorder(16000, 1, "int16")
            self.player = AudioPlayer(16000)
        else:
            self.recorder = None
            self.player = None

        self.immediate_memory = ImmediateMemory(capacity=6)

        # initialize the tool decider
        self.tools_decider = ToolsDecider()

        # Optional NER processor (only if configured)
        if 'ner' in ModelManager._registry:
            self.ner_processor = NERProcessor(ModelManager.ner)
        else:
            self.ner_processor = None

        # Memory manager (handles its own missing-model logic)
        self.memory_manager = MemoryManager()
        self.memory_fetcher = MemoryFetcher(self.memory_manager, self.ner_processor)

        self.charactor_details = agent["details"]

        # Optional speaker recogniser (only if embedding model provided)
        if 'speaker_embedder' in ModelManager._registry:
            self.audio_recogniser = AudioRecogniserManager(
                ModelManager.speaker_embedder,
                embedding_dim=ModelManager.speaker_embedder.embedding_dim,
                db_path=settings['db']['audio_recogniser']
            )
        else:
            self.audio_recogniser = None

        self.saket_id = "ef87ca7a1f974e89beeadfcc15c4597b"

    def record_audio(self, seconds: int = 5):
        """Record raw audio from microphone."""
        if not self.use_voice or self.recorder is None:
            raise RuntimeError("Audio recording is disabled in no-voice mode")
        return self.recorder.record_until_speech_end()

    def transcribe(self, audio) -> str:
        """Convert audio to text. If `audio` is already a string (text-mode), return it."""
        if isinstance(audio, str):
            return audio
        stt = getattr(ModelManager, 'stt')
        text = stt.infer(audio).strip()
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
            short_term = getattr(self.memory_manager, 'short_term', None)
            hits = short_term.get(query_text=query, k=5) if short_term is not None else []
            print("📚 You said:")
            for h in hits:
                print("   ", h)
            return True

        return True

    def build_prompt(self, **kwargs) -> Any:
        """Construct the LLM prompt from dialogue history."""
        history = self.immediate_memory.get()
        prompt = self.prompt_builder(context=history, charactor_details=self.charactor_details, **kwargs)
        return prompt

    def generate_response(self, prompt: Any) -> str:
        """Call the LLM model and then MCP processors."""
        llm = getattr(ModelManager, 'llm')
        raw = llm.infer(
                    system_prompt=prompt.get("system"),
                    prompt=prompt.get("user"),
                    assistant_message=prompt.get("assistant")
        )
        self.logger.info(f"LLM raw output: {raw!r}")
        # optional post-processing via MCP chain
        processed = self.prompt_builder.parser(raw)
        return processed

    def think(self, user_text: str) -> list:
        """
        Process a single text input through the agent pipeline and return parsed sentences.
        This bundles: tool selection, MCP calls, prompt building, LLM inference, parsing
        and memory writes.
        """
        # Decide tools and call MCPs
        fetched_memory = self.memory_fetcher(self.immediate_memory.get(), self.saket_id)
        tools = self.tools_decider.decide_tools(user_text)

        # Build prompt and query LLM
        prompt = self.build_prompt(
            new_message=user_text,
            memory_snippets="",
            memory_writes="",
            mcp_results=self.mcp_processor(tools, user_text),
        )
        response = self.generate_response(prompt)
        sentences = self.parser(response).get("sentences", [])

        # Persist to memory
        try:
            self.add_to_memory("Saket", user_text)
            self.add_to_memory("Smruti", " ".join(sentences))
        except Exception:
            self.logger.exception("Failed to write to memory during think()")

        return sentences

    def play_response(self, sentences: list):
        """Play the parsed sentences via TTS (or log when in no-voice mode)."""
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

        tts = getattr(ModelManager, 'tts')
        for sent in sentences:
            wav = tts.infer(sent, speaker_embedding=embed_path)
            if self.player is not None:
                self.player.play(wav)
            else:
                # no-voice mode: log TTS output metadata
                try:
                    import numpy as _np
                    if isinstance(wav, _np.ndarray):
                        self.logger.info(f"[no-voice] Generated wav for sentence (shape={wav.shape}, dtype={wav.dtype})")
                    else:
                        self.logger.info(f"[no-voice] Generated TTS output: {type(wav)}")
                except Exception:
                    self.logger.info("[no-voice] Generated TTS output (non-array)")

    def play(self, sentences):
        """Play the response; accepts either a list of sentences or a single string."""
        # normalize to list
        if isinstance(sentences, str):
            sentences = [sentences]
        self.play_response(sentences)

    def is_english(self, text: str) -> bool:
        """Return True if text is made of basic ASCII chars only."""
        return all(ord(ch) < 128 for ch in text)

    def _input_iterator(self, text_inputs: list[str] | None = None, text_only: bool = False):
        """
        Prepare and return a tuple (source_iter, using_text_iter) for the main loop.

        - text_inputs: optional iterable of strings to consume (used when forcing text mode or when use_voice=False)
        - text_only: if True, force text mode regardless of `self.use_voice`.
        """
        # force text-only terminal mode
        if text_only:
            using_text_iter = True
            if text_inputs is None:
                def stdin_gen():
                    while True:
                        try:
                            val = input('> ')
                        except EOFError:
                            break
                        yield val
                return stdin_gen(), True
            return iter(text_inputs), True

        # not forced text-only
        if self.use_voice:
            return self.recorder.listen(), False
        else:
            using_text_iter = True
            if text_inputs is None:
                def stdin_gen():
                    while True:
                        try:
                            val = input('> ')
                        except EOFError:
                            break
                        yield val
                return stdin_gen(), True
            return iter(text_inputs), True

    def hello_world(self, text_inputs: list[str] | None = None, text_only: bool = False):
        """
        Main interaction loop. Unified for both voice and text modes.

        Args:
            text_inputs: Optional iterable of user text strings used when `use_voice=False`.
                         If None and `use_voice=False`, the function will read from stdin.
            text_only: If True, force text input mode and read from terminal (ignore audio)
        """
        print("AgentSmruti ready (say 'exit' to quit)")

        # Prepare an iterator for inputs (delegated to helper)
        source_iter, using_text_iter = self._input_iterator(text_inputs=text_inputs, text_only=text_only)

        for turn, item in enumerate(source_iter):
            # obtain user_text depending on mode
            try:
                if using_text_iter:
                    user_text = self.transcribe(item)
                else:
                    # audio frame produced by recorder
                    user_text = self.transcribe(item)
            except Exception as e:
                self.logger.exception("Failed to obtain user text: %s", e)
                continue

            print(f"You said: {user_text}")

            # Speaker verification on first turn (only when audio and not text-only forced)
            if turn == 0 and not using_text_iter and self.audio_recogniser is not None:
                try:
                    sid, name, score = self.audio_recogniser.verify(
                        item.astype(np.float32) / np.iinfo(np.int16).max
                    )
                    if sid == self.saket_id:
                        self.play(["Hello Mr. Saket!! Welcome back."])
                except Exception:
                    self.logger.exception("Speaker verification failed")

            # Handle special commands
            if not self.check_special_commands(user_text):
                print("Goodbye!")
                break

            # Process the input through think()
            try:
                sentences = self.think(user_text)
            except Exception:
                self.logger.exception("think() failed for input")
                continue

            # In text_only mode we print the response; otherwise use TTS playback
            if text_only or (self.player is None):
                # Print the response sentences to terminal
                print("Response:", " ".join(sentences))
            else:
                # Use TTS playback
                self.play_response(sentences)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-voice", action="store_true", help="Run agent in no-voice/text mode")
    parser.add_argument("--text", action="append", help="Provide a test input (can be repeated)")
    args = parser.parse_args()

    # Ensure environment is prepared (setup.sh must be sourced to set PROJECT_ROOT)
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        print("ERROR: PROJECT_ROOT is not set. Please run: \n    source setup.sh\nthen retry.")
        raise SystemExit(1)

    ags = AgentSmruti(use_voice=not args.no_voice)
    if args.no_voice:
        # if --text supplied, run through those once; else interactive stdin
        ags.hello_world(text_inputs=args.text if args.text else None, text_only=True)
    else:
        ags.hello_world()
