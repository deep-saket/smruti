"""
local_test.py - safe local runner for AgentSmruti

This script creates a lightweight, safe environment to instantiate and smoke-test
`AgentSmruti` without loading the full, heavy `ModelManager` and large models.

Usage:
    python local_test.py            # dry-run (safe stubbed models)
    python local_test.py --live     # attempt full live run (may load large models)
    python local_test.py --smoke    # dry-run + run a small smoke interaction

Notes:
- The dry-run mode injects a small stub implementation of `models.ModelManager`
  into `sys.modules` so importing `src.AgentSmruti` doesn't trigger heavy model
  initialisation (useful for development, tests, and CI).
- If you want the real agent with real models, run with --live and make sure to
  run `setup.sh` and have dependencies installed; this may take time/memory.

"""

import sys
import types
import argparse
import numpy as np

# Helper: create a small stubbed ModelManager module to avoid heavy imports
def install_stub_modelmanager():
    mod_name = "models.ModelManager"
    if mod_name in sys.modules:
        return False
    mod = types.ModuleType(mod_name)

    class DummySTT:
        def infer(self, audio):
            return "[dry-run stt] this is a dummy transcription"

    class DummyTTS:
        def infer(self, text, speaker_embedding=None):
            # return 1s of silence at 16kHz mono int16
            return np.zeros(16000, dtype=np.int16)

    class DummyLLM:
        def infer(self, system_prompt=None, prompt=None, assistant_message=None):
            return "[dry-run llm] Hello from dummy LLM."

    class DummyEmbedder:
        embedding_dim = 256
        def embed(self, wav: np.ndarray):
            return np.random.rand(self.embedding_dim).astype(np.float32)

    class ModelManager:
        # class-level registry checks in AgentSmruti look for _registry
        _registry = {}
        stt = DummySTT()
        tts = DummyTTS()
        llm = DummyLLM()
        ner = None
        speaker_embedder = DummyEmbedder()

    mod.ModelManager = ModelManager
    sys.modules[mod_name] = mod
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run the real agent (loads real models)")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke test after instantiation")
    args = parser.parse_args()

    if args.live:
        print("Running in LIVE mode: attempting to import real models. Make sure you've run setup.sh and have resources available.")
    else:
        installed = install_stub_modelmanager()
        if installed:
            print("Installed stubbed models.ModelManager for a fast, safe dry-run.")

    # Import the Agent class (will use stub if not running --live)
    try:
        from src.AgentSmruti import AgentSmruti
    except Exception as e:
        print("Failed to import AgentSmruti:", e)
        print("If you intended to run the full agent, try: python local_test.py --live")
        raise

    print("Instantiating AgentSmruti...")
    agent = AgentSmruti()
    print("Agent instantiated successfully.")

    if args.smoke:
        print("Running smoke tests:")
        # test transcribe (uses ModelManager.stt)
        try:
            dummy_audio = np.zeros(16000, dtype=np.int16)
            text = agent.transcribe(dummy_audio)
            print("transcribe ->", text)
        except Exception as e:
            print("transcribe failed:", e)

        # build prompt
        try:
            prompt = agent.build_prompt(new_message="Hello", memory_snippets="", memory_writes="", mcp_results=[])
            print("build_prompt -> type:", type(prompt))
        except Exception as e:
            print("build_prompt failed:", e)

        # generate response (uses ModelManager.llm)
        try:
            resp = agent.generate_response({"system": None, "user": "Say hi", "assistant": None})
            print("generate_response ->", resp)
        except Exception as e:
            print("generate_response failed:", e)

        # play (uses ModelManager.tts) - will not actually play sound in dry-run
        try:
            wav = agent.player if False else None
            sentences = ["This is a test."]
            # Use play_response directly so we don't need external audio input
            agent.play_response(sentences)
            print("play_response executed (no audio verification).")
        except Exception as e:
            print("play_response failed:", e)

    else:
        print("Dry-run complete. To run smoke tests add --smoke, or run the real agent with --live")

if __name__ == '__main__':
    main()
