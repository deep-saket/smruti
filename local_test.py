"""
local_test.py - safe local runner for AgentSmruti

This script creates a lightweight, safe environment to instantiate and smoke-test
`AgentSmruti` without loading the full, heavy `ModelManager` and large models.

Now it reads runtime options from `local_test.yml` in the project root instead
of accepting command-line args. Example keys:
  live: false
  smoke: true

"""

import sys
import types
import yaml
import os
import numpy as np


class SampleTest:
    """Sample test environment with stubbed models for dry-run testing."""

    def __init__(self):
        self.installed_stub = False

    def install_stub_modelmanager(self):
        """Install a stubbed ModelManager to avoid heavy model loading."""
        mod_name = "models.ModelManager"
        if mod_name in sys.modules:
            return False

        mod = types.ModuleType(mod_name)
        pkg = types.ModuleType('models')

        class DummySTT:
            def infer(self, audio):
                return "[dry-run stt] this is a dummy transcription"

        class DummyTTS:
            def infer(self, text, speaker_embedding=None):
                return np.zeros(16000, dtype=np.int16)

        class DummyLLM:
            def infer(self, system_prompt=None, prompt=None, assistant_message=None):
                return "[dry-run llm] Hello from dummy LLM."

        class DummyEmbedder:
            embedding_dim = 256
            def embed(self, wav: np.ndarray):
                return np.random.rand(self.embedding_dim).astype(np.float32)

        class ModelManager:
            _registry = {}
            stt = DummySTT()
            tts = DummyTTS()
            llm = DummyLLM()
            ner = None
            speaker_embedder = DummyEmbedder()
            embedder = DummyEmbedder()  # Add missing embedder attribute

            @classmethod
            def load_models(cls):
                cls._registry = {
                    'stt': cls.stt,
                    'tts': cls.tts,
                    'llm': cls.llm,
                    'speaker_embedder': cls.speaker_embedder,
                    'embedder': cls.embedder
                }
                return cls

        mod.ModelManager = ModelManager
        pkg.ModelManager = ModelManager
        sys.modules['models'] = pkg
        sys.modules[mod_name] = mod
        self.installed_stub = True
        return True

    def run_agent_test(self, use_voice=True, text_only=False, text_inputs=None):
        """Run the agent with sample configuration."""
        try:
            from src.AgentSmruti import AgentSmruti
            print(f"Instantiating AgentSmruti (use_voice={use_voice})...")
            agent = AgentSmruti(use_voice=use_voice)
            print("Agent instantiated successfully.")

            print(f"Running Agent.hello_world() (text_only={text_only})...")
            agent.hello_world(text_inputs=text_inputs, text_only=text_only)
            return agent

        except Exception as e:
            print(f"Agent test failed: {e}")
            raise


class SmokeTest:
    """Smoke test environment for testing individual agent components."""

    def __init__(self, agent):
        self.agent = agent

    def test_transcribe(self):
        """Test audio transcription."""
        try:
            dummy_audio = np.zeros(16000, dtype=np.int16)
            text = self.agent.transcribe(dummy_audio)
            print("transcribe ->", text)
            return True
        except Exception as e:
            print("transcribe failed:", e)
            return False

    def test_build_prompt(self):
        """Test prompt building."""
        try:
            prompt = self.agent.build_prompt(
                new_message="Hello",
                memory_snippets="",
                memory_writes="",
                mcp_results=[]
            )
            print("build_prompt -> type:", type(prompt))
            return True
        except Exception as e:
            print("build_prompt failed:", e)
            return False

    def test_generate_response(self):
        """Test LLM response generation."""
        try:
            resp = self.agent.generate_response({
                "system": None,
                "user": "Say hi",
                "assistant": None
            })
            print("generate_response ->", resp)
            return True
        except Exception as e:
            print("generate_response failed:", e)
            return False

    def test_play_response(self):
        """Test TTS playback."""
        try:
            sentences = ["This is a test."]
            self.agent.play_response(sentences)
            print("play_response executed (no audio verification).")
            return True
        except Exception as e:
            print("play_response failed:", e)
            return False

    def run_all_tests(self):
        """Run all smoke tests."""
        print("Running smoke tests:")
        tests = [
            self.test_transcribe,
            self.test_build_prompt,
            self.test_generate_response,
            self.test_play_response
        ]

        results = {}
        for test in tests:
            test_name = test.__name__
            results[test_name] = test()

        passed = sum(results.values())
        total = len(results)
        print(f"\nSmoke test results: {passed}/{total} tests passed")
        return results


def load_local_test_config():
    """Load configuration from local_test.yml."""
    project_root = os.environ.get('PROJECT_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(project_root, 'local_test.yml')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), 'local_test.yml')
        if not os.path.exists(cfg_path):
            return {"live": False, "smoke": False}

    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return {
                "live": bool(data.get('live', False)),
                "smoke": bool(data.get('smoke', False)),
                "use_voice": bool(data.get('use_voice', True)),
                "hello_world": bool(data.get('hello_world', True)),
                "text_only": bool(data.get('text_only', False)),
                "text_inputs": data.get('text_inputs')
            }
    except Exception as e:
        print(f"Failed to read local_test.yml ({cfg_path}): {e}")
        return {"live": False, "smoke": False}


def main():
    """Main test runner."""
    cfg = load_local_test_config()

    if cfg.get('live', False):
        print("Running in LIVE mode: attempting to import real models.")
        print("Make sure you've run setup.sh and have resources available.")
    else:
        print("Running in SAMPLE mode with stubbed models for dry-run testing.")
        sample_test = SampleTest()
        sample_test.install_stub_modelmanager()
        print("Installed stubbed models.ModelManager for a fast, safe dry-run.")

    # Run agent test
    agent = None
    if cfg.get('hello_world', True):
        try:
            if cfg.get('live', False):
                # Live mode - import directly
                from src.AgentSmruti import AgentSmruti
                print(f"Instantiating AgentSmruti (use_voice={cfg.get('use_voice', True)})...")
                agent = AgentSmruti(use_voice=cfg.get('use_voice', True))
                print("Agent instantiated successfully.")

                print(f"Running Agent.hello_world() (text_only={cfg.get('text_only', False)})...")
                agent.hello_world(
                    text_inputs=cfg.get('text_inputs'),
                    text_only=cfg.get('text_only', False)
                )
            else:
                # Sample mode - use SampleTest
                sample_test = SampleTest()
                sample_test.install_stub_modelmanager()
                agent = sample_test.run_agent_test(
                    use_voice=cfg.get('use_voice', True),
                    text_only=cfg.get('text_only', False),
                    text_inputs=cfg.get('text_inputs')
                )

        except Exception as e:
            print(f"Agent.hello_world() raised an exception: {e}")

    # Run smoke tests if enabled
    if cfg.get('smoke', False) and agent is not None:
        smoke_test = SmokeTest(agent)
        smoke_test.run_all_tests()
    elif cfg.get('smoke', False):
        print("Smoke tests requested but agent instantiation failed.")
    else:
        print("Dry-run complete. To run smoke tests set 'smoke: true' in local_test.yml, or enable live mode with 'live: true'")


if __name__ == '__main__':
    main()
