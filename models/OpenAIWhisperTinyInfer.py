import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common.InferenceAudioComponent import InferenceAudioComponent

class OpenAIWhisperTinyInfer(InferenceAudioComponent):
    """
    Inference component for the OpenAI Whisper Tiny model.
    """
    def __init__(self, model_name="openai/whisper-tiny", samplerate=16000):
        self.samplerate = samplerate
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def infer(self, audio_data: np.ndarray) -> str:
        """
        Transcribe the given audio array using Whisper Tiny.
        """
        inputs = self.processor(
            audio_data, sampling_rate=self.samplerate, return_tensors="pt"
        ).input_features
        generated_ids = self.model.generate(inputs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]