import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common.InferenceAudioComponent import InferenceAudioComponent

class WhisperMediumInfer(InferenceAudioComponent):
    """
    STT inference component using the OpenAI Whisper 'medium' model.

    Notes:
    - This class uses Hugging Face transformers' Whisper model classes.
    - The medium model is large and requires significant RAM and a capable CPU/GPU.
    - For edge deployment prefer converting to ONNX or using a quantized whisper.cpp build.

    Args (via settings):
      model_name: HF model id (default: "openai/whisper-medium").
      samplerate: expected audio sampling rate (default: 16000).
      device: 'cpu' or 'cuda' (default: 'cpu').
    """
    def __init__(self, model_name: str = "openai/whisper-medium", samplerate: int = 16000, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.samplerate = samplerate
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def infer(self, audio_data: np.ndarray) -> str:
        """
        Transcribe the given audio array using Whisper medium.

        Args:
            audio_data: 1D numpy array of floats in the range [-1, 1] or int16 array.

        Returns:
            str: Transcribed text.
        """
        # Ensure numpy float array normalized to -1..1
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        else:
            audio = audio_data.astype(np.float32)

        inputs = self.processor(audio, sampling_rate=self.samplerate, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(self.device)

        # Pad input features to the required length of 3000
        expected_seq_length = 3000
        current_seq_length = input_features.shape[-1]
        if current_seq_length < expected_seq_length:
            padding = torch.zeros((input_features.shape[0], input_features.shape[1], expected_seq_length - current_seq_length), device=self.device)
            input_features = torch.cat((input_features, padding), dim=-1)

        generated_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
