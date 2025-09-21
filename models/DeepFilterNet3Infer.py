from common import AudioToAudioComponent
import numpy as np
from openvino.runtime import Core
from optimum.intel import OVModelForSpeechEnhancement
import torch


class DeepFilterNet3Infer(AudioToAudioComponent):
    """
    Audio-to-audio component using DeepFilterNet 3 loaded from Hugging Face.
    """

    def __init__(self, model_name="Intel/deepfilternet-openvino"):
        # Load model from HuggingFace Hub using Optimum and OpenVINO
        self.model = OVModelForSpeechEnhancement.from_pretrained(
            model_name, model_id="deepfilternet3"
        )
        self.ov_model = self.model.to_openvino()

        # Compile the model for CPU
        self.core = Core()
        self.compiled_model = self.core.compile_model(self.ov_model.model, "CPU")
        self.input_tensor_name = self.compiled_model.input(0)
        self.output_tensor_name = self.compiled_model.output(0)

    def process(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the input audio array through DeepFilterNet3 and return denoised audio.

        Args:
            audio (np.ndarray): Input audio at 48kHz, mono, float32.

        Returns:
            np.ndarray: Denoised audio at 48kHz.
        """
        # Ensure audio is 2D for OpenVINO (batch, samples)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        # Perform inference
        result = self.compiled_model([audio.astype(np.float32)])
        denoised_audio = result[self.output_tensor_name]

        # Output is (1, samples), convert back to 1D
        return denoised_audio.squeeze(0)