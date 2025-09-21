import os
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from common.InferenceTTSComponent import InferenceTTSComponent

class SpeechT5TTSInfer(InferenceTTSComponent):
    """
    TTS component using Microsoft SpeechT5 + HiFi-GAN vocoder.

    Settings accepted via config (settings['models']["SpeechT5TTSInfer"]):
      - model_name: HF id for SpeechT5 model (default: "microsoft/speecht5_tts")
      - vocoder_name: HF id for HiFi-GAN vocoder (default: "microsoft/speecht5_hifigan")
      - samplerate: output rate (default: 16000)
      - device: 'cpu' or 'cuda'

    infer(text: str, speaker_embedding: np.ndarray|torch.Tensor|str=None, **kwargs)
      - speaker_embedding: either a numpy array / torch tensor of shape (1,512)
        or a path to a .npy file containing that embedding. If None, generation
        will attempt to run without a speaker embedding (may be neutral voice).

    Returns: 1D np.float32 waveform at samplerate.
    """
    def __init__(self, model_name: str = "microsoft/speecht5_tts", vocoder_name: str = "microsoft/speecht5_hifigan", samplerate: int = 16000, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.samplerate = samplerate
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Load processor and models
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name).to(self.device)

    def _load_speaker_embedding(self, speaker_embedding):
        # Accept path or numpy array or torch tensor
        if speaker_embedding is None:
            return None
        if isinstance(speaker_embedding, str):
            if os.path.exists(speaker_embedding):
                emb = np.load(speaker_embedding)
            else:
                raise FileNotFoundError(f"Speaker embedding file not found: {speaker_embedding}")
            speaker_embedding = emb
        if isinstance(speaker_embedding, np.ndarray):
            tensor = torch.from_numpy(speaker_embedding).to(self.device).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return tensor
        if isinstance(speaker_embedding, torch.Tensor):
            return speaker_embedding.to(self.device).float()
        raise TypeError("speaker_embedding must be path, numpy array, torch tensor, or None")

    def infer(self, text: str, speaker_embedding=None, **kwargs) -> np.ndarray:
        # Prepare inputs
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        spk = self._load_speaker_embedding(speaker_embedding)

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=spk,
                vocoder=self.vocoder
            )

        # speech is a torch tensor on device
        wav = speech.cpu().numpy()
        # ensure float32 and 1D
        wav = wav.astype(np.float32).reshape(-1)
        return wav



