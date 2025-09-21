import time
import sphn
import torch
import numpy as np
from common import InferenceAudioComponent

from moshi.models import loaders, MimiModel, LMModel, LMGen

class KyutaiSTTInfer(InferenceAudioComponent):
    """
    Inference component for the Moshi model.
    """
    def __init__(self, model_name: str ="kyutai/stt-1b-en_fr", device: str = "cpu", **kwargs):
        super().__init__(**kwargs) # Call the base class constructor
        self.device = device
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_name)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.samplerate = self.mimi.sample_rate

        stt_config = checkpoint_info.stt_config
        self.pad_left = int(stt_config.get("audio_silence_prefix_seconds", 0.0) * self.samplerate)
        self.pad_right = int((stt_config.get("audio_delay_seconds", 0.0) + 1.0) * self.samplerate)

        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        batch_size = 1 # Assuming batch size of 1 for inference
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)


    def infer(self, audio_data: np.ndarray) -> str:
        """
        Transcribe the given audio array using Moshi.
        """
        in_pcms = torch.from_numpy(audio_data).to(device=self.device)
        in_pcms = torch.nn.functional.pad(in_pcms, (self.pad_left, self.pad_right), mode="constant")
        in_pcms = in_pcms[None, 0:1].expand(1, -1, -1)

        ntokens = 0
        first_frame = True
        chunks = [
            c
            for c in in_pcms.split(self.frame_size, dim=2)
            if c.shape[-1] == self.frame_size
        ]
        start_time = time.time()
        all_text = []
        for chunk in chunks:
            codes = self.mimi.encode(chunk)
            if first_frame:
                tokens = self.lm_gen.step(codes)
                first_frame = False
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == 1
            one_text = tokens[0, 0].cpu()
            if one_text.item() not in [0, 3]:
                text = self.text_tokenizer.id_to_piece(one_text.item())
                text = text.replace(" ", " ")
                all_text.append(text)
            ntokens += 1
        dt = time.time() - start_time
        print(
            f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step"
        )
        return "".join(all_text)