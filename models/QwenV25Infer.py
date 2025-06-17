import torch
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import InferenceClient
from common.InferenceVLComponent import InferenceVLComponent
from qwen_vl_utils import process_vision_info

class QwenV25Infer(InferenceVLComponent):
    """
    Inference component for the Qwen2.5-VL model, supporting
    local and API modes, with text, image, video and pure-language calls.
    """

    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='mps'):
        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.device = device
        self.client = None
        self.model = None
        self.processor = None

        if self.api_endpoint and self.api_token:
            self.client = InferenceClient(model=api_endpoint, token=api_token)
        elif model_name:
            print(f"Loading {model_name} model...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            print("Model loaded!")
        else:
            raise ValueError("Either `model_name` or API details must be provided.")

    def infer(
        self,
        image_data=None,
        video_data=None,
        prompt: str = None,
        system_prompt: str = None,
        assistant_message: str = None,
        max_new_tokens: int = 512,
        **generate_kwargs
    ) -> str:
        if not any([image_data, video_data, prompt]):
            raise ValueError("Provide at least one of image_data, video_data, or prompt.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content":[{"type":"text","text": system_prompt}]})
        if assistant_message:
            messages.append({"role": "assistant", "content":[{"type":"text","text": assistant_message}]})

        user_content = []
        if image_data is not None:
            if isinstance(image_data, bytes):
                img = Image.open(BytesIO(image_data)).convert("RGB")
            elif isinstance(image_data, Image.Image):
                img = image_data
            elif isinstance(image_data, str):
                img = Image.open(image_data).convert("RGB")
            else:
                raise ValueError("image_data must be bytes, PIL.Image, or file path")
            user_content.append({"type":"image","image": img})

        if video_data is not None:
            user_content.append({"type":"video","video": video_data})

        if prompt:
            user_content.append({"type":"text","text": prompt})

        messages.append({"role":"user","content": user_content})

        if self.client:
            resp = self.client.text_generation({"inputs": messages})
            return resp if isinstance(resp, str) else resp.get("generated_text", str(resp))

        chat_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
        out_ids = gen_ids[:, prompt_len:]
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    def infer_lang(
        self,
        prompt: str = None,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        **generate_kwargs
    ) -> str:
        """
        Run pure-language inference, without images or video.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("`prompt` must be a non-empty string.")

        if self.client:
            # API call
            payload = {"inputs": [{"role":"user","content":[{"type":"text","text": prompt}]}]}
            if system_prompt:
                payload["inputs"].insert(0, {"role":"system","content":[{"type":"text","text": system_prompt}]})
            resp = self.client.text_generation(payload)
            return resp if isinstance(resp, str) else resp.get("generated_text", str(resp))

        # Local chat formatting
        messages = []
        if system_prompt:
            messages.append({"role":"system","content":[{"type":"text","text": system_prompt}]})
        messages.append({"role":"user","content":[{"type":"text","text": prompt}]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
        out_ids = gen_ids[:, prompt_len:]
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]