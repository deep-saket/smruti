# models/qwen25_infer.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
from common import InferenceLLMComponent

class Qwen25Infer(InferenceLLMComponent):
    """
    Inference component for Qwen2.5-3B, supporting both local and API-based modes.
    """
    def __init__(
        self,
        model_name: str = None,
        api_endpoint: str = None,
        api_token: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model_name: Hugging Face model ID for local inference.
            api_endpoint: remote API model endpoint (if using hosted inference).
            api_token: authentication token for the API.
            device: 'cuda' or 'cpu'.
        """
        # Determine device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.api_endpoint = api_endpoint
        self.api_token = api_token
        self.client = None
        self.tokenizer = None
        self.model = None

        # Initialize API client if credentials provided
        if api_endpoint and api_token:
            self.client = InferenceClient(model=api_endpoint, token=api_token)
        # Otherwise load local model
        elif model_name:
            print(f"Loading local model '{model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
            print("Model loaded!")
        else:
            raise ValueError("Either model_name or API credentials must be provided.")

    def infer(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text for the given prompt.

        Args:
            prompt: The input text prompt.
            max_length: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        # API-based inference
        if self.client:
            response = self.client.text_generation(prompt)
            return response if isinstance(response, str) else str(response)

        # Local inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)