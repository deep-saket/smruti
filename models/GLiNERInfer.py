import torch
from gliner import GLiNER
from common.InferenceNERComponent import InferenceNERComponent


class GLiNERInfer(InferenceNERComponent):
    """
    Inference component for GLiNER NER model.
    Supports extracting entities for arbitrary label sets.
    """

    def __init__(self, model_name: str = "urchade/gliner_base", device: str = "cpu"):
        """
        Args:
            model_name: Hugging Face model ID for GLiNER ("urchade/gliner_base" or "urchade/gliner_small").
            device: Device to run the model on ('cpu', 'cuda', or 'mps').
        """
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            dev = device.lower()
            if dev == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif dev == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif dev == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError(f"Unsupported or unavailable device: {device}")
        else:
            raise TypeError("`device` must be a str or torch.device")

        print(f"Loading GLiNER model '{model_name}' on {self.device}...")
        self.model = GLiNER.from_pretrained(model_name).to(self.device)
        print("Model loaded!")

    def infer(self, text: str, labels: list[str]) -> list[dict]:
        """
        Perform entity extraction on the input text using the specified labels.

        Args:
            text: Input string to extract entities from.
            labels: List of entity types to look for.

        Returns:
            A list of dictionaries with 'entity' and 'type' keys.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
            raise ValueError("Labels must be a list of strings.")

        return self.model.predict_entities(text, labels)


if __name__ == "__main__":
    # Quick test
    ner_model = GLiNERInfer(model_name="urchade/gliner_small", device="cpu")
    sample_text = "Barack Obama was born in Hawaii and served as the president of the United States."
    sample_labels = ["Person", "Location", "Organization", "Position"]
    entities = ner_model.infer(sample_text, sample_labels)
    print("Extracted Entities:")
    for ent in entities:
        print(f"  - {ent}")