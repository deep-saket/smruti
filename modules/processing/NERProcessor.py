from common import CallableComponent


class NERProcessor(CallableComponent):
    """
    Processor class to handle Named Entity Recognition using GLiNER.

    Usage:
        ner_model = GLiNERInfer(model_name="urchade/gliner_small", device="cpu")
        ner_processor = NERProcessor(ner_model)

        entities = ner_processor("Barack Obama was president.", keys=["Person", "Position"])
        print(entities)
        # Output: {'Person': ['Barack Obama'], 'Position': ['president']}
    """

    def __init__(self, ner_model):
        self.ner_model = ner_model

    def __call__(self, text: str, keys: list[str] = None) -> dict:
        """
        Extract entities from text based on specified keys.

        Args:
            text (str): Text input from which to extract entities.
            keys (list[str], optional): Specific entity types to extract.
                                        Defaults to ["Person", "Location", "Organization", "Position"].

        Returns:
            dict: Dictionary of entities categorized by keys.
        """
        default_labels = ["Person", "Location", "Organization", "Position"]
        labels = keys if keys else default_labels

        entities = self.ner_model.infer(text, labels)

        result = {label: [] for label in labels}
        for entity in entities:
            entity_type = entity.get('type')
            entity_value = entity.get('entity')
            if entity_type in result:
                result[entity_type].append(entity_value)

        # Remove empty lists
        result = {k: v for k, v in result.items() if v}

        return result


# Example usage
if __name__ == "__main__":
    ner_model = GLiNERInfer(model_name="urchade/gliner_small", device="cpu")
    ner_processor = NERProcessor(ner_model)

    sample_text = "Barack Obama was born in Hawaii and served as the president of the United States."

    entities = ner_processor(sample_text, keys=["Person", "Location", "Position"])
    print("Extracted Entities:")
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}:")
        for ent in entity_list:
            print(f"  - {ent}")