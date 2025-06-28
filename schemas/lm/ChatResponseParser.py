import json
from pydantic import ValidationError
from common.DirtyJsonParser import DirtyJsonParser
from schemas.lm.ChatResponse import ChatResponse

class ChatResponseParser:
    @staticmethod
    def parse(raw: str) -> ChatResponse:
        """
        Parses raw LLM output into a ChatResponse:
        1) Extracts the first JSON-like block via DirtyJsonParser.
        2) Validates it directly against the ChatResponse schema.
        """
        # 1) Extract the JSON-tolerant object
        try:
            data = DirtyJsonParser.parse(raw)
        except Exception as e:
            raise ValueError(f"ChatResponseParser: failed to extract JSON → {e}")

        # 2) Validate against ChatResponse
        try:
            return ChatResponse.model_validate(data)
        except ValidationError as ve:
            raise ValueError(f"ChatResponseParser: invalid ChatResponse schema → {ve}")