# modules/processing/LLMResponseParser.py

import json
from pydantic import ValidationError
from io.lm.ChatResponse import ChatResponse
from common import CallableComponent

class ChatesponseParser(CallableComponent):
    """
    Parses a raw JSON string into a ChatResponse Pydantic model.
    Raises a ValidationError if the JSON is missing required fields
    or has the wrong structure.
    """
    def __call__(self, raw_json: str) -> ChatResponse:
        try:
            # Ensure it’s valid JSON first
            obj = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Now validate against the ChatResponse schema
        return ChatResponse.parse_obj(obj)