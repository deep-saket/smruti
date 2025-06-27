from common.DirtyJsonParser import DirtyJsonParser
from schemas.lm.ToolsSelection import ToolsSelection

class ToolsSelectionParser:
    @staticmethod
    def parse(raw: str) -> ToolsSelection:
        """
        Cleans up the raw LLM output and parses it into a ToolsSelection model.
        """
        cleaned = DirtyJsonParser.parse(raw)
        return ToolsSelection.model_validate(cleaned)
