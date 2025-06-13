import re
from common import CallableComponent

class LLMResponseParser(CallableComponent):
    """
    Parser for LLM-generated text, extracting and cleaning various components:
      - Code blocks (```…```)
      - Bullet/list items (-, *, or numbered)
      - Plain sentences for TTS or further processing

    Usage:
        parser = LLMResponseParser()
        result = parser(raw_llm_output)
        # result == {
        #    "code_blocks": [...],
        #    "bullets": [...],
        #    "sentences": [...],
        # }

    Future enhancements:
      - Integrate robust sentence tokenization (e.g., spaCy, NLTK).
      - Handle inline code (`…`) and markdown links/images.
      - Support numbered lists and nested lists.
      - Extract headings and section metadata.
      - Add language detection to adapt splitting rules.
      - Provide streaming/iterator API for very long responses.
    """
    def __call__(self, text: str) -> dict:
        """
        Parse the raw LLM output into structured pieces.

        Steps:
          1) Extract and remove triple-backtick code blocks.
          2) Extract and remove bullet or list items.
          3) Normalize whitespace.
          4) Split remaining text into sentences.

        Args:
            text: Raw LLM output potentially containing code, lists, and markdown.

        Returns:
            dict with keys:
              - "code_blocks": List[str] of code block contents.
              - "bullets": List[str] of list item lines.
              - "sentences": List[str] of cleaned sentences.
        """
        result = {
            "code_blocks": [],
            "bullets": [],
            "sentences": []
        }

        # 1) Extract code blocks
        code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
        result["code_blocks"] = code_pattern.findall(text)
        no_code = code_pattern.sub('', text)

        # 2) Extract bullet items (−, *, or numbered)
        bullet_pattern = re.compile(r'(?m)^\s*(?:-|\*|\d+\.)\s+(.*)$')
        result["bullets"] = bullet_pattern.findall(no_code)
        no_bullets = bullet_pattern.sub('', no_code)

        # 3) Normalize whitespace
        clean = re.sub(r'\s+', ' ', no_bullets).strip()

        # 4) Split into sentences on .!? boundaries
        parts = re.split(r'(?<=[\.!?])\s+', clean)
        result["sentences"] = [p for p in parts if p]

        return result