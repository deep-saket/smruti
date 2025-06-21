from typing import List, Optional
from pydantic import BaseModel
from typing_extensions import Literal

class ToolSelection(BaseModel):
    tool: Literal["weather", "vlm", "none"]
    requires_media: Optional[Literal["image", "video"]] = None
    parameters: dict

class ToolsSelection(BaseModel):
    tools: List[ToolSelection]