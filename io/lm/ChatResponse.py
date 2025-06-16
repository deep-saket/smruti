from pydantic import BaseModel, Field
from typing import Literal, Dict, List, Optional


class Message(BaseModel):
    """
    Represents a single turn in the conversation.
    """
    turn_id: int
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: float  # Unix epoch seconds


class NextAction(BaseModel):
    """
    Describes an agentic action the assistant should perform next.
    """
    name: str                # e.g., "schedule_reminder"
    args: Dict[str, str] = {}  # parameters for the action


class RetrievalResult(BaseModel):
    """
    A single retrieved snippet with its relevance score.
    """
    text: str
    score: float


class Retrieval(BaseModel):
    """
    Details of any semantic or keyword retrieval performed.
    """
    method: str                     # e.g., "semantic" or "keyword"
    query: str
    results: List[RetrievalResult] = []


class ChatResponse(BaseModel):
    """
    The structured JSON response schema from the LLM.
    """
    schema_version: str = Field("1.0", const=True)
    status: Literal["ok", "error"]
    messages: List[Message]

    intent: Optional[str] = None
    entities: Optional[Dict[str, str]] = None

    next_action: Optional[NextAction] = None

    memory_read: Optional[List[str]] = None
    memory_write: Optional[List[str]] = None

    warnings: Optional[List[str]] = None
    retrieval: Optional[Retrieval] = None