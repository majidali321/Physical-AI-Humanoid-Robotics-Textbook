from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    selected_text: Optional[str] = None  # For selection-based RAG
    history: List[Message] = []


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []
    timestamp: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ChatSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    selected_text: Optional[str] = None  # For selection-based search


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict = {}