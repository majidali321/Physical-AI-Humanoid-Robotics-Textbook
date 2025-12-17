from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

# Simplified models
class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    selected_text: Optional[str] = None
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application starting up...")
    yield
    # Shutdown
    print("Application shutting down...")

app = FastAPI(
    title="RAG Chatbot API - Simplified",
    description="Simplified version for testing",
    version="1.0.0",
    lifespan=lifespan
)

# In-memory storage for testing
chat_sessions = {}

@app.get("/")
def read_root():
    return {"message": "Simplified RAG Chatbot API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Simplified chat endpoint for testing
    """
    try:
        # Generate a session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # For testing purposes, return a simple response
        # In the full implementation, this would use RAG
        if chat_request.selected_text:
            response = f"I received your selected text: '{chat_request.selected_text}'. Based on this, I can answer your question about: '{chat_request.message}'"
        else:
            response = f"You asked: '{chat_request.message}'. This is a simplified response for testing purposes."

        # Store session (simplified)
        chat_sessions[session_id] = chat_request.history + [
            Message(role=MessageRole.USER, content=chat_request.message),
            Message(role=MessageRole.ASSISTANT, content=response)
        ]

        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=["test-source"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/api/v1/search")
async def search_endpoint(chat_request: ChatRequest):
    """
    Simplified search endpoint for testing
    """
    return {
        "results": [
            {"id": "1", "content": "Test search result", "score": 0.9}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)