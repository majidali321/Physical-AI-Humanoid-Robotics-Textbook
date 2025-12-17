from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime
import google.generativeai as genai
import os

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA4YfeJ0r51eFklgKrXbcVKRPBY1L82hMc")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

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
    description="Simplified version for testing with Gemini API",
    version="1.0.0",
    lifespan=lifespan
)

# In-memory storage for testing
chat_sessions = {}

@app.get("/")
def read_root():
    return {"message": "Simplified RAG Chatbot API with Gemini is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Chat endpoint using Gemini API
    """
    try:
        # Generate a session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Build the prompt based on whether there's selected text
        if chat_request.selected_text:
            # Special prompt for selection-based RAG
            prompt = f"""
            You are a helpful assistant for a digital book. Answer the user's question based STRICTLY on the PRIMARY CONTEXT provided below.
            If the PRIMARY CONTEXT doesn't contain relevant information, say so.
            Do not fabricate information.

            PRIMARY CONTEXT (Selected Text):
            {chat_request.selected_text}

            User's Question: {chat_request.message}

            Please provide a helpful response based strictly on the PRIMARY CONTEXT.
            """
        else:
            # Standard prompt
            prompt = f"""
            You are a helpful assistant. The user asked: {chat_request.message}
            Provide a helpful response.
            """

        # Generate response using Gemini
        try:
            response = await gemini_model.generate_content_async(prompt)
            ai_response = response.text
        except Exception as e:
            ai_response = f"Sorry, I encountered an error: {str(e)}. This is a fallback response."

        # Store session (simplified)
        chat_sessions[session_id] = chat_request.history + [
            Message(role=MessageRole.USER, content=chat_request.message),
            Message(role=MessageRole.ASSISTANT, content=ai_response)
        ]

        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            sources=["gemini-api"]
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