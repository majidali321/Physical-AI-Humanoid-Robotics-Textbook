from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import uuid

from src.models.chat import ChatRequest, ChatResponse, SearchRequest
from src.services.chat_service import ChatService
from src.config.settings import settings

chat_router = APIRouter()

# Initialize the chat service
chat_service = ChatService()


@chat_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint that handles user queries and returns AI responses.
    If selected_text is provided, it will be used as primary context for selection-based RAG.
    """
    try:
        # Generate a session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Process the chat request using the chat service
        response = await chat_service.process_chat(
            message=chat_request.message,
            session_id=session_id,
            selected_text=chat_request.selected_text,
            history=chat_request.history
        )

        return ChatResponse(
            response=response.response,
            session_id=session_id,
            sources=response.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@chat_router.post("/search")
async def search_endpoint(search_request: SearchRequest):
    """
    Search endpoint to retrieve relevant documents from the vector database.
    """
    try:
        results = await chat_service.search_documents(
            query=search_request.query,
            limit=search_request.limit,
            selected_text=search_request.selected_text
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@chat_router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve chat history for a specific session.
    """
    try:
        history = await chat_service.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")


@chat_router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session and its history.
    """
    try:
        await chat_service.delete_session(session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")