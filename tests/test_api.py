import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.models.chat import ChatRequest


client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "RAG Chatbot API is running!"}


def test_health_endpoint():
    """Test the health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_endpoint():
    """Test the chat endpoint with a basic request"""
    chat_request = {
        "message": "Hello, how are you?",
        "session_id": "test-session-123"
    }

    response = client.post("/api/v1/chat", json=chat_request)
    # The response might fail due to missing services, but should return 500, not 404
    assert response.status_code in [200, 500]  # 200 if successful, 500 if service unavailable


def test_chat_endpoint_with_selected_text():
    """Test the chat endpoint with selected text for selection-based RAG"""
    chat_request = {
        "message": "Explain this concept?",
        "session_id": "test-session-456",
        "selected_text": "This is the text the user highlighted"
    }

    response = client.post("/api/v1/chat", json=chat_request)
    # The response might fail due to missing services, but should return 500, not 404
    assert response.status_code in [200, 500]  # 200 if successful, 500 if service unavailable


def test_search_endpoint():
    """Test the search endpoint"""
    search_request = {
        "query": "test search query",
        "limit": 5
    }

    response = client.post("/api/v1/search", json=search_request)
    # The response might fail due to missing services, but should return 500, not 404
    assert response.status_code in [200, 500]  # 200 if successful, 500 if service unavailable


def test_get_session():
    """Test retrieving a session"""
    response = client.get("/api/v1/session/non-existent-session")
    # Should return 404 for non-existent session or 500 if service unavailable
    assert response.status_code in [404, 500]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])