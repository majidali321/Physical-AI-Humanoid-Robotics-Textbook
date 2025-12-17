import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.chat_service import ChatService
from src.models.chat import Message, MessageRole


@pytest.fixture
def mock_chat_service():
    """Create a mock chat service for testing"""
    service = ChatService()
    service.db_service = AsyncMock()
    service.vector_store = MagicMock()
    service.embedding_service = MagicMock()
    service.openai_client = None  # Skip OpenAI for tests

    return service


@pytest.mark.asyncio
async def test_process_chat_without_selected_text(mock_chat_service):
    """Test chat processing without selected text (regular RAG)"""
    # Mock the necessary methods
    mock_chat_service.db_service.add_message = AsyncMock()
    mock_chat_service.embedding_service.embed_query = MagicMock(return_value=[0.1] * 1024)
    mock_chat_service.vector_store.search = MagicMock(return_value=[
        {
            "id": "1",
            "content": "This is relevant content",
            "score": 0.9,
            "metadata": {"source": "chapter1"}
        }
    ])

    # Call the method
    response = await mock_chat_service.process_chat(
        message="What is this about?",
        session_id="test-session"
    )

    # Assertions
    assert response.session_id == "test-session"
    assert len(response.sources) == 1
    assert response.sources[0] == "chapter1"


@pytest.mark.asyncio
async def test_process_chat_with_selected_text(mock_chat_service):
    """Test chat processing with selected text (selection-based RAG)"""
    # Mock the necessary methods
    mock_chat_service.db_service.add_message = AsyncMock()
    mock_chat_service.embedding_service.embed_query = MagicMock(return_value=[0.1] * 1024)
    mock_chat_service.vector_store.search = MagicMock(return_value=[
        {
            "id": "1",
            "content": "This is related content",
            "score": 0.8,
            "metadata": {"source": "chapter2"}
        }
    ])

    # Call the method with selected text
    response = await mock_chat_service.process_chat(
        message="Explain this?",
        session_id="test-session-2",
        selected_text="This is the selected text that should be prioritized"
    )

    # Assertions
    assert response.session_id == "test-session-2"
    assert len(response.sources) == 1
    assert response.sources[0] == "chapter2"


@pytest.mark.asyncio
async def test_search_documents(mock_chat_service):
    """Test document search functionality"""
    # Mock the embedding and search methods
    mock_chat_service.embedding_service.embed_query = MagicMock(return_value=[0.2] * 1024)
    mock_chat_service.vector_store.search = MagicMock(return_value=[
        {
            "id": "1",
            "content": "Test document content",
            "score": 0.95,
            "metadata": {"source": "test_source"}
        }
    ])

    # Call the search method
    results = await mock_chat_service.search_documents(
        query="test query",
        limit=5
    )

    # Assertions
    assert len(results) == 1
    assert results[0].content == "Test document content"
    assert results[0].score == 0.95


@pytest.mark.asyncio
async def test_get_session_history(mock_chat_service):
    """Test retrieving session history"""
    # Mock the database response
    mock_msg = MagicMock()
    mock_msg.role = "user"
    mock_msg.content = "Hello"
    mock_msg.timestamp = "2023-01-01T00:00:00"

    mock_chat_service.db_service.get_messages = AsyncMock(return_value=[mock_msg])

    # Call the method
    history = await mock_chat_service.get_session_history("test-session")

    # Assertions
    assert len(history) == 1
    assert history[0].role == MessageRole.USER
    assert history[0].content == "Hello"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])