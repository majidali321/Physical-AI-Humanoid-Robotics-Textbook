import requests
import json

# Test the API endpoints
BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print("Testing the RAG Chatbot API...")

    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error testing root endpoint: {e}")

    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error testing health endpoint: {e}")

    # Test chat endpoint
    try:
        chat_data = {
            "message": "Hello, how are you?",
            "session_id": "test-session-123"
        }
        response = requests.post(f"{BASE_URL}/api/v1/chat", json=chat_data)
        print(f"Chat endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error testing chat endpoint: {e}")

    # Test chat endpoint with selected text (selection-based RAG)
    try:
        chat_data = {
            "message": "Explain this concept?",
            "session_id": "test-session-456",
            "selected_text": "This is the text the user highlighted"
        }
        response = requests.post(f"{BASE_URL}/api/v1/chat", json=chat_data)
        print(f"Selection-based RAG: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error testing selection-based RAG: {e}")

if __name__ == "__main__":
    test_endpoints()