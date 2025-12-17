# RAG Chatbot for Digital Book

This project implements an integrated RAG (Retrieval-Augmented Generation) Chatbot for a digital book using FastAPI, Qdrant Cloud, Cohere, Neon Postgres, and OpenAI.

## Tech Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant Cloud
- **Embeddings Provider**: Cohere
- **Database**: Neon Postgres
- **AI Models**: OpenAI

## Features

1. **Retrieval from Qdrant using Cohere Embeddings**: Uses Cohere's 'embed-english-v3.0' model for generating embeddings and Qdrant Cloud for vector storage and retrieval.

2. **Selection-Based RAG**: Allows users to highlight specific text passages and answers questions based strictly on the highlighted text.

3. **Chat History Storage**: Stores conversation history in Neon Postgres database for session persistence.

## Configuration Details

- **Qdrant Endpoint**: https://f7896955-bf33-46c5-b4e8-454d2cb95879.europe-west3-0.gcp.cloud.qdrant.io
- **Qdrant API Key**: (Stored in environment variables)
- **Cohere API Key**: (Stored in environment variables)
- **Neon Postgres Connection**: (Stored in environment variables)

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Copy the `.env.example` file to `.env` and update the values:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

5. **Run the application**:
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/v1/chat` - Main chat endpoint
- `POST /api/v1/search` - Document search endpoint
- `GET /api/v1/session/{session_id}` - Get session history
- `DELETE /api/v1/session/{session_id}` - Delete a session

## Environment Variables

Create a `.env` file with the following variables:

```env
# Qdrant Configuration
QDRANT_URL=https://f7896955-bf33-46c5-b4e8-454d2cb95879.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content

# Cohere Configuration
COHERE_API_KEY=your_cohere_api_key
EMBEDDING_MODEL=embed-english-v3.0

# PostgreSQL Configuration
DATABASE_URL=your_neon_postgres_connection_string

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Application Configuration
APP_NAME=RAG Chatbot
DEBUG=True
```

## Usage

### Chat Endpoint
```json
{
  "message": "Your question here",
  "session_id": "optional_session_id",
  "selected_text": "Optional text that user has highlighted",
  "history": [
    {
      "role": "user",
      "content": "Previous message"
    }
  ]
}
```

### Search Endpoint
```json
{
  "query": "Your search query",
  "limit": 5,
  "selected_text": "Optional text to search for similar content"
}
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
.
├── src/
│   ├── main.py             # FastAPI application entry point
│   ├── api/
│   │   └── routers.py      # API routes
│   ├── models/
│   │   ├── chat.py         # Chat data models
│   │   └── database.py     # Database models
│   ├── services/
│   │   ├── chat_service.py # Main chat service
│   │   ├── database.py     # Database service
│   │   ├── vector_store.py # Qdrant service
│   │   └── embedding_service.py # Cohere service
│   ├── config/
│   │   └── settings.py     # Configuration settings
│   └── init_db.py          # Database initialization
├── tests/
│   ├── test_chat_service.py # Chat service tests
│   └── test_api.py         # API tests
├── requirements.txt        # Dependencies
├── .env.example           # Environment variables example
└── README.md              # This file
```

## Development

1. Install dependencies: `pip install -r requirements.txt`
2. Install dev dependencies: `pip install pytest pytest-asyncio`
3. Run with auto-reload: `uvicorn src.main:app --reload`
4. Run tests: `python -m pytest tests/`

## Architecture

The application follows a service-oriented architecture:

- **API Layer**: FastAPI handles HTTP requests and responses
- **Service Layer**: Business logic for chat processing, vector storage, and database operations
- **Data Layer**: PostgreSQL for chat history and Qdrant for vector storage
- **AI Layer**: Cohere for embeddings and OpenAI for generation