import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "https://f7896955-bf33-46c5-b4e8-454d2cb95879.europe-west3-0.gcp.cloud.qdrant.io")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6jyatLdOi0JSm_K75ZZb4BBVi_7YJ6WZKrrn_-IqSKM")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

    # Cohere Configuration
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "LR9E3trjxOQxStV0sDLqxyYKIVJyRbohsZlT9aPQ")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")

    # PostgreSQL Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_5CXpMAVgFtO1@ep-dark-haze-abairq5s-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Application Configuration
    APP_NAME: str = os.getenv("APP_NAME", "RAG Chatbot")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()