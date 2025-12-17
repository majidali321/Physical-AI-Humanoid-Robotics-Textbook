import cohere
from typing import List
import numpy as np
import logging

from src.config.settings import settings


logger = logging.getLogger(__name__)


class CohereEmbeddingService:
    def __init__(self):
        # Initialize Cohere client
        self.client = cohere.Client(settings.COHERE_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query"  # Using search_query for user queries
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # Cohere has a limit on batch size, so we'll process in chunks if needed
            max_batch_size = 96  # Cohere's recommended max batch size

            all_embeddings = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type
                )
                all_embeddings.extend([embedding for embedding in response.embeddings])

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for texts: {str(e)}")
            raise

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for documents (using search_document input type)"""
        return self.embed_texts(documents, input_type="search_document")

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (using search_query input type)"""
        return self.embed_text(query)

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embeddings for the current model"""
        # For Cohere's embed-english-v3.0, the dimensions are 1024
        return 1024