from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import uuid

from src.config.settings import settings


class VectorStoreService:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=True
        )

        # Collection name
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the Qdrant collection with appropriate configuration"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            # Using 1024 dimensions as specified for Cohere embed-english-v3.0
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )

    def add_document(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Add a document to the vector store"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # This method would typically be called with embeddings from Cohere
        # For now, we'll return the doc_id
        # In a real implementation, we'd need to pass the embedding vector here
        return doc_id

    def add_documents_with_embeddings(self, texts: List[str], embeddings: List[List[float]],
                                    metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add multiple documents with their embeddings to the vector store"""
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Prepare points for insertion
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {
                "content": text,
                "metadata": metadatas[i] if metadatas else {}
            }
            points.append(models.PointStruct(
                id=ids[i],
                vector=embedding,
                payload=payload
            ))

        # Insert points into the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def search(self, query_vector: List[float], limit: int = 5,
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store"""
        search_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                )
            if must_conditions:
                search_filter = models.Filter(must=must_conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_filter
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "content": result.payload.get("content", ""),
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })

        return formatted_results

    def search_by_text(self, query_text: str, cohere_service, limit: int = 5,
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using text query - generates embedding internally"""
        # Generate embedding for the query text
        query_embedding = cohere_service.embed_text(query_text)

        # Perform search with the generated embedding
        return self.search(query_embedding, limit, filters)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[doc_id])
            )
            return True
        except Exception:
            return False

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )
            if records:
                record = records[0]
                return {
                    "id": record.id,
                    "content": record.payload.get("content", ""),
                    "metadata": record.payload.get("metadata", {})
                }
        except Exception:
            pass
        return None

    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all documents from the collection (up to limit)"""
        try:
            records = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit
            )[0]  # scroll returns (records, next_page_offset)

            documents = []
            for record in records:
                documents.append({
                    "id": record.id,
                    "content": record.payload.get("content", ""),
                    "metadata": record.payload.get("metadata", {})
                })
            return documents
        except Exception:
            return []