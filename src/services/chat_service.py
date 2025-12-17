import asyncio
from typing import List, Optional
from openai import OpenAI
import os

from src.models.chat import ChatRequest, ChatResponse, Message, MessageRole, SearchResult
from src.services.database import DatabaseService
from src.services.vector_store import VectorStoreService
from src.services.embedding_service import CohereEmbeddingService
from src.config.settings import settings


class ChatService:
    def __init__(self):
        # Initialize all required services
        self.db_service = DatabaseService()
        self.vector_store = VectorStoreService()
        self.embedding_service = CohereEmbeddingService()

        # Initialize OpenAI client
        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            # For now, we'll handle this gracefully - in a real app you'd want to ensure this is set
            self.openai_client = None

    async def process_chat(self, message: str, session_id: str,
                          selected_text: Optional[str] = None,
                          history: List[Message] = None) -> ChatResponse:
        """Process a chat message and return a response"""
        # First, save the user message to the database
        await self.db_service.add_message(session_id, MessageRole.USER.value, message)

        # Determine context for RAG
        context_text = ""
        sources = []

        if selected_text:
            # Use selection-based RAG - prioritize the selected text
            # Search in vector store for similar content to the selected text
            query_embedding = self.embedding_service.embed_query(selected_text)
            search_results = self.vector_store.search(query_embedding, limit=3)

            # Build context starting with the selected text as primary context
            context_text = f"PRIMARY CONTEXT (Selected Text):\n{selected_text}\n\n"

            # Add similar content from the vector store
            for result in search_results:
                context_text += f"RELATED CONTENT:\n{result['content']}\n\n"
                sources.append(result.get("metadata", {}).get("source", "Unknown source"))
        else:
            # Regular RAG - search based on the user's query
            query_embedding = self.embedding_service.embed_query(message)
            search_results = self.vector_store.search(query_embedding, limit=5)

            # Combine search results for context
            for result in search_results:
                context_text += f"{result['content']}\n\n"
                sources.append(result.get("metadata", {}).get("source", "Unknown source"))

        # Prepare the conversation history for context
        conversation_context = ""
        if history:
            for msg in history:
                conversation_context += f"\n{msg.role.value.capitalize()}: {msg.content}"

        # Generate response using OpenAI
        if selected_text:
            # Special prompt for selection-based RAG to emphasize using the selected text
            full_prompt = f"""
            You are a helpful assistant for a digital book. Answer the user's question based STRICTLY on the PRIMARY CONTEXT provided below.
            If the PRIMARY CONTEXT doesn't contain relevant information, say so.
            Do not fabricate information or rely heavily on the RELATED CONTENT if it contradicts the PRIMARY CONTEXT.

            PRIMARY CONTEXT (Selected Text):
            {selected_text}

            RELATED CONTENT (for additional context only):
            {"".join([f"- {result['content']}\n" for result in search_results[:2]]) if selected_text else ''}

            Conversation History:
            {conversation_context}

            User's Question: {message}

            Please provide a helpful response based strictly on the PRIMARY CONTEXT and cite sources when possible.
            """
        else:
            # Standard RAG prompt
            full_prompt = f"""
            You are a helpful assistant for a digital book. Use the following context to answer the user's question.
            If the context doesn't contain relevant information, say so.

            Context:
            {context_text}

            Conversation History:
            {conversation_context}

            User's Question: {message}

            Please provide a helpful response based on the context and cite sources when possible.
            """

        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",  # You can change this to gpt-4 if preferred
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content
            except Exception as e:
                ai_response = f"Sorry, I encountered an error processing your request: {str(e)}"
        else:
            # Fallback response if OpenAI key is not configured
            ai_response = "OpenAI API key not configured. In a real implementation, this would generate a response based on the context provided."

        # Save the assistant's response to the database
        await self.db_service.add_message(session_id, MessageRole.ASSISTANT.value, ai_response)

        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            sources=sources
        )

    async def search_documents(self, query: str, limit: int = 5,
                              selected_text: Optional[str] = None) -> List[SearchResult]:
        """Search for relevant documents in the vector store"""
        if selected_text:
            # If selected text is provided, search based on that
            query_embedding = self.embedding_service.embed_query(selected_text)
        else:
            # Otherwise search based on the original query
            query_embedding = self.embedding_service.embed_query(query)

        search_results = self.vector_store.search(query_embedding, limit=limit)

        # Convert to SearchResult objects
        results = []
        for result in search_results:
            results.append(SearchResult(
                id=result["id"],
                content=result["content"],
                score=result["score"],
                metadata=result["metadata"]
            ))

        return results

    async def get_session_history(self, session_id: str) -> List[Message]:
        """Retrieve chat history for a session"""
        db_messages = await self.db_service.get_messages(session_id)

        messages = []
        for db_msg in db_messages:
            messages.append(Message(
                role=MessageRole(db_msg.role),
                content=db_msg.content,
                timestamp=db_msg.timestamp
            ))

        return messages

    async def delete_session(self, session_id: str):
        """Delete a chat session"""
        await self.db_service.delete_session(session_id)