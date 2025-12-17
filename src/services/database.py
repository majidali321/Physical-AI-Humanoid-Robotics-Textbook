from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime
import logging

from src.models.database import ChatSessionDB, ChatMessageDB
from src.config.settings import settings
from src.models.chat import Message, MessageRole


logger = logging.getLogger(__name__)

# Create async engine and session
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    pool_pre_ping=True,  # Verify connections before use
    pool_size=5,
    max_overflow=10
)
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class DatabaseService:
    def __init__(self):
        pass

    async def get_session(self, session_id: str) -> Optional[ChatSessionDB]:
        """Get a chat session by session_id"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(ChatSessionDB).where(ChatSessionDB.session_id == session_id)
                )
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting session {session_id}: {str(e)}")
                return None

    async def create_session(self, session_id: str, user_id: Optional[str] = None) -> ChatSessionDB:
        """Create a new chat session"""
        async with AsyncSessionLocal() as session:
            try:
                db_session = ChatSessionDB(session_id=session_id, user_id=user_id)
                session.add(db_session)
                await session.commit()
                await session.refresh(db_session)
                return db_session
            except Exception as e:
                logger.error(f"Error creating session {session_id}: {str(e)}")
                await session.rollback()
                raise

    async def get_messages(self, session_id: str) -> List[ChatMessageDB]:
        """Get all messages for a session"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(ChatMessageDB)
                    .where(ChatMessageDB.session_id == session_id)
                    .order_by(ChatMessageDB.timestamp)
                )
                return result.scalars().all()
            except Exception as e:
                logger.error(f"Error getting messages for session {session_id}: {str(e)}")
                return []

    async def add_message(self, session_id: str, role: str, content: str) -> ChatMessageDB:
        """Add a message to a session"""
        async with AsyncSessionLocal() as session:
            try:
                # Ensure session exists
                session_exists = await self.get_session(session_id)
                if not session_exists:
                    await self.create_session(session_id)

                db_message = ChatMessageDB(
                    session_id=session_id,
                    role=role,
                    content=content
                )
                session.add(db_message)
                await session.commit()
                await session.refresh(db_message)
                return db_message
            except Exception as e:
                logger.error(f"Error adding message to session {session_id}: {str(e)}")
                await session.rollback()
                raise

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        async with AsyncSessionLocal() as session:
            try:
                session_obj = await self.get_session(session_id)
                if session_obj:
                    await session.delete(session_obj)
                    await session.commit()
                    return True
                return False
            except Exception as e:
                logger.error(f"Error deleting session {session_id}: {str(e)}")
                await session.rollback()
                return False

    async def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ChatSessionDB:
        """Get a session or create it if it doesn't exist"""
        session_obj = await self.get_session(session_id)
        if not session_obj:
            session_obj = await self.create_session(session_id, user_id)
        return session_obj