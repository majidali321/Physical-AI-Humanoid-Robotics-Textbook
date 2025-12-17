import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from src.models.database import Base
from src.config.settings import settings


async def init_db():
    """Initialize the database and create tables"""
    engine = create_async_engine(settings.DATABASE_URL)

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("Database tables created successfully!")


if __name__ == "__main__":
    asyncio.run(init_db())