import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat_router
from src.init_db import init_db
from src.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if settings.DEBUG:
        await init_db()
    yield
    # Shutdown
    # Add any cleanup code here if needed


app = FastAPI(
    title="RAG Chatbot API",
    description="An API for interacting with a RAG-based chatbot for digital books",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}