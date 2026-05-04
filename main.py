"""FastAPI server for YouTube RAG Chatbot.

Provides endpoints to load YouTube videos and ask questions about their content.
Uses in-memory caching for both agent pipelines and question answers.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from agents import create_agent, ask_agent, get_video_id

app = FastAPI(title="YouTube RAG Chatbot", version="1.0")

# In-memory storage for RAG agents (video_id -> pipeline)
AGENTS = {}

# Cache for question answers to avoid redundant processing
QUESTION_CACHE = {}


class VideoRequest(BaseModel):
    """Request model for loading a YouTube video."""
    url: str


class QueryRequest(BaseModel):
    """Request model for asking a question about a video."""
    video_id: str
    question: str


@app.post("/load_video")
def load_video(req: VideoRequest):
    """Load a YouTube video and create a RAG pipeline for it.
    
    Args:
        req: VideoRequest containing the YouTube URL
        
    Returns:
        dict: video_id and cached status
    """
    video_id = get_video_id(req.url)

    # Return cached agent if already processed
    if video_id in AGENTS:
        return {"video_id": video_id, "cached": True}

    # Create new agent and store in memory
    agent = create_agent(req.url)
    AGENTS[video_id] = agent

    return {"video_id": video_id, "cached": False}



@app.post("/ask")
def ask(req: QueryRequest):
    """Ask a question about a loaded video's content.
    
    Args:
        req: QueryRequest containing video_id and question
        
    Returns:
        dict: answer and cached status, or error message
    """
    # Verify video has been loaded
    if req.video_id not in AGENTS:
        return {"error": "Video not loaded"}
    
    # Create normalized cache key from video_id and normalized question
    cache_key = (req.video_id, req.question.strip().lower())
    
    # Return cached answer if available
    if cache_key in QUESTION_CACHE:
        return {"answer": QUESTION_CACHE[cache_key], "cached": True}

    # Generate answer using RAG pipeline
    agent = AGENTS[req.video_id]
    answer = ask_agent(agent, req.question)
    
    # Store in cache for future queries
    QUESTION_CACHE[cache_key] = answer

    return {"answer": answer, "cached": False}