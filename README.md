# YouTube RAG Chatbot API

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot that extracts insights from YouTube videos. Load any YouTube video and ask questions about its content using local LLM models via Ollama.

## Features

- **YouTube Transcript Extraction**: Automatically fetches transcripts from YouTube videos
- **Multi-language Support**: Handles transcripts in English, Hindi, and auto-detects other languages
- **Intelligent Translation**: Uses lightweight Ollama LLM to translate non-English transcripts to English
- **Vector Database**: Stores embeddings using ChromaDB for efficient semantic search
- **Local LLM**: Uses Ollama for both embeddings and generation (no external API calls)
- **Caching**: In-memory caching for videos and question answers to improve performance
- **RAG Pipeline**: Retrieves relevant context from transcripts and generates accurate answers

## Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running with models:
  - `qwen2.5:1.5b` (for translation)
  - `qwen2.5:7b` (for question answering)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GuyOnAKeyboard/YouTube-RAG-Chatbot-API-with-FastAPI-ChromaDB-Local-LLM-Ollama-.git
cd YouTube-RAG-Chatbot-API-with-FastAPI-ChromaDB-Local-LLM-Ollama-
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Load Video
**POST** `/load_video`

Load a YouTube video for analysis.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**Response:**
```json
{
  "video_id": "hash_of_url",
  "cached": false
}
```

### 2. Ask Question
**POST** `/ask`

Ask a question about a loaded video's content.

**Request:**
```json
{
  "video_id": "hash_of_url",
  "question": "What is the main topic discussed?"
}
```

**Response:**
```json
{
  "answer": "The video discusses...",
  "cached": false
}
```

## Architecture

```
YouTube Video
    ↓
[Extract Transcript]
    ↓
[Translate to English] ← Uses Ollama qwen2.5:1.5b
    ↓
[Split & Chunk Text]
    ↓
[Generate Embeddings] ← Uses Ollama
    ↓
[Store in ChromaDB]
    ↓
[RAG Query Pipeline]
    ↓
Answer Generation ← Uses Ollama qwen2.5:7b
```

### Components

- **agents.py**: RAG pipeline logic, video loading, document processing
- **main.py**: FastAPI endpoints and request handling

## Key Notes

### Translation Behavior
- Translation only occurs when a transcript is in a language other than English
- The system uses Ollama (qwen2.5:1.5b) to translate non-English transcripts to English for better LLM compatibility
- If translation fails or is unavailable, the system falls back to the original transcript
- Translation quality depends on the Ollama model used
- **This is acceptable** — the system gracefully handles translation failures without breaking

## Future Updates & Improvements

1. **Multiple LLM Models**
   - Support for additional Ollama models
   - Model selection via API parameter

2. **Advanced Caching**
   - Persistent caching (Redis/PostgreSQL)
   - Cache TTL management
   - Cache statistics and monitoring

3. **Enhanced Translation**
   - Support for more languages
   - Improved translation accuracy
   - Language-specific prompts

4. **Vector DB Improvements**
   - Metadata filtering (video title, date, speaker)
   - Hybrid search (keyword + semantic)
   - Database persistence across sessions

5. **API Enhancements**
   - User authentication and API keys
   - Rate limiting
   - Request logging and analytics
   - WebSocket support for real-time streaming responses

6. **UI/Frontend**
   - Web interface for easy video upload and querying
   - Chat history and saved conversations
   - Export answers to PDF/Markdown

7. **Quality Improvements**
   - Unit and integration tests
   - Error handling improvements
   - Structured logging
   - Docker containerization

8. **Scalability**
   - Async request processing (Celery/RQ)
   - Horizontal scaling support
   - Load balancing
   - Queue-based video processing

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

You are free to:
- Use the project for personal or commercial purposes
- Modify and distribute the code
- Use privately or publicly

Just include the original license notice in distributions.

### Commercial Use Attribution
If you are earning money from any commercial use or deployment of this project, you must provide clear attribution to the original creator. Please mention the original repository and author in your application's documentation, credits, or appropriate attribution section.

## Technologies Used

- **FastAPI** - Modern web framework
- **LangChain** - LLM orchestration and RAG
- **ChromaDB** - Vector database
- **Ollama** - Local LLM runtime
- **youtube-transcript-api** - YouTube transcript extraction
- **Pydantic** - Data validation

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Support

For questions or issues, please open an issue on the GitHub repository.

---

**Happy questioning!**
