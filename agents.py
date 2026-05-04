"""RAG pipeline for YouTube video analysis.

Handles video transcript extraction, translation, embedding generation,
and RAG-based question answering using local LLM models via Ollama.
"""

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api._errors import NoTranscriptFound, NotTranslatable

import hashlib

# Cache for processed video agents (video_id -> RAG pipeline)
VIDEO_CACHE = {}

# LLM model for translating non-English transcripts
translator_llm = OllamaLLM(model="qwen2.5:1.5b")

def translate_to_english(text: str) -> str:
    """Translate non-English text to English using Ollama LLM.
    
    Args:
        text: Text to translate
        
    Returns:
        Translated text, or original text if translation fails
    """
    try:
        prompt = f"""Translate the following text to clear English.
Keep meaning same. Do NOT add extra info.

Text:
{text}"""
        return translator_llm.invoke(prompt)
    except Exception:
        # Gracefully fallback to original text on translation failure
        return text

def get_video_id(url: str) -> str:
    """Generate a unique identifier for a YouTube URL using MD5 hash.
    
    Args:
        url: YouTube video URL
        
    Returns:
        MD5 hash of the URL as unique video identifier
    """
    return hashlib.md5(url.encode()).hexdigest()


def build_rag_pipeline(vectorstore):
    """Build a RAG pipeline for question answering.
    
    Args:
        vectorstore: ChromaDB vector store with embedded documents
        
    Returns:
        Configured RAG pipeline (retrieval -> prompt -> LLM -> output parsing)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = OllamaLLM(model="qwen2.5:7b")

    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.

Answer ONLY using the context below.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{question}""")

    def format_docs(docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Compose RAG pipeline: retrieve context, format, prompt, generate, parse
    pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return pipeline


 

def load_youtube_docs(url: str):
    """Extract and process YouTube transcript.
    
    1. Attempts to load transcript with auto-translation
    2. Falls back to transcript without translation if API translation fails
    3. Applies manual translation for non-English content using Ollama
    
    Args:
        url: YouTube video URL
        
    Returns:
        List of Document objects with translated content
        
    Raises:
        ValueError: If video has no captions or transcript is empty
    """
    try:
        try:
            # Attempt to load transcript with auto-translation
            loader = YoutubeLoader.from_youtube_url(
                url, language=["en", "hi", "auto"], translation="en"
            )
            docs = loader.load()
        except (NotTranslatable,Exception):
            # Fallback: load without auto-translation
            loader = YoutubeLoader.from_youtube_url(
                url, language=["en", "hi", "auto"]
            )
            docs = loader.load()

        if not docs:
            raise ValueError("Empty transcript")

        # Apply manual translation for non-English content
        translated_docs = []
        for doc in docs:
            content = doc.page_content
            # Only translate if content contains non-ASCII characters
            if not content.isascii():
                content = translate_to_english(content)
            doc.page_content = content
            translated_docs.append(doc)

        return translated_docs

    except NoTranscriptFound:
        raise ValueError(
            "No captions available for this video. Please try another video with subtitles."
        )


def create_agent(youtube_url: str):
    """Create a RAG agent for a YouTube video with caching.
    
    Pipeline:
    1. Load and translate transcript
    2. Split text into chunks
    3. Generate embeddings (local Ollama)
    4. Store in ChromaDB vector store
    5. Build RAG query pipeline
    
    Args:
        youtube_url: YouTube video URL
        
    Returns:
        RAG pipeline ready for question answering
    """
    video_id = get_video_id(youtube_url)

    # Return cached pipeline if already processed
    if video_id in VIDEO_CACHE:
        return VIDEO_CACHE[video_id]

    # Load and translate transcript
    docs = load_youtube_docs(youtube_url)

    # Split documents into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    # Generate embeddings using local Ollama model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create vector database with embeddings
    vectorstore = Chroma.from_documents(
        split_docs, embeddings, collection_name=video_id
    )

    # Build RAG pipeline for this video
    rag_pipeline = build_rag_pipeline(vectorstore)

    # Cache the pipeline for reuse
    VIDEO_CACHE[video_id] = rag_pipeline

    return rag_pipeline


def ask_agent(agent, question: str) -> str:
    """Ask a question to the RAG agent.
    
    Args:
        agent: RAG pipeline
        question: Question to answer
        
    Returns:
        Answer generated by the RAG pipeline, or error message
    """
    try:
        return agent.invoke(question)
    except Exception as e:
        return "An error occurred while processing your question."


if __name__ == "__main__":
    # Test the agent with a sample YouTube video
    agent = create_agent("https://www.youtube.com/watch?v=ukzFI9rgwfU")
    print(ask_agent(agent, "Who is the creator of this video?"))