"""
Central place for all configuration. Every other module imports from here
instead of calling os.getenv() directly, so there's exactly one source of truth.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the same folder (if present)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

# Folder the backend reads PDFs from. There is no upload endpoint - documents only
# enter the system by being placed in this folder (by whoever runs the backend),
# then either restarting the server or calling POST /ingest.
DATA_DIR = os.getenv("DATA_DIR", "../data")

CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

TOP_K = int(os.getenv("TOP_K", "4"))
