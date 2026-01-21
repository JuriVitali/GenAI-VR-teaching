import os
from services.rag_manager import RagManager

PERSIST_ROOT = os.getenv("RAG_PERSIST_ROOT")

rag_manager = RagManager(
    persist_root=PERSIST_ROOT,
    embedding_model=os.getenv("RAG_EMBED_MODEL", "nomic-embed-text"),
)
