import os
import structlog
from services.rag_service import RagService

log = structlog.get_logger()

rag = RagService(
    persist_dir="./chroma_store",
    collection_name="pdf_knowledge",
    ollama_embedding_model="nomic-embed-text",
)

_is_ready = False


def init_rag_from_env() -> bool:
    """Indicizza il PDF se RAG_PDF_PATH è valido. Non deve mai crashare l'app."""
    global _is_ready

    pdf_path = os.getenv("RAG_PDF_PATH", "").strip()

    if not pdf_path:
        log.warning("rag_init_skipped", reason="RAG_PDF_PATH missing")
        _is_ready = False
        return False

    if not os.path.exists(pdf_path):
        log.warning("rag_init_skipped", reason="pdf not found", pdf_path=pdf_path)
        _is_ready = False
        return False

    try:
        n_chunks = rag.build_index(pdf_path)
        log.info("rag_index_built", pdf_path=pdf_path, chunks=n_chunks)
        _is_ready = True
        return True
    except Exception:
        log.exception("rag_index_failed", pdf_path=pdf_path)
        _is_ready = False
        return False


def rag_ready() -> bool:
    return _is_ready
