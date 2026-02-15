import os
import threading
from typing import Tuple, List, Optional
import structlog
from services.rag_service import RagService

log = structlog.get_logger()

class RagManager:
    def __init__(self, persist_root: str, embedding_model: str, pdf_dir: Optional[str] = None):
        self.persist_root = persist_root
        self.embedding_model = embedding_model
        self.pdf_dir = pdf_dir or os.getenv("PDF_DIR")

        self._lock = threading.Lock()
        self._services = {}
        self._status = {} 
        self._error = {}

    def get_status(self, pdf_name: str) -> Tuple[str, Optional[str]]:
        return self._status.get(pdf_name, "missing"), self._error.get(pdf_name)

    def _make_service(self, pdf_name: str) -> RagService:
        collection = f"pdf_knowledge_{pdf_name}"
        persist_dir = os.path.join(self.persist_root, collection)
        os.makedirs(persist_dir, exist_ok=True)

        return RagService(
            persist_dir=persist_dir,
            collection_name=collection,
            ollama_embedding_model=self.embedding_model,
        )

    def ensure_ready(self, pdf_name: str, session_id: str = None) -> Tuple[bool, str]:
        if session_id:
            structlog.contextvars.bind_contextvars(session_id=session_id)
        
        pdf_path = os.path.join(self.pdf_dir, f"{pdf_name}.pdf")

        with self._lock:
            status = self._status.get(pdf_name, "missing")
            if status == "ready":
                return True, "ready"
            if status == "building":
                return False, "building"
            
            # Impostiamo lo stato a building per evitare elaborazioni parallele
            self._status[pdf_name] = "building"
            self._error[pdf_name] = None

        # Verifica se il file esiste fisicamente sul disco
        if not os.path.exists(pdf_path):
            with self._lock:
                self._status[pdf_name] = "error"
                self._error[pdf_name] = f"pdf_not_found:{pdf_path}"
            return False, "pdf_not_found"

        try:
            svc = self._make_service(pdf_name)
            n_chunks = svc.build_index(pdf_path)

            with self._lock:
                self._services[pdf_name] = svc
                self._status[pdf_name] = "ready"

            log.info("rag_index_built", pdf=pdf_name, chunks=n_chunks)
            return True, "ready"

        except Exception as e:
            with self._lock:
                self._status[pdf_name] = "error"
                self._error[pdf_name] = str(e)
            log.exception("rag_index_failed", pdf=pdf_name)
            return False, "error"

    def retrieve_context(self, pdf_name: str, question: str, k: int = 5) -> Tuple[str, List[dict]]:
        svc = self._services.get(pdf_name)
        if not svc:
            return "", []
        return svc.retrieve_context(question, k=k)