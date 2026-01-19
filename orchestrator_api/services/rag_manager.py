import os
import threading
import structlog
from typing import Optional, Tuple, List

from config.pdf_map import PDF_MAP
from services.rag_service import RagService

log = structlog.get_logger()

class RagManager:
    def __init__(self, persist_root: str, embedding_model: str = "nomic-embed-text"):
        self.persist_root = persist_root
        self.embedding_model = embedding_model

        self._lock = threading.Lock()
        self._services = {}  # pdf_name -> RagService
        self._status = {k: "missing" for k in PDF_MAP.keys()}  # missing|building|ready|error
        self._error = {k: None for k in PDF_MAP.keys()}

    def get_status(self, pdf_name: str) -> Tuple[str, Optional[str]]:
        return self._status.get(pdf_name, "error"), self._error.get(pdf_name, "unknown_pdf")

    def _make_service(self, pdf_name: str) -> RagService:
        collection = f"pdf_knowledge_{pdf_name}"
        persist_dir = os.path.join(self.persist_root, collection)
        os.makedirs(persist_dir, exist_ok=True)

        return RagService(
            persist_dir=persist_dir,
            collection_name=collection,
            ollama_embedding_model=self.embedding_model,
        )

    def ensure_ready(self, pdf_name: str) -> Tuple[bool, str]:
        if pdf_name not in PDF_MAP:
            return False, "unknown_pdf"

        with self._lock:
            st = self._status[pdf_name]
            if st == "ready":
                return True, "ready"
            if st == "building":
                return False, "building"
            self._status[pdf_name] = "building"
            self._error[pdf_name] = None

        pdf_path = PDF_MAP[pdf_name]
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
                self._error[pdf_name] = None

            log.info("rag_index_built", pdf_name=pdf_name, chunks=n_chunks, pdf_path=pdf_path)
            return True, "ready"
        except Exception as e:
            with self._lock:
                self._status[pdf_name] = "error"
                self._error[pdf_name] = str(e)
            log.exception("rag_index_failed", pdf_name=pdf_name)
            return False, "error"

    def retrieve_context(self, pdf_name: str, question: str, k: int = 5) -> Tuple[str, List[dict]]:
        svc = self._services.get(pdf_name)
        if svc is None:
            return "", []
        return svc.retrieve_context(question, k=k, use_mmr=True)
