import os
import re
import json
import structlog
import time
from shared.utils import log_event
from typing import List, Tuple, Optional, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
from config.model_config_loader import ModelConfig
from services.llm_service import get_llm_model
from pydantic import BaseModel, Field
from typing import Literal

# Define exact valid categories
class IntentCategory(BaseModel):
    intent: Literal["definition", "comparison", "overview", "general_question"] = Field(
        ..., 
        description="The semantic category of the user's question"
    )

load_dotenv(find_dotenv())

class RagService:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        ollama_embedding_model: str = os.getenv("RAG_EMBED_MODEL"),
        chunk_size: int = 900,
        chunk_overlap: int = 250
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = OllamaEmbeddings(model=ollama_embedding_model)

        # Configurazione Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            add_start_index=True
        )

        self.vdb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        # --- CARICAMENTO CONFIGURAZIONE LLM ROUTER ---
        config_loader = ModelConfig()
        try:
            router_config = config_loader.get("intent_classifier")
            self.router_model = get_llm_model(
                router_config["model"], 
                router_config["temperature"]
            )
            self.router_prompt_template = router_config["prompt"]
        except Exception as e:
            logger.error("rag_router_config_error", error=str(e))
            # Fallback hardcoded se il config fallisce
            self.router_model = None 

    @staticmethod
    def _clean_text(text: str) -> str:
        """Pulisce il testo estratto dal PDF."""
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r", "", text)
        return text.strip()

    def _extract_json_from_response(self, text: str) -> dict:
        """
        Pulisce la risposta dell'LLM
        e prova a parsare il JSON.
        """
        try:
            # 1. Rimuovi i tag <think>...</think> se presenti
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            
            # 2. Cerca il primo '{' e l'ultimo '}'
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                # Tentativo disperato se non trova graffe
                return json.loads(text)
        except Exception as e:
            logger.warning("router_json_parse_failed", raw_text=text, error=str(e))
            return None

    @log_event("intent_classification", result_mapper=lambda x: {"intent": x})
    def classify_intent(self, question: str) -> str:
        if not self.router_model:
            return "general_question"

        try:
            # Create a version of the model that MUST return the Pydantic object
            # Note: If your model provider (e.g., Ollama) supports it, this enforces JSON mode.
            structured_llm = self.router_model.with_structured_output(IntentCategory)

            
            # Invoke directly. The prompt just needs the instructions, 
            # you can remove the "Respond ONLY with JSON" part from the prompt template.
            intent_obj = structured_llm.invoke(self.router_prompt_template.format(question=question))

            # Access the data directly as a python attribute
            return intent_obj.intent

        except Exception as e:
            logger.error(f"router_inference_failed: {e}")
            return "general_question"
    
    def build_index(self, pdf_path: str) -> int:
        try:
            existing_count = self.vdb._collection.count()
            if existing_count > 0:
                print(f"[RAG] Index già esistente ({existing_count} chunks). Salto il rebuild per: {pdf_path}")
                return existing_count
        except Exception as e:
            print(f"[RAG] Errore verifica index esistente: {e}, rebuild.")
    
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        
        for doc in raw_docs:
            doc.page_content = self._clean_text(doc.page_content)
            
        chunks = self.splitter.split_documents(raw_docs)

        for i, c in enumerate(chunks):
            c.metadata = c.metadata or {}
            c.metadata["chunk_index"] = i
            if "page" in c.metadata:
                try:
                    c.metadata["page"] = int(c.metadata["page"])
                except Exception:
                    pass

        try:
            self.vdb.delete_collection()
        except Exception:
            pass

        self.vdb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        self.vdb.add_documents(chunks)

        return len(chunks)

    def _get_by_chunk_index(self, idx: int) -> Optional[Document]:
        try:
            res = self.vdb._collection.get(where={"chunk_index": idx})
        except Exception:
            return None

        if not res or not res.get("documents"):
            return None

        return Document(
            page_content=res["documents"][0],
            metadata=res["metadatas"][0],
        )

    @staticmethod
    def _dedupe(docs: List[Document]) -> List[Document]:
        seen: Set[int] = set()
        out: List[Document] = []

        for d in docs:
            ci = d.metadata.get("chunk_index")
            if ci in seen:
                continue
            seen.add(ci)
            out.append(d)

        return out

    @staticmethod
    def _sort(docs: List[Document]) -> List[Document]:
        def key(d: Document):
            page = d.metadata.get("page")
            ci = d.metadata.get("chunk_index")
            try:
                page = int(page)
            except Exception:
                page = 10**9
            try:
                ci = int(ci)
            except Exception:
                ci = 10**9
            return (page, ci)

        return sorted(docs, key=key)

    @log_event("rag_retrieval", result_mapper=lambda x: {"sources_count": len(x[1])})
    def retrieve_context(
        self,
        question: str,
        k: int = 5,
        fetch_k: int = 20,
    ) -> Tuple[str, List[dict]]:

        intent = self.classify_intent(question)

        # Logica di retrieval invariata
        if intent in ("definition", "general_question"):
            search_type = "similarity"
            k_eff = max(4, k)
            do_window = intent == "general_question"
        else:
            search_type = "mmr"
            k_eff = max(8, k)
            do_window = False

        if search_type == "mmr":
            retriever = self.vdb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_eff, "fetch_k": max(fetch_k, 40)},
            )
        else:
            retriever = self.vdb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_eff},
            )

        docs = retriever.invoke(question)
        if not docs:
            return "", []

        if do_window:
            expanded = list(docs)
            for d in docs:
                ci = d.metadata.get("chunk_index")
                if ci is None:
                    continue
                for ni in (ci - 1, ci + 1):
                    if ni >= 0:
                        nd = self._get_by_chunk_index(ni)
                        if nd:
                            expanded.append(nd)
            docs = expanded

        docs = self._dedupe(docs)
        docs = self._sort(docs)

        context = "\n\n---\n\n".join(
            f"[p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )

        sources = [
            {
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source"),
                "chunk_index": d.metadata.get("chunk_index"),
            }
            for d in docs
        ]

        return context, sources