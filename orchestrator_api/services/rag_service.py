import os
import re
from typing import List, Tuple, Optional, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class RagService:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        ollama_embedding_model: str = "nomic-embed-text",
        chunk_size: int = 900,
        chunk_overlap: int = 250
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = OllamaEmbeddings(model=ollama_embedding_model)
        

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

    @staticmethod
    def _clean_text(text: str) -> str:
        """Pulisce il testo estratto dal PDF dai tipici artefatti di scansione."""
        # Rimuove i trattini di divisione a fine riga
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        
        # Sostituisce i singoli a capo con uno spazio (mantenendo i doppi a capo per i paragrafi)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        
        # Rimuove spazi multipli e caratteri speciali invisibili
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r", "", text)
        
        return text.strip()

    @staticmethod
    def classify_intent(question: str) -> str:
        q = (question or "").strip().lower()
        q = re.sub(r"[’`´]", "'", q)

        if re.match(r"^(cos'è|che cos'è|definisci|che significa)", q):
            return "definition"

        if re.match(r"^(perché|perche|come|in che modo|spiega)", q):
            return "explanation"

        if re.search(r"\b(differenza|confronta|confronto|relazione|rapporto|vs)\b", q):
            return "comparison"

        if re.search(r"\b(riassumi|sintesi|panoramica|in generale|principali)\b", q):
            return "overview"

        return "explanation"

    def build_index(self, pdf_path: str) -> int:
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

    def retrieve_context(
        self,
        question: str,
        k: int = 5,
        fetch_k: int = 20,
    ) -> Tuple[str, List[dict]]:

        intent = self.classify_intent(question)

        if intent in ("definition", "explanation"):
            search_type = "similarity"
            k_eff = max(4, k)
            do_window = intent == "explanation"
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
