import os
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class RagService:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        collection_name: str = "pdf_knowledge",
        ollama_embedding_model: str = "nomic-embed-text",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ):
        self.persist_dir = os.getenv("RAG_PERSIST_DIR", persist_dir)
        self.collection_name = collection_name

        os.makedirs(self.persist_dir, exist_ok=True)

        self.embeddings = OllamaEmbeddings(model=ollama_embedding_model)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.vdb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


    def build_index(self, pdf_path: str) -> int:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        # Dev-mode: ricrea tutta la collection
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

    def retrieve_context(
        self,
        question: str,
        k: int = 5,
        use_mmr: bool = True,
        fetch_k: int = 20,
    ) -> Tuple[str, List[dict]]:
        if use_mmr:
            retriever = self.vdb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
        else:
            retriever = self.vdb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "", []

        context_str = "\n\n---\n\n".join(
            f"[p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )

        sources = [
            {"page": d.metadata.get("page", None), "source": d.metadata.get("source", None)}
            for d in docs
        ]
        return context_str, sources
