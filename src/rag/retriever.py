from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.vector_store import open_vector_store


@dataclass
class RetrievalResult:
    documents: List[Document]
    context: str
    sources: List[dict]
    best_relevance_score: float | None


def format_source(document: Document) -> dict:
    metadata = document.metadata or {}
    page = metadata.get("page")
    try:
        # PyMuPDFLoader costuma numerar a primeira página como 0.
        display_page = int(page) + 1 if page is not None else None
    except (TypeError, ValueError):
        display_page = None
    return {
        "source": str(metadata.get("source", "fonte_desconhecida")),
        "page": display_page,
        "chunk_index": metadata.get("chunk_index"),
    }


class KnowledgeRetriever:
    """Recuperação semântica com limiar configurável para contexto insuficiente."""

    def __init__(
        self,
        vector_store: Chroma | None = None,
        persist_directory=None,
        collection_name: str = "goodwe_knowledge_base",
        k: int = 4,
        min_relevance_score: float = 0.20,
    ):
        if k < 1:
            raise ValueError("k deve ser pelo menos 1.")
        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score deve estar entre 0 e 1.")
        if vector_store is not None:
            self.vector_store = vector_store
        elif persist_directory is not None:
            self.vector_store = open_vector_store(
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            self.vector_store = open_vector_store(
                collection_name=collection_name,
            )
        self.k = k
        self.min_relevance_score = min_relevance_score

    def retrieve(self, question: str) -> RetrievalResult:
        query = (question or "").strip()
        if not query:
            return RetrievalResult([], "", [], None)

        # relevance score é normalizado pela integração LangChain/Chroma; valores
        # podem variar conforme a métrica configurada. O limiar precisa ser calibrado
        # com perguntas reais e anotadas, não tratado como universal.
        pairs = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k
        )
        accepted = [
            (doc, float(score)) for doc, score in pairs
            if score >= self.min_relevance_score
        ]
        docs = [doc for doc, _ in accepted]
        best_score = max((score for _, score in accepted), default=None)
        context_parts = []
        seen_sources = set()
        sources = []
        for document in docs:
            source = format_source(document)
            context_parts.append(
                f"[Fonte: {source['source']}; página: {source['page'] or 'não identificada'}]\n"
                f"{document.page_content.strip()}"
            )
            key = (source["source"], source["page"])
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(source)

        return RetrievalResult(
            documents=docs,
            context="\n\n---\n\n".join(context_parts),
            sources=sources,
            best_relevance_score=best_score,
        )
