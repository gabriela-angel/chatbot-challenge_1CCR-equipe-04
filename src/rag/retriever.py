from dataclasses import dataclass, field
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.guardrails.moderation import detectar_injection_em_documento
from src.rag.vector_store import open_vector_store


@dataclass
class RetrievalResult:
    documents: List[Document]
    context: str
    sources: List[dict]
    best_relevance_score: float | None
    scores: List[float] = field(default_factory=list)
    discarded_injection: int = 0

    @property
    def best_score(self) -> float | None:
        """Alias usado pelo engine."""
        return self.best_relevance_score


def format_source(document: Document) -> dict:
    metadata = document.metadata or {}
    page = metadata.get("page")
    try:
        # PyMuPDFLoader numera a primeira página como 0.
        display_page = int(page) + 1 if page is not None else None
    except (TypeError, ValueError):
        display_page = None
    return {
        "source": str(metadata.get("source", "fonte_desconhecida")),
        "page": display_page,
        "chunk_index": metadata.get("chunk_index"),
    }


class KnowledgeRetriever:
    """Recuperação semântica com limiar de relevância e filtro anti-injection."""

    def __init__(
        self,
        vector_store: Chroma | None = None,
        persist_directory=None,
        collection_name: str = "goodwe_knowledge_base",
        k: int = 4,
        min_relevance_score: float = 0.20,
        min_score: float | None = None,
    ):
        # `min_score` é o nome usado pelo engine; mantido como alias.
        if min_score is not None:
            min_relevance_score = min_score
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
            self.vector_store = open_vector_store(collection_name=collection_name)
        self.k = k
        self.min_relevance_score = min_relevance_score

    def retrieve(self, question: str) -> RetrievalResult:
        query = (question or "").strip()
        if not query:
            return RetrievalResult([], "", [], None)

        pairs = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k
        )

        accepted = []
        discarded = 0
        for document, score in pairs:
            if float(score) < self.min_relevance_score:
                continue
            # Documentos são conteúdo não confiável: descarta trechos que
            # tentam dar ordens ao modelo.
            if detectar_injection_em_documento(document.page_content):
                discarded += 1
                continue
            accepted.append((document, float(score)))

        docs = [document for document, _ in accepted]
        scores = [score for _, score in accepted]

        context_parts = []
        source_index: dict[tuple, dict] = {}
        for document, score in accepted:
            source = format_source(document)
            context_parts.append(
                f"[Fonte: {source['source']}; página: {source['page'] or 'não identificada'}]\n"
                f"{document.page_content.strip()}"
            )
            key = (source["source"], source["page"])
            if key not in source_index:
                source["score"] = round(score, 4)
                source_index[key] = source

        return RetrievalResult(
            documents=docs,
            context="\n\n---\n\n".join(context_parts),
            sources=list(source_index.values()),
            best_relevance_score=max(scores, default=None),
            scores=scores,
            discarded_injection=discarded,
        )