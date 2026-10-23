import hashlib
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.rag.embeddings import build_embeddings

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIRECTORY = BASE_DIR / "data" / "chroma_db"
DEFAULT_COLLECTION_NAME = "goodwe_knowledge_base"


def _document_id(document: Document) -> str:
    """ID determinístico: reindexar o mesmo chunk atualiza em vez de duplicar."""
    metadata = document.metadata or {}
    identity = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("start_index", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def open_vector_store(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: OllamaEmbeddings | None = None,
) -> Chroma:
    """Abre/cria uma coleção Chroma persistente sem apagar dados existentes."""
    path = Path(persist_directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    embedding_function = embeddings or build_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(path),
    )


def index_documents(
    chunks: Sequence[Document],
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: OllamaEmbeddings | None = None,
) -> Chroma:
    """Indexa chunks no ChromaDB persistente, com IDs determinísticos."""
    if not chunks:
        raise ValueError("Nenhum chunk recebido para indexação.")
    store = open_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    ids = [_document_id(document) for document in chunks]
    # Chroma usa upsert para IDs existentes; isso evita duplicação ao reindexar
    # os mesmos documentos. Chunks antigos de documentos removidos não são apagados
    # automaticamente e devem ser tratados por uma rotina de rebuild controlada.
    store.add_documents(documents=list(chunks), ids=ids)
    return store
