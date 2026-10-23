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
DEFAULT_BATCH_SIZE = 64


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
    """Abre/cria uma coleção Chroma persistente sem apagar dados existentes.

    A coleção usa distância de cosseno, o que torna o score de relevância
    (1 - distância) interpretável. A métrica só vale para coleções novas.
    """
    path = Path(persist_directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    embedding_function = embeddings or build_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(path),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def reset_collection(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: OllamaEmbeddings | None = None,
) -> None:
    """Apaga apenas a coleção indicada (rebuild controlado)."""
    store = open_vector_store(persist_directory, collection_name, embeddings)
    store.delete_collection()


def index_documents(
    chunks: Sequence[Document],
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: OllamaEmbeddings | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = False,
) -> Chroma:
    """Indexa chunks no ChromaDB persistente, com IDs determinísticos e em lotes."""
    if not chunks:
        raise ValueError("Nenhum chunk recebido para indexação.")
    if batch_size < 1:
        raise ValueError("batch_size deve ser pelo menos 1.")

    store = open_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    # Chunks idênticos (mesmo ID) na mesma carga causariam erro de ID duplicado.
    unique: dict[str, Document] = {}
    for document in chunks:
        unique[_document_id(document)] = document
    ids = list(unique.keys())
    documents = list(unique.values())

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        store.add_documents(documents=documents[start:end], ids=ids[start:end])
        if progress:
            print(f"  indexados {min(end, len(documents))}/{len(documents)}", flush=True)
    return store