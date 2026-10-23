"""CLI para carregar, dividir e indexar a base documental no ChromaDB."""
import argparse

from src.rag.chunking import split_documents
from src.rag.loader import DEFAULT_KNOWLEDGE_BASE, load_documents
from src.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    index_documents,
    reset_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa a base de conhecimento no ChromaDB persistente.")
    parser.add_argument("--knowledge-base", default=str(DEFAULT_KNOWLEDGE_BASE))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Apaga a coleção indicada antes de indexar (use ao mudar chunking).",
    )
    args = parser.parse_args()

    persist_dir = args.persist_dir or DEFAULT_PERSIST_DIRECTORY

    documents = load_documents(args.knowledge_base)
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.rebuild:
        reset_collection(persist_dir, args.collection)

    store = index_documents(
        chunks,
        persist_directory=persist_dir,
        collection_name=args.collection,
        progress=True,
    )
    print(f"Documentos/páginas carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)} (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Coleção Chroma: {args.collection}")
    print(f"Total de registros na coleção: {store._collection.count()}")


if __name__ == "__main__":
    main()