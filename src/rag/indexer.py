"""CLI simples para carregar, dividir e indexar a base documental."""
import argparse

from src.rag.chunking import split_documents
from src.rag.loader import DEFAULT_KNOWLEDGE_BASE, load_documents
from src.rag.vector_store import DEFAULT_PERSIST_DIRECTORY, index_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa PDFs no ChromaDB persistente.")
    parser.add_argument("--knowledge-base", default=str(DEFAULT_KNOWLEDGE_BASE))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--collection", default="goodwe_knowledge_base")
    parser.add_argument("--persist-dir", default=None)
    args = parser.parse_args()

    documents = load_documents(args.knowledge_base)
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store = index_documents(
        chunks,
        persist_directory=args.persist_dir or DEFAULT_PERSIST_DIRECTORY,
        collection_name=args.collection,
    )
    print(f"Documentos/páginas carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Coleção Chroma: {args.collection}")
    print(f"Total de registros na coleção: {store._collection.count()}")


if __name__ == "__main__":
    main()
