from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Divide documentos em chunks preservando os metadados de origem."""
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap não pode ser negativo.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size.")
    if not documents:
        raise ValueError("A lista de documentos está vazia; não há conteúdo para dividir.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    if not chunks:
        raise ValueError("O processo de chunking não gerou nenhum fragmento.")

    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = index
        chunk.metadata.setdefault("source", "fonte_desconhecida")
    return chunks
