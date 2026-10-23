import re
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_DOT_LEADERS = re.compile(r"[.\u2026·]{4,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_SPACES = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Remove ruído típico de PDFs: pontilhados de sumário, caracteres de
    controle e espaços repetidos."""
    text = _CONTROL_CHARS.sub(" ", text or "")
    text = _DOT_LEADERS.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    clean: bool = True,
    min_chunk_chars: int = 40,
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

    prepared: List[Document] = []
    for document in documents:
        content = clean_text(document.page_content) if clean else document.page_content
        if not content.strip():
            continue
        prepared.append(
            Document(page_content=content, metadata=dict(document.metadata or {}))
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(prepared)
        if len(chunk.page_content.strip()) >= min_chunk_chars
    ]
    if not chunks:
        raise ValueError("O processo de chunking não gerou nenhum fragmento.")

    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = index
        chunk.metadata.setdefault("source", "fonte_desconhecida")
    return chunks