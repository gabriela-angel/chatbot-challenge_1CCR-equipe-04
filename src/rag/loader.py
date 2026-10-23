from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_BASE = BASE_DIR / "data" / "knowledge_base"


def load_documents(knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE) -> List[Document]:
    """Carrega PDFs da base documental e preserva fonte/página nos metadados.

    O diretório é percorrido recursivamente. PDFs sem texto são ignorados, pois
    exigiriam uma etapa de OCR que não faz parte desta primeira implementação.
    """
    base = Path(knowledge_base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(
            f"Base de conhecimento não encontrada: {base}. Crie o diretório e adicione PDFs."
        )
    if not base.is_dir():
        raise NotADirectoryError(f"A base de conhecimento não é um diretório: {base}")

    pdf_files = sorted(base.rglob("*.pdf")) + sorted(base.rglob("*.PDF"))
    # Evita carregar duas vezes o mesmo caminho em sistemas case-insensitive.
    pdf_files = list(dict.fromkeys(pdf_files))
    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF foi encontrado em {base}. Adicione documentos antes de indexar."
        )

    documents: List[Document] = []
    for pdf_path in pdf_files:
        try:
            loaded = PyMuPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            raise RuntimeError(f"Falha ao carregar o PDF '{pdf_path}': {exc}") from exc

        relative_source = str(pdf_path.relative_to(base))
        for document in loaded:
            text = (document.page_content or "").strip()
            if not text:
                continue
            document.metadata = dict(document.metadata or {})
            document.metadata["source"] = relative_source
            # PyMuPDFLoader normalmente já fornece page; normalizamos para inteiro quando possível.
            if "page" in document.metadata:
                try:
                    document.metadata["page"] = int(document.metadata["page"])
                except (TypeError, ValueError):
                    pass
            documents.append(document)

    if not documents:
        raise ValueError(
            "Os PDFs encontrados não contêm texto extraível. Verifique os arquivos; PDFs escaneados precisam de OCR."
        )
    return documents
