from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_BASE = BASE_DIR / "data" / "knowledge_base"

TEXT_EXTENSIONS = {".md", ".txt"}


def _list_files(base: Path) -> List[Path]:
    files = [
        path
        for path in base.rglob("*")
        if path.is_file()
        and (path.suffix.lower() == ".pdf" or path.suffix.lower() in TEXT_EXTENSIONS)
    ]
    return sorted(set(files))


def load_documents(knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE) -> List[Document]:
    """Carrega PDFs (PyMuPDFLoader) e, opcionalmente, FAQs/tabelas em .md/.txt.

    O diretório é percorrido recursivamente. PDFs sem texto são ignorados, pois
    exigiriam OCR, que não faz parte desta implementação.
    """
    base = Path(knowledge_base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(
            f"Base de conhecimento não encontrada: {base}. Crie o diretório e adicione documentos."
        )
    if not base.is_dir():
        raise NotADirectoryError(f"A base de conhecimento não é um diretório: {base}")

    files = _list_files(base)
    if not files:
        raise FileNotFoundError(
            f"Nenhum documento (.pdf/.md/.txt) foi encontrado em {base}."
        )

    documents: List[Document] = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                loaded = PyMuPDFLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()
        except Exception as exc:
            raise RuntimeError(f"Falha ao carregar '{path}': {exc}") from exc

        relative_source = path.relative_to(base).as_posix()
        for document in loaded:
            text = (document.page_content or "").strip()
            if not text:
                continue
            document.metadata = dict(document.metadata or {})
            document.metadata["source"] = relative_source
            if "page" in document.metadata:
                try:
                    document.metadata["page"] = int(document.metadata["page"])
                except (TypeError, ValueError):
                    pass
            documents.append(document)

    if not documents:
        raise ValueError(
            "Os documentos encontrados não contêm texto extraível. PDFs escaneados precisam de OCR."
        )
    return documents