import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_CLOUD_URL = "https://ollama.com"


def build_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Cria embeddings via Ollama (Cloud por padrão).

    Para usar um Ollama local (sem chave), defina no .env:
    OLLAMA_EMBED_BASE_URL=http://localhost:11434
    """
    base_url = (
        os.getenv("OLLAMA_EMBED_BASE_URL", "").strip() or OLLAMA_CLOUD_URL
    ).rstrip("/")

    kwargs = {}
    if base_url == OLLAMA_CLOUD_URL:
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY não encontrada. "
                "Configure a variável no arquivo .env local."
            )
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {api_key}"}
        }

    return OllamaEmbeddings(model=model, base_url=base_url, **kwargs)