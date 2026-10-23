import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_CLOUD_URL = "https://ollama.com"


def build_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Cria embeddings via Ollama Cloud usando a chave do ambiente."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. Configure a variável no arquivo .env local."
        )
    return OllamaEmbeddings(
        model=model,
        base_url=OLLAMA_CLOUD_URL,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )
