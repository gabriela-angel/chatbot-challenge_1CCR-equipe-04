import os
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from src.schemas.consulta_recarga import ConsultaRecarga


BASE_DIR = Path(__file__).resolve().parents[2]
PROMPTS_DIR = BASE_DIR / "prompts"

# Registro versionado. Cada arquivo contém o placeholder {context}.
PROMPT_VERSIONS = {
    "v1": "rag_prompt_v1.md",
    "v2": "rag_prompt_v2.md",
    "v3": "rag_prompt_v3.md",
}

# Versão padrão: sobrescreva com RAG_PROMPT_VERSION no .env após medir os scores.
DEFAULT_PROMPT_VERSION = os.getenv("RAG_PROMPT_VERSION", "v3")

# Frase que os prompts v2+ instruem o modelo a usar quando falta evidência.
NO_INFO_PHRASE = "Não encontrei informação suficiente na base de conhecimento"

# Compatibilidade com a versão anterior do módulo.
PROMPT_PATH = PROMPTS_DIR / PROMPT_VERSIONS["v1"]


def list_prompt_versions() -> list[str]:
    return list(PROMPT_VERSIONS)


def load_rag_prompt(version: str | None = None) -> str:
    """Carrega o prompt RAG versionado do projeto."""

    version = version or DEFAULT_PROMPT_VERSION

    if version not in PROMPT_VERSIONS:
        raise ValueError(
            f"Versão de prompt RAG desconhecida: {version!r}. "
            f"Disponíveis: {', '.join(PROMPT_VERSIONS)}."
        )

    path = PROMPTS_DIR / PROMPT_VERSIONS[version]

    if not path.exists():
        raise FileNotFoundError(f"Prompt RAG não encontrado: {path}")

    return path.read_text(encoding="utf-8")


def build_rag_chain(llm, prompt_version: str | None = None):
    """Cria a chain RAG com histórico e saída Pydantic."""

    parser = PydanticOutputParser(
        pydantic_object=ConsultaRecarga
    )

    system_template = load_rag_prompt(prompt_version)

    system_template += (
        "\n\nA resposta deve obedecer ao seguinte formato estruturado:\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Pergunta do usuário:\n{input}",
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    return prompt | llm | parser