
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from src.schemas.consulta_recarga import ConsultaRecarga


BASE_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt_v1.md"


def load_rag_prompt() -> str:
    """Carrega o prompt RAG versionado do projeto."""

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt RAG não encontrado: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


def build_rag_chain(llm):
    """Cria a chain RAG com histórico e saída Pydantic."""

    parser = PydanticOutputParser(
        pydantic_object=ConsultaRecarga
    )

    system_template = load_rag_prompt()

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
