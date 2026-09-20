import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_ollama import ChatOllama

from src.schemas.consulta_recarga import ConsultaRecarga


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]

PROMPT_FILE = (
    BASE_DIR
    / "prompts"
    / "system_prompt.md"
)


def carregar_system_prompt() -> str:

    if not PROMPT_FILE.exists():

        raise FileNotFoundError(
            f"System prompt não encontrado: {PROMPT_FILE}"
        )

    return PROMPT_FILE.read_text(
        encoding="utf-8"
    )


def build_llm(
    model: str = "gpt-oss:120b",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 800,
):

    api_key = os.getenv(
        "OLLAMA_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada "
            "no arquivo .env"
        )

    return ChatOllama(
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_tokens,
        base_url="https://ollama.com",
        client_kwargs={
            "headers": {
                "Authorization": (
                    f"Bearer {api_key}"
                )
            }
        },
    )


def build_prompt():

    parser = PydanticOutputParser(
        pydantic_object=ConsultaRecarga
    )

    system_prompt = carregar_system_prompt()

    system_prompt = (
        system_prompt
        + "\n\n"
        + "<structured_output>\n"
        + "{format_instructions}\n"
        + "</structured_output>"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),

            MessagesPlaceholder(
                variable_name="history"
            ),

            (
                "human",
                "{input}",
            ),
        ]
    )

    prompt = prompt.partial(
        format_instructions=(
            parser.get_format_instructions()
        )
    )

    return prompt, parser


def build_chain(
    llm=None,
    model: str = "gpt-oss:120b",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 800,
):

    if llm is None:

        llm = build_llm(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    prompt, parser = build_prompt()

    return (
        prompt
        | llm
        | parser
    )


def criar_chain(
    model: str = "gpt-oss:120b",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 800,
):

    return build_chain(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )