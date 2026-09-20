from typing import Iterable

import tiktoken
from langchain_core.messages import BaseMessage


def get_encoding():

    try:
        return tiktoken.encoding_for_model(
            "gpt-4o-mini"
        )

    except KeyError:
        return tiktoken.get_encoding(
            "cl100k_base"
        )


def contar_tokens_texto(
    texto: str
) -> int:

    encoding = get_encoding()

    return len(
        encoding.encode(
            texto or ""
        )
    )


def contar_tokens_mensagens(
    mensagens: Iterable[BaseMessage]
) -> int:

    total = 0

    for mensagem in mensagens:

        conteudo = mensagem.content

        if isinstance(conteudo, str):

            total += contar_tokens_texto(
                conteudo
            )

        else:

            total += contar_tokens_texto(
                str(conteudo)
            )

    return total