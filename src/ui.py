from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

import pyfiglet

from datetime import datetime

from tests.test_model import rodar_bateria_testes


console = Console()

session = PromptSession(
    style=Style.from_dict(
        {
            "prompts": "#FFFFFF bold"
        }
    )
)


def show_banner():

    banner = pyfiglet.figlet_format(
        "GoodWe chatbot",
        font="slant"
    )

    console.print(
        Text(
            banner,
            style="bold #E30613"
        )
    )

    console.print(
        Panel.fit(
            "◆ GoodWe ChargeGrid Intelligence ◆\n"
            "Assistente virtual para suporte a estações de recarga "
            "e gestão energética.\n"
            "Sprint 03 — LangChain LCEL + Memória + Pydantic.\n"
            "Comandos: 'teste' para validação; 'sair' para encerrar.\n"
            "Modelo: gpt-oss:120b via Ollama Cloud",
            title="",
            border_style="#FFFFFF"
        )
    )


def show_response(text):

    now = datetime.now().strftime("%H:%M")

    console.print(
        Panel(
            text,
            title="GOODWE",
            subtitle=now,
            border_style="#E30613"
        )
    )


def run_cli(engine):

    show_banner()

    while True:

        try:

            user_input = session.prompt(
                "❯ "
            ).strip()

        except (KeyboardInterrupt, EOFError):

            break

        if not user_input:

            console.print(
                "⚠ Digite uma pergunta ou 'sair' para encerrar.",
                style="yellow"
            )

            continue

        comando = user_input.lower()

        if comando == "teste":

            rodar_bateria_testes()

            continue

        if comando == "sair":

            console.print(
                "\n👋 Encerrando o GoodWe ChargeGrid Intelligence...",
                style="red"
            )

            break

        resposta = engine.analyze(
            user_input
        )

        show_response(
            resposta
        )