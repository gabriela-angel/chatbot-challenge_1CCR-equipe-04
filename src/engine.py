import os
from ollama import Client
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")
    }
)


def load_system_prompt():
    path = Path(__file__).parent.parent / "prompts" / "system_prompt.md"

    if path.exists():
        return path.read_text(encoding="utf-8")

    return ""


class MissionEngine:

    MAX_HISTORICO = 10

    def __init__(self):
        self.system_prompt = load_system_prompt()
        self.historico = []

    def is_ready(self) -> bool:
        return True

    def analyze(self, pergunta_usuario: str) -> str:

        self.historico.append({
            "role": "user",
            "content": pergunta_usuario
        })

        if len(self.historico) > self.MAX_HISTORICO * 2:
            self.historico = self.historico[-self.MAX_HISTORICO * 2:]

        mensagens = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        mensagens.extend(self.historico)

        try:
            resposta = client.chat(
                model="gpt-oss:120b",
                messages=mensagens,
                options={
                    "num_predict": 800,
                    "temperature": 0.3
                },
                stream=False
            )["message"]["content"].strip()

            self.historico.append({
                "role": "assistant",
                "content": resposta
            })

            return resposta

        except Exception as e:
            return f"⚠️ Erro ao consultar IA: {e}"