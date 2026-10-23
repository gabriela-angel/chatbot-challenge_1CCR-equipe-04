import logging

from src.ui import run_cli
from src.engine import MissionEngine


# Erros internos (ex.: falha do modelo ou do ChromaDB) vão para arquivo,
# sem poluir a interface do terminal.
logging.basicConfig(
    filename="chatbot.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


if __name__ == "__main__":

    engine = MissionEngine(
        model="gpt-oss:120b",
        temperature=0.0,
        top_p=0.9,
        max_tokens=800,
        max_memory_tokens=2500,
        use_rag=True,
        rag_k=4,
    )

    run_cli(engine)