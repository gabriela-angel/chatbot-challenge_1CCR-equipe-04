from src.ui import run_cli
from src.engine import MissionEngine


if __name__ == "__main__":

    engine = MissionEngine(
        model="gpt-oss:120b",
        temperature=0.3,
        top_p=0.9,
        max_tokens=800,
        max_memory_tokens=2500
    )

    run_cli(engine)