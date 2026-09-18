import json
import time
from pathlib import Path

from src.engine import MissionEngine


BASE_DIR = Path(__file__).resolve().parent

EVAL_FILE = BASE_DIR / "eval_set.json"
RESULT_FILE = BASE_DIR / "sprint3_results.json"


def carregar_eval():

    with open(
        EVAL_FILE,
        "r",
        encoding="utf-8"
    ) as arquivo:

        return json.load(arquivo)


def calcular_cobertura_keywords(
    resposta: str,
    keywords: list[str]
) -> float:

    texto = resposta.lower()

    if not keywords:
        return 1.0

    encontrados = sum(
        1
        for keyword in keywords
        if keyword.lower() in texto
    )

    return encontrados / len(keywords)


def executar():

    casos = carregar_eval()

    engine = MissionEngine()

    resultados = []

    for caso in casos:

        inicio = time.perf_counter()

        resposta = engine.analyze(
            caso["pergunta"]
        )

        fim = time.perf_counter()

        latencia = fim - inicio

        cobertura = calcular_cobertura_keywords(
            resposta,
            caso["palavras_chave"]
        )

        resultados.append(
            {
                "id": caso["id"],
                "categoria": caso["categoria"],
                "cenario": caso["cenario"],
                "pergunta": caso["pergunta"],
                "resposta": resposta,
                "latencia_segundos": round(
                    latencia,
                    3
                ),
                "keyword_coverage": round(
                    cobertura,
                    3
                ),
                "passou": cobertura >= 0.5
            }
        )

    total = len(resultados)

    aprovados = sum(
        1
        for resultado in resultados
        if resultado["passou"]
    )

    resultado_final = {

        "modelo": engine.model,

        "parametros": {
            "temperature": engine.temperature,
            "top_p": engine.top_p,
            "max_tokens": engine.max_tokens
        },

        "total_casos": total,

        "casos_aprovados": aprovados,

        "acuracia_keyword": round(
            aprovados / total,
            3
        ) if total else 0,

        "resultados": resultados
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultado_final,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\n=============================================="
    )

    print(
        "GOODWE — SPRINT 03 — EVAL"
    )

    print(
        "=============================================="
    )

    print(
        f"Casos: {total}"
    )

    print(
        f"Aprovados: {aprovados}"
    )

    print(
        f"Acurácia: "
        f"{resultado_final['acuracia_keyword'] * 100:.1f}%"
    )

    print(
        f"\nResultados salvos em: {RESULT_FILE}"
    )


if __name__ == "__main__":
    executar()