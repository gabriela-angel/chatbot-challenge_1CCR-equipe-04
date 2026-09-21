import json
import time
from pathlib import Path

from src.engine import MissionEngine
from src.utils.tokens import contar_tokens_texto


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

    texto = (resposta or "").lower()

    if not keywords:
        return 1.0

    encontrados = sum(
        1
        for keyword in keywords
        if keyword.lower() in texto
    )

    return encontrados / len(keywords)


def executar_caso(
    engine: MissionEngine,
    caso: dict
):

    session_id = f"eval_{caso['id']}"

    inicio = time.perf_counter()

    detalhe = engine.analyze_detailed(
        caso["pergunta"],
        session_id=session_id,
    )

    fim = time.perf_counter()

    resposta = detalhe["output"]

    cobertura = calcular_cobertura_keywords(
        resposta,
        caso.get("palavras_chave", [])
    )

    structured = detalhe.get(
        "structured"
    )

    tokens_pergunta = contar_tokens_texto(
        caso["pergunta"]
    )

    tokens_resposta = contar_tokens_texto(
        resposta
    )

    tokens_memoria = (
        engine.get_memory_token_count(
            session_id
        )
    )

    structured_valid = (
        structured is not None
    )

    passou = (
        cobertura >= 0.5
        and (
            structured_valid
            or detalhe.get("blocked", False)
        )
    )

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "cenario": caso["cenario"],
        "pergunta": caso["pergunta"],
        "resposta": resposta,

        "bloqueado_por_guardrail": (
            detalhe.get("blocked", False)
        ),

        "structured_output_valido": (
            structured_valid
        ),

        "structured_output": structured,

        "tokens_pergunta": tokens_pergunta,

        "tokens_resposta": tokens_resposta,

        "tokens_memoria_apos_turno": (
            tokens_memoria
        ),

        "latencia_segundos": round(
            fim - inicio,
            3
        ),

        "keyword_coverage": round(
            cobertura,
            3
        ),

        "passou": passou,
    }


def executar_memoria(
    engine: MissionEngine,
    caso: dict
):

    session_id = f"eval_memory_{caso['id']}"

    engine.clear_session(
        session_id
    )

    turnos = caso["turnos"]

    resultados_turnos = []

    inicio_total = time.perf_counter()

    for indice, pergunta in enumerate(
        turnos,
        start=1
    ):

        inicio = time.perf_counter()

        detalhe = engine.analyze_detailed(
            pergunta,
            session_id=session_id,
        )

        fim = time.perf_counter()

        resposta = detalhe["output"]

        keywords = (
            caso["palavras_chave_por_turno"][
                indice - 1
            ]
        )

        cobertura = calcular_cobertura_keywords(
            resposta,
            keywords
        )

        resultados_turnos.append(
            {
                "turno": indice,
                "pergunta": pergunta,
                "resposta": resposta,

                "structured_output_valido": (
                    detalhe.get("structured")
                    is not None
                ),

                "keyword_coverage": round(
                    cobertura,
                    3
                ),

                "tokens_memoria": (
                    engine.get_memory_token_count(
                        session_id
                    )
                ),

                "latencia_segundos": round(
                    fim - inicio,
                    3
                ),

                "passou": (
                    cobertura >= 0.5
                ),
            }
        )

    fim_total = time.perf_counter()

    memoria_final = (
        engine.get_memory_token_count(
            session_id
        )
    )

    todos_passaram = all(
        turno["passou"]
        for turno in resultados_turnos
    )

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "cenario": caso["cenario"],
        "total_turnos": len(turnos),

        "turnos": resultados_turnos,

        "memoria_tokens_final": (
            memoria_final
        ),

        "limite_memoria_tokens": (
            engine.max_memory_tokens
        ),

        "memoria_respeitou_limite": (
            memoria_final
            <= engine.max_memory_tokens
        ),

        "passou": todos_passaram,

        "latencia_total_segundos": round(
            fim_total - inicio_total,
            3
        ),
    }


def executar():

    casos = carregar_eval()

    engine = MissionEngine()

    resultados = []

    casos_memoria = []

    for caso in casos:

        if caso["categoria"] == "memory":

            resultado = executar_memoria(
                engine,
                caso
            )

            casos_memoria.append(
                resultado
            )

        else:

            resultado = executar_caso(
                engine,
                caso
            )

            resultados.append(
                resultado
            )

    total = len(resultados)

    aprovados = sum(
        1
        for resultado in resultados
        if resultado["passou"]
    )

    latencias = [
        resultado["latencia_segundos"]
        for resultado in resultados
    ]

    media_latencia = (
        sum(latencias) / len(latencias)
        if latencias
        else 0
    )

    casos_com_structured = [
        resultado
        for resultado in resultados
        if not resultado[
            "bloqueado_por_guardrail"
        ]
    ]

    structured_validos = sum(
        1
        for resultado in casos_com_structured
        if resultado[
            "structured_output_valido"
        ]
    )

    structured_accuracy = (
        structured_validos
        / len(casos_com_structured)
        if casos_com_structured
        else 0
    )

    keyword_accuracy = (
        aprovados / total
        if total
        else 0
    )

    resultado_final = {

        "versao": "sprint3",

        "modelo": engine.model,

        "parametros": {
            "temperature": engine.temperature,
            "top_p": engine.top_p,
            "max_tokens": engine.max_tokens,
            "max_memory_tokens": (
                engine.max_memory_tokens
            ),
        },

        "total_casos": total,

        "casos_aprovados": aprovados,

        "acuracia_keyword": round(
            keyword_accuracy,
            3
        ),

        "structured_output_accuracy": round(
            structured_accuracy,
            3
        ),

        "structured_output_validos": (
            structured_validos
        ),

        "structured_output_casos_avaliados": (
            len(casos_com_structured)
        ),

        "latencia_media_segundos": round(
            media_latencia,
            3
        ),

        "tokens_pergunta_total": sum(
            resultado["tokens_pergunta"]
            for resultado in resultados
        ),

        "tokens_resposta_total": sum(
            resultado["tokens_resposta"]
            for resultado in resultados
        ),

        "tokens_total": (
            sum(
                resultado["tokens_pergunta"]
                for resultado in resultados
            )
            +
            sum(
                resultado["tokens_resposta"]
                for resultado in resultados
            )
        ),

        "memoria": {
            "casos": casos_memoria,

            "casos_aprovados": sum(
                1
                for caso in casos_memoria
                if caso["passou"]
            ),

            "limite_respeitado": all(
                caso["memoria_respeitou_limite"]
                for caso in casos_memoria
            ),
        },

        "resultados": resultados,
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

    print()
    print("=" * 55)
    print("GOODWE — SPRINT 03 — EVAL")
    print("=" * 55)

    print(
        f"Casos: {total}"
    )

    print(
        f"Aprovados: {aprovados}"
    )

    print(
        "Acurácia keyword: "
        f"{keyword_accuracy * 100:.1f}%"
    )

    print(
        "Structured Output Accuracy: "
        f"{structured_accuracy * 100:.1f}%"
    )

    print(
        "Latência média: "
        f"{media_latencia:.3f}s"
    )

    print(
        "Casos de memória: "
        f"{len(casos_memoria)}"
    )

    if casos_memoria:
        memoria_respeitou_limite = all(
            caso["memoria_respeitou_limite"]
            for caso in casos_memoria
        )

        print(
            f"Memória respeitou limite: {memoria_respeitou_limite}"
        )

    print()
    print(
        f"Resultados salvos em: {RESULT_FILE}"
    )
    print()


if __name__ == "__main__":
    executar()