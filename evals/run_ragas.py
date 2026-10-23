"""Avaliação RAG da Sprint 04.

Métricas principais: faithfulness e answer_relevancy via RAGAS, com fallback
para a rubrica manual documentada em evals/rubrica_manual.md.

Exemplos:
    python evals/run_ragas.py --label baseline_sprint3 --mode baseline
    python evals/run_ragas.py --label it1_prompt_v1 --prompt-version v1
    python evals/run_ragas.py --label it2_prompt_v2 --prompt-version v2
    python evals/run_ragas.py --compare baseline_sprint3 it1_prompt_v1 it2_prompt_v2
"""
import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = Path(__file__).resolve().parent
EVAL_FILE = BASE_DIR / "eval_set_rag.json"
RESULTS_DIR = BASE_DIR / "results"

REFUSAL_REASONS = {"guardrail", "no_relevant_context", "insufficient_evidence"}
MIN_VALID_RATIO = 0.7


# ----------------------------------------------------------------- utilidades

def carregar_eval(caminho: Path = EVAL_FILE) -> list[dict]:
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def limpar_resposta(texto: str) -> str:
    """Remove o bloco de fontes e os marcadores [Fonte: ...] antes de pontuar,
    para que nomes de arquivo não sejam tratados como afirmações factuais."""
    texto = (texto or "").split("**Fontes consultadas:**")[0]
    texto = re.sub(r"\[Fonte:[^\]]*\]", "", texto)
    return texto.strip()


def tem_citacao(texto: str) -> bool:
    return "[Fonte:" in (texto or "") or "Fontes consultadas" in (texto or "")


def cobertura_keywords(resposta: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    texto = (resposta or "").lower()
    return sum(1 for k in keywords if k.lower() in texto) / len(keywords)


def media(valores) -> float | None:
    validos = [
        v for v in valores
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return round(sum(validos) / len(validos), 4) if validos else None


def _valido(valor) -> bool:
    return valor is not None and not (isinstance(valor, float) and math.isnan(valor))


# ------------------------------------------------------------ execução do RAG

def rodar_casos(engine, casos: list[dict]) -> list[dict]:
    registros = []
    for caso in casos:
        inicio = time.perf_counter()
        detalhe = engine.analyze_detailed(
            caso["pergunta"], session_id=f"eval_rag_{caso['id']}"
        )
        latencia = time.perf_counter() - inicio

        resposta = detalhe["output"]
        recusou = bool(detalhe.get("blocked")) or detalhe.get("reason") in REFUSAL_REASONS
        esperada = caso.get("fonte_esperada")
        fontes = detalhe.get("retrieved_sources") or []

        registros.append(
            {
                "id": caso["id"],
                "categoria": caso["categoria"],
                "pergunta": caso["pergunta"],
                "resposta": resposta,
                "reason": detalhe.get("reason"),
                "recusou": recusou,
                "citacao_presente": tem_citacao(resposta),
                "retrieval_score": detalhe.get("retrieval_score"),
                "fontes_recuperadas": fontes,
                "fonte_esperada_recuperada": (
                    any(esperada in str(f.get("source", "")) for f in fontes)
                    if esperada else None
                ),
                "keyword_coverage": round(
                    cobertura_keywords(resposta, caso.get("palavras_chave", [])), 3
                ),
                "latencia_segundos": round(latencia, 3),
                "_contexts": detalhe.get("contexts") or [],
                "faithfulness": None,
                "answer_relevancy": None,
            }
        )
    return registros


def selecionar_para_metricas(registros: list[dict], modo: str) -> list[int]:
    """Índices de casos answerable efetivamente respondidos."""
    indices = []
    for i, r in enumerate(registros):
        if r["categoria"] != "answerable" or r["recusou"]:
            continue
        if modo == "rag" and not r["_contexts"]:
            continue
        if not limpar_resposta(r["resposta"]):
            continue
        indices.append(i)
    return indices


# ------------------------------------------------------------------------ RAGAS

def avaliar_ragas(registros, indices, judge_llm, embeddings, modo: str):
    """Retorna {indice: {'faithfulness': x, 'answer_relevancy': y}}."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    amostras = []
    for i in indices:
        r = registros[i]
        amostras.append(
            {
                "user_input": r["pergunta"],
                "response": limpar_resposta(r["resposta"]),
                "retrieved_contexts": r["_contexts"] or ["(sem contexto)"],
            }
        )

    metricas = [ResponseRelevancy()]
    if modo == "rag":
        metricas.insert(0, Faithfulness())

    resultado = evaluate(
        EvaluationDataset.from_list(amostras),
        metrics=metricas,
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(timeout=240, max_workers=2, max_retries=2),
        raise_exceptions=False,
        show_progress=True,
    )
    tabela = resultado.to_pandas()

    saida = {}
    for posicao, i in enumerate(indices):
        linha = tabela.iloc[posicao]
        saida[i] = {
            "faithfulness": (
                float(linha["faithfulness"]) if "faithfulness" in tabela.columns else None
            ),
            "answer_relevancy": float(linha["answer_relevancy"]),
        }
    return saida


# ---------------------------------------------------------------- rubrica manual

PROMPT_JUIZ = """Você é um avaliador rigoroso de sistemas RAG. Responda SOMENTE com um JSON válido, sem texto extra.

PERGUNTA:
__PERGUNTA__

CONTEXTO RECUPERADO:
__CONTEXTO__

RESPOSTA AVALIADA:
__RESPOSTA__

Tarefas:
1. Conte as afirmações factuais da RESPOSTA (ignore saudações e citações): "afirmacoes_total".
2. Conte quantas delas são diretamente sustentadas pelo CONTEXTO: "afirmacoes_suportadas".
3. "relevancia": nota 0, 0.25, 0.5, 0.75 ou 1 de quão diretamente a RESPOSTA responde à PERGUNTA
   (1 = completa e sem desvios; 0.5 = parcial; 0 = não responde).

Formato exato:
{"afirmacoes_total": 0, "afirmacoes_suportadas": 0, "relevancia": 0.0}"""


def _extrair_json(texto: str) -> dict | None:
    encontrado = re.search(r"\{.*\}", texto or "", flags=re.DOTALL)
    if not encontrado:
        return None
    try:
        return json.loads(encontrado.group(0))
    except json.JSONDecodeError:
        return None


def julgar_manual(judge_llm, pergunta: str, contexto: str, resposta: str) -> dict:
    prompt = (
        PROMPT_JUIZ.replace("__PERGUNTA__", pergunta)
        .replace("__CONTEXTO__", contexto or "(sem contexto)")
        .replace("__RESPOSTA__", resposta)
    )
    for _ in range(2):
        bruto = judge_llm.invoke(prompt)
        dados = _extrair_json(getattr(bruto, "content", str(bruto)))
        if not dados:
            continue
        try:
            total = int(dados["afirmacoes_total"])
            suportadas = min(int(dados["afirmacoes_suportadas"]), total)
            relevancia = min(max(float(dados["relevancia"]), 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            continue
        return {
            "faithfulness": round(suportadas / total, 4) if total > 0 else None,
            "answer_relevancy": relevancia,
        }
    return {"faithfulness": None, "answer_relevancy": None}


def avaliar_manual(registros, indices, judge_llm, modo: str):
    saida = {}
    for i in indices:
        r = registros[i]
        nota = julgar_manual(
            judge_llm,
            r["pergunta"],
            "\n\n---\n\n".join(r["_contexts"]),
            limpar_resposta(r["resposta"]),
        )
        if modo != "rag":
            nota["faithfulness"] = None
        saida[i] = nota
    return saida


# --------------------------------------------------------------------- agregação

def pontuar(registros, indices, judge_llm, embeddings, modo, backend):
    """Aplica RAGAS (com fallback) e devolve (notas, backend_usado, erro)."""
    erro = None
    if backend in ("ragas", "auto") and indices:
        try:
            notas = avaliar_ragas(registros, indices, judge_llm, embeddings, modo)
            campos = ["answer_relevancy"] + (["faithfulness"] if modo == "rag" else [])
            validade = min(
                sum(_valido(n[c]) for n in notas.values()) / len(notas) for c in campos
            )
            if validade >= MIN_VALID_RATIO:
                return notas, "ragas", None
            erro = f"RAGAS retornou apenas {validade:.0%} de valores válidos."
        except Exception as exc:  # noqa: BLE001 - qualquer falha aciona o fallback
            erro = f"{type(exc).__name__}: {exc}"
        if backend == "ragas":
            raise RuntimeError(f"RAGAS falhou e o fallback foi desativado: {erro}")

    notas = avaliar_manual(registros, indices, judge_llm, modo)
    return notas, "manual", erro


def agregar(registros, indices, notas) -> dict:
    for i, nota in notas.items():
        registros[i]["faithfulness"] = nota["faithfulness"]
        registros[i]["answer_relevancy"] = nota["answer_relevancy"]

    answerable = [r for r in registros if r["categoria"] == "answerable"]
    esperadas_recusa = [r for r in registros if r["categoria"] in ("unanswerable", "guardrail")]

    com_fonte = [r for r in answerable if r["fonte_esperada_recuperada"] is not None]
    respondidos = [r for r in answerable if not r["recusou"]]

    return {
        "faithfulness": media(r["faithfulness"] for r in registros),
        "answer_relevancy": media(r["answer_relevancy"] for r in registros),
        "keyword_coverage": media(r["keyword_coverage"] for r in answerable),
        "citation_rate": media(1.0 if r["citacao_presente"] else 0.0 for r in respondidos),
        "source_hit_rate": media(1.0 if r["fonte_esperada_recuperada"] else 0.0 for r in com_fonte),
        "mean_retrieval_score": media(r["retrieval_score"] for r in answerable),
        "false_refusal_rate": media(1.0 if r["recusou"] else 0.0 for r in answerable),
        "correct_refusal_rate": media(1.0 if r["recusou"] else 0.0 for r in esperadas_recusa),
        "latencia_media_segundos": media(r["latencia_segundos"] for r in registros),
        "n_answerable": len(answerable),
        "n_scored": len(indices),
    }


def executar_avaliacao(engine, casos, judge_llm, embeddings, modo, backend) -> dict:
    registros = rodar_casos(engine, casos)
    indices = selecionar_para_metricas(registros, modo)
    notas, backend_usado, erro = pontuar(
        registros, indices, judge_llm, embeddings, modo, backend
    )
    agregado = agregar(registros, indices, notas)
    for r in registros:
        r.pop("_contexts", None)
    return {"metric_backend": backend_usado, "ragas_error": erro, "agregado": agregado, "casos": registros}


# -------------------------------------------------------------------------- CLI

def _fmt(valor) -> str:
    return "n/d" if valor is None else (f"{valor:.3f}" if isinstance(valor, float) else str(valor))


def comparar(labels: list[str]) -> None:
    resultados = []
    for label in labels:
        with open(RESULTS_DIR / f"{label}.json", "r", encoding="utf-8") as f:
            resultados.append(json.load(f))

    linhas = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer relevancy"),
        ("source_hit_rate", "Fonte esperada recuperada"),
        ("mean_retrieval_score", "Score médio de recuperação"),
        ("citation_rate", "Respostas com citação"),
        ("false_refusal_rate", "Recusas indevidas"),
        ("correct_refusal_rate", "Recusas corretas"),
        ("keyword_coverage", "Cobertura de keywords"),
    ]
    print("| Métrica | " + " | ".join(r["label"] for r in resultados) + " |")
    print("|---|" + "---|" * len(resultados))
    print("| Backend | " + " | ".join(r["metric_backend"] for r in resultados) + " |")
    for chave, nome in linhas:
        print(f"| {nome} | " + " | ".join(_fmt(r["agregado"][chave]) for r in resultados) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação RAG (RAGAS + fallback manual).")
    parser.add_argument("--label", default=None, help="Nome da iteração (gera evals/results/<label>.json).")
    parser.add_argument("--mode", choices=["rag", "baseline"], default="rag",
                        help="baseline = chain da Sprint 3 sem RAG (use_rag=False).")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--collection", default="goodwe_knowledge_base")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--judge-model", default="gpt-oss:120b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--metrics", choices=["auto", "ragas", "manual"], default="auto")
    parser.add_argument("--eval-file", default=str(EVAL_FILE))
    parser.add_argument("--compare", nargs="+", metavar="LABEL")
    args = parser.parse_args()

    if args.compare:
        comparar(args.compare)
        return
    if not args.label:
        parser.error("--label é obrigatório (ou use --compare).")

    from src.chain.builder import build_llm
    from src.engine import MissionEngine
    from src.rag.embeddings import build_embeddings
    from src.rag.prompt_rag import DEFAULT_PROMPT_VERSION

    prompt_version = args.prompt_version or DEFAULT_PROMPT_VERSION
    engine = MissionEngine(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        use_rag=(args.mode == "rag"),
        rag_k=args.k,
        rag_min_score=args.min_score,
        rag_prompt_version=prompt_version,
        collection_name=args.collection,
    )
    judge_llm = build_llm(model=args.judge_model, temperature=0.0, top_p=1.0, max_tokens=2000)

    embeddings = None
    if args.metrics != "manual":
        try:
            embeddings = build_embeddings()
        except Exception as exc:  # noqa: BLE001
            print(f"[aviso] embeddings indisponíveis para o RAGAS: {exc}")

    backend = "manual" if embeddings is None and args.metrics == "auto" else args.metrics

    casos = carregar_eval(Path(args.eval_file))
    saida = executar_avaliacao(engine, casos, judge_llm, embeddings, args.mode, backend)

    final = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modo": args.mode,
        "config": {
            "model": args.model,
            "judge_model": args.judge_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "k": args.k,
            "min_score": args.min_score,
            "prompt_version": prompt_version if args.mode == "rag" else "system_prompt.md (Sprint 3)",
            "collection": args.collection if args.mode == "rag" else None,
        },
        **saida,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = RESULTS_DIR / f"{args.label}.json"
    with open(destino, "w", encoding="utf-8") as arquivo:
        json.dump(final, arquivo, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"AVALIAÇÃO RAG — {args.label} (backend: {final['metric_backend']})")
    print("=" * 60)
    for chave, valor in final["agregado"].items():
        print(f"{chave}: {_fmt(valor)}")
    if final["ragas_error"]:
        print(f"[aviso] fallback acionado: {final['ragas_error']}")
    print(f"\nSalvo em: {destino}")


if __name__ == "__main__":
    main()