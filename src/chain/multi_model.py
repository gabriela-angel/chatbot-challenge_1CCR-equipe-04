from src.chain.builder import criar_chain


def comparar_modelos(pergunta: str):

    modelos = [
        "gpt-oss:120b",
        "qwen3:8b"
    ]

    resultados = {}

    for modelo in modelos:

        try:

            chain = criar_chain(
                model=modelo,
                temperature=0.3,
                top_p=0.9,
                max_tokens=800
            )

            resultado = chain.invoke(
                {
                    "input": pergunta,
                    "history": []
                }
            )

            resultados[modelo] = {
                "sucesso": True,
                "resultado": resultado.model_dump()
            }

        except Exception as e:

            resultados[modelo] = {
                "sucesso": False,
                "erro": str(e)
            }

    return resultados


def comparar_rag(
    pergunta: str,
    modelos: list[str] | None = None,
    versoes_prompt: list[str] | None = None,
    collection_name: str = "goodwe_knowledge_base",
    k: int = 4,
):
    """Bônus Sprint 04: consulta o pipeline RAG com mais de um modelo
    e mais de um prompt versionado.

    Retorna um dicionário indexado por "modelo|versão_do_prompt".
    """

    from src.engine import MissionEngine

    modelos = modelos or ["gpt-oss:120b", "qwen3:8b"]
    versoes_prompt = versoes_prompt or ["v2", "v3"]

    resultados = {}

    for modelo in modelos:

        for versao in versoes_prompt:

            chave = f"{modelo}|{versao}"

            try:

                engine = MissionEngine(
                    model=modelo,
                    temperature=0.0,
                    use_rag=True,
                    rag_k=k,
                    rag_prompt_version=versao,
                    collection_name=collection_name,
                )

                detalhe = engine.analyze_detailed(
                    pergunta,
                    session_id=f"multi_{modelo}_{versao}",
                )

                resultados[chave] = {
                    "sucesso": detalhe["reason"] != "processing_error",
                    "resposta": detalhe["output"],
                    "motivo": detalhe["reason"],
                    "fontes": detalhe["sources"],
                    "retrieval_score": detalhe["retrieval_score"],
                }

            except Exception as e:

                resultados[chave] = {
                    "sucesso": False,
                    "erro": str(e),
                }

    return resultados