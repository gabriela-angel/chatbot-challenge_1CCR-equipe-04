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