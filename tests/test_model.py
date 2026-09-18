from src.engine import MissionEngine


def rodar_bateria_testes():

    engine = MissionEngine()

    casos_de_teste = [

        {
            "id": 1,
            "cenario": "Gerenciamento de Potência",
            "pergunta": (
                "Estou com dois carros conectados no mesmo "
                "eletroposto corporativo e notei que a velocidade "
                "de carregamento diminuiu para ambos. Isso é um "
                "defeito na estação?"
            )
        },

        {
            "id": 2,
            "cenario": "Cobrança e Faturamento",
            "pergunta": (
                "Como operador, preciso extrair um relatório "
                "consolidado com o consumo energético e os valores "
                "faturados da estação central na última semana. "
                "Onde localizo isso?"
            )
        },

        {
            "id": 3,
            "cenario": "Status",
            "pergunta": (
                "Um cliente ligou dizendo que o carregador da "
                "filial Sul aparece como 'Indisponível' no mapa. "
                "O que isso significa?"
            )
        },

        {
            "id": 4,
            "cenario": "Falha Física",
            "pergunta": (
                "O carregador número 3 está com a trava da tomada "
                "emperrada e a luz vermelha está piscando. "
                "Como faço para destravar pelo sistema?"
            )
        },

        {
            "id": 5,
            "cenario": "Risco Elétrico",
            "pergunta": (
                "Notei que o cabo do eletroposto está danificado "
                "e com a fiação exposta. Posso conectar meu carro?"
            )
        }
    ]

    print("\n")
    print("=" * 80)
    print("GOODWE CHARGEGRID INTELLIGENCE")
    print("SPRINT 03 — TESTES LCEL")
    print("=" * 80)

    for caso in casos_de_teste:

        print("\n")
        print("-" * 80)

        print(
            f"CASO {caso['id']} - "
            f"{caso['cenario']}"
        )

        print("-" * 80)

        print(
            f"\nPergunta:\n{caso['pergunta']}"
        )

        resposta = engine.analyze(
            caso["pergunta"]
        )

        print(
            f"\nResposta:\n{resposta}"
        )

        print(
            "\n[ OK ] Caso executado."
        )

    print("\n")


if __name__ == "__main__":
    rodar_bateria_testes()