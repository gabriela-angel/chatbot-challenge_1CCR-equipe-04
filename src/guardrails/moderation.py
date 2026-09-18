import re


PADROES_INJECTION = [
    r"ignore\s+(all|previous|prior)\s+instructions",
    r"ignore\s+as\s+instruções",
    r"ignore\s+as\s+instrucoes",
    r"ignore\s+o\s+system\s+prompt",
    r"ignore\s+seu\s+prompt",
    r"ignore\s+as\s+regras",
    r"esqueça\s+as\s+instruções",
    r"esqueca\s+as\s+instrucoes",
    r"mostre\s+seu\s+system\s+prompt",
    r"mostre\s+as\s+instruções\s+internas",
    r"mostre\s+as\s+instrucoes\s+internas",
    r"revele\s+seu\s+prompt",
    r"reveal\s+your\s+system\s+prompt",
    r"jailbreak",
    r"bypass\s+.*guardrail",
    r"desative\s+.*segurança",
    r"desative\s+.*guardrail",
    r"finja\s+que\s+não\s+há\s+restrições",
    r"finja\s+que\s+nao\s+ha\s+restricoes",
]


def detectar_prompt_injection(texto: str) -> bool:
    texto_normalizado = texto.lower()

    return any(
        re.search(padrao, texto_normalizado)
        for padrao in PADROES_INJECTION
    )


def resposta_prompt_injection() -> str:
    return (
        "Não posso ignorar minhas instruções, revelar instruções internas "
        "ou alterar as regras de segurança do sistema. "
        "Posso ajudar com dúvidas relacionadas à GoodWe ChargeGrid "
        "Intelligence, recarga de veículos elétricos, consumo, "
        "faturamento e suporte à plataforma."
    )


def check_prompt_injection(texto: str):
    """
    Compatibilidade com o engine.py.

    Retorna:
        (True, "") quando a entrada pode continuar.
        (False, mensagem) quando deve ser recusada.
    """

    if detectar_prompt_injection(texto):
        return False, resposta_prompt_injection()

    return True, ""


def validar_entrada(texto: str):
    """
    Mantém compatibilidade com testes ou arquivos antigos.
    """

    return check_prompt_injection(texto)