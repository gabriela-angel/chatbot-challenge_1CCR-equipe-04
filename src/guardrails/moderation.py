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


PADROES_RISCO_ELETRICO = [
    r"fiação\s+.*exposta",
    r"fiacao\s+.*exposta",
    r"fio\s+.*exposto",
    r"cabo\s+.*danificado",
    r"cabo\s+.*rompido",
    r"choque\s+elétrico",
    r"choque\s+eletrico",
    r"curto[\s-]circuito",
    r"faísca",
    r"faisca",
    r"fumaça",
    r"fumaca",
    r"queimado",
    r"cheiro\s+de\s+queimado",
    r"parte\s+elétrica",
    r"parte\s+eletrica",
]


PADROES_MANUTENCAO_FISICA = [
    r"desmontar",
    r"abrir\s+o\s+carregador",
    r"abrir\s+o\s+eletroposto",
    r"trocar\s+fusível",
    r"trocar\s+fusivel",
    r"ligar\s+fio",
    r"desligar\s+fio",
    r"reparar\s+fiação",
    r"reparar\s+fiacao",
    r"consertar\s+o\s+cabo",
]


def normalizar(texto: str) -> str:
    return texto.lower().strip()


def detectar_prompt_injection(
    texto: str
) -> bool:

    texto_normalizado = normalizar(texto)

    return any(
        re.search(
            padrao,
            texto_normalizado
        )
        for padrao in PADROES_INJECTION
    )


def resposta_prompt_injection() -> str:

    return (
        "Não posso ignorar minhas instruções, revelar "
        "instruções internas ou alterar as regras de segurança "
        "do sistema. Posso ajudar com dúvidas relacionadas à "
        "GoodWe ChargeGrid Intelligence, recarga de veículos "
        "elétricos, consumo, faturamento e suporte à plataforma."
    )


def check_prompt_injection(
    texto: str
):

    if detectar_prompt_injection(texto):

        return (
            False,
            resposta_prompt_injection()
        )

    return True, ""


def detectar_risco_eletrico(
    texto: str
) -> bool:

    texto_normalizado = normalizar(texto)

    return any(
        re.search(
            padrao,
            texto_normalizado
        )
        for padrao in PADROES_RISCO_ELETRICO
    )


def detectar_manutencao_fisica(
    texto: str
) -> bool:

    texto_normalizado = normalizar(texto)

    return any(
        re.search(
            padrao,
            texto_normalizado
        )
        for padrao in PADROES_MANUTENCAO_FISICA
    )


def resposta_risco_eletrico() -> str:

    return (
        "Não utilize o equipamento enquanto houver risco "
        "elétrico ou dano físico aparente. Interrompa o uso "
        "com segurança e entre em contato com o suporte técnico "
        "ou com um profissional habilitado. Não tente realizar "
        "reparos ou intervenções elétricas por conta própria."
    )


def resposta_manutencao_fisica() -> str:

    return (
        "Não posso orientar intervenções físicas ou elétricas "
        "no equipamento. Para manutenção, reparo ou desbloqueio "
        "físico do carregador, entre em contato com o suporte "
        "técnico ou com um profissional habilitado."
    )


def check_safety_risk(
    texto: str
):

    if detectar_risco_eletrico(texto):

        return (
            False,
            resposta_risco_eletrico()
        )

    if detectar_manutencao_fisica(texto):

        return (
            False,
            resposta_manutencao_fisica()
        )

    return True, ""


def validar_entrada(
    texto: str
):

    return check_prompt_injection(texto)