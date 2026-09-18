import re


DOMAIN_KEYWORDS = {
    "goodwe",
    "goodwe brasil",
    "chargegrid",
    "chargegrid intelligence",

    "eletroposto",
    "eletropostos",
    "carregador",
    "carregadores",
    "carregamento",
    "carregar",
    "recarga",
    "recarregar",
    "sessão",
    "sessões",

    "energia",
    "consumo",
    "consumir",
    "potência",
    "potencia",
    "bateria",

    "status",
    "disponível",
    "disponivel",
    "indisponível",
    "indisponivel",
    "falha",
    "erro",
    "manutenção",
    "manutencao",

    "faturamento",
    "fatura",
    "cobrança",
    "cobranca",
    "pagamento",
    "relatório",
    "relatorio",
    "estação",
    "estacao",
}


GREETING_PATTERNS = [
    r"^\s*olá[!?.]*\s*$",
    r"^\s*ola[!?.]*\s*$",
    r"^\s*oi[!?.]*\s*$",
    r"^\s*oie[!?.]*\s*$",
    r"^\s*bom dia[!?.]*\s*$",
    r"^\s*boa tarde[!?.]*\s*$",
    r"^\s*boa noite[!?.]*\s*$",
]


FOLLOW_UP_PATTERNS = [
    r"\be como\b",
    r"\be isso\b",
    r"\be ele\b",
    r"\be ela\b",
    r"\be nesse caso\b",
    r"\be depois\b",
    r"\be então\b",
    r"\be entao\b",
    r"\bisso\b",
    r"\bnesse caso\b",
    r"\bmais detalhes\b",
    r"\bexplique mais\b",
    r"\bcomo funciona\b",
    r"\be qual\b",
    r"\be quais\b",
]


OUT_OF_SCOPE_PATTERNS = [
    r"\bcapital da frança\b",
    r"\bcapital da franca\b",
    r"\breceita de bolo\b",
    r"\bpartida de futebol\b",
    r"\bresultado do jogo\b",
    r"\bprogramar um jogo\b",
]


def is_greeting(text: str) -> bool:
    text = text.strip().lower()

    return any(
        re.search(pattern, text)
        for pattern in GREETING_PATTERNS
    )


def has_domain_keyword(text: str) -> bool:
    text = text.lower()

    return any(
        keyword in text
        for keyword in DOMAIN_KEYWORDS
    )


def is_follow_up(text: str) -> bool:
    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in FOLLOW_UP_PATTERNS
    )


def is_explicitly_out_of_scope(text: str) -> bool:
    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in OUT_OF_SCOPE_PATTERNS
    )


def validate_scope(
    text: str,
    has_history: bool = False
) -> tuple[bool, str]:

    text = text.strip()

    if not text:
        return (
            False,
            "Por favor, informe uma dúvida sobre a "
            "GoodWe ChargeGrid Intelligence."
        )

    # Permite saudações.
    if is_greeting(text):
        return True, ""

    # Bloqueia assuntos claramente fora do escopo.
    if is_explicitly_out_of_scope(text):
        return (
            False,
            "Esse assunto está fora do meu escopo. "
            "Posso ajudar com GoodWe ChargeGrid, "
            "recarga, consumo, potência, status de carregadores, "
            "faturamento e suporte à plataforma."
        )

    # Assunto diretamente relacionado ao domínio.
    if has_domain_keyword(text):
        return True, ""

    # Permite continuidade quando existe histórico.
    if has_history and is_follow_up(text):
        return True, ""

    # Perguntas curtas após uma conversa podem ser continuação.
    if has_history and len(text.split()) <= 5:
        return True, ""

    return (
        False,
        "Esse assunto está fora do meu escopo. "
        "Posso ajudar com GoodWe ChargeGrid, recarga, "
        "consumo, potência, status de carregadores, "
        "faturamento e suporte à plataforma."
    )


def validar_escopo(
    texto: str,
    tem_historico: bool = False,
    has_history: bool | None = None
):
    """
    Mantém compatibilidade com chamadas antigas.
    """

    if has_history is not None:
        tem_historico = has_history

    return validate_scope(
        texto,
        has_history=tem_historico
    )