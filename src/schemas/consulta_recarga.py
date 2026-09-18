from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator


class ConsultaRecarga(BaseModel):
    """
    Saída estruturada para consultas relacionadas
    ao ChargeGrid Intelligence.
    """

    resposta: str = Field(
        description="Resposta textual final apresentada ao usuário."
    )

    categoria: Literal[
        "recarga",
        "status",
        "potencia",
        "consumo",
        "faturamento",
        "falha",
        "seguranca",
        "suporte",
        "fora_do_escopo"
    ] = Field(
        description="Categoria principal da consulta."
    )

    estado_carregador: Optional[str] = Field(
        default=None,
        description="Estado do carregador quando informado ou relevante."
    )

    potencia_kw: Optional[float] = Field(
        default=None,
        description="Potência em kW quando explicitamente informada ou disponível."
    )

    consumo_kwh: Optional[float] = Field(
        default=None,
        description="Consumo energético em kWh quando informado."
    )

    valor_faturado: Optional[float] = Field(
        default=None,
        description="Valor faturado quando explicitamente informado."
    )

    necessita_suporte: bool = Field(
        default=False,
        description="Indica se o usuário deve procurar suporte especializado."
    )

    @field_validator("potencia_kw", "consumo_kwh", "valor_faturado")
    @classmethod
    def validar_valores_nao_negativos(cls, value):
        if value is not None and value < 0:
            raise ValueError(
                "Valores de potência, consumo e faturamento não podem ser negativos."
            )

        return value

    @field_validator("resposta")
    @classmethod
    def validar_resposta(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("A resposta não pode estar vazia.")

        return value