"""Interface web (Gradio) — GoodWe ChargeGrid Intelligence, Sprint 04.

Execução, a partir da raiz do projeto:
    python app/app.py
"""
import logging
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(
    filename=str(ROOT / "chatbot.log"),
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_engine = None

STATUS_POR_MOTIVO = {
    "guardrail": "🛡️ Solicitação bloqueada pelos guardrails de segurança/escopo.",
    "no_relevant_context": "⚠️ Nenhum trecho relevante encontrado na base de conhecimento.",
    "insufficient_evidence": "⚠️ A base não traz evidência suficiente para responder.",
    "processing_error": "❌ Erro de processamento. Consulte o arquivo chatbot.log.",
    "empty_model_response": "❌ O modelo não retornou uma resposta válida.",
    "greeting": "👋 Saudação.",
    "empty_input": "Digite uma pergunta.",
}


def get_engine():
    """Cria o engine na primeira chamada (evita falhar na importação)."""
    global _engine
    if _engine is None:
        from src.engine import MissionEngine

        _engine = MissionEngine(
            model=os.getenv("RAG_MODEL", "gpt-oss:120b"),
            temperature=0.0,
            top_p=0.9,
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", "800")),
            use_rag=True,
            rag_k=int(os.getenv("RAG_K", "4")),
            rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.20")),
            rag_prompt_version=os.getenv("RAG_PROMPT_VERSION") or None,
            collection_name=os.getenv("RAG_COLLECTION", "goodwe_knowledge_base"),
        )
    return _engine


def _linhas_fontes(fontes: list[dict]) -> list[list]:
    return [
        [
            fonte.get("source", ""),
            fonte.get("page") if fonte.get("page") is not None else "—",
            fonte.get("score") if fonte.get("score") is not None else "—",
        ]
        for fonte in fontes or []
    ]


def _trechos_markdown(contextos: list[str]) -> str:
    if not contextos:
        return "_Nenhum trecho recuperado._"
    blocos = []
    for indice, texto in enumerate(contextos, start=1):
        resumo = " ".join(texto.split())
        if len(resumo) > 320:
            resumo = resumo[:320] + "…"
        blocos.append(f"**Trecho {indice}:** {resumo}")
    return "\n\n".join(blocos)


def _status(detalhe: dict | None) -> str:
    if not detalhe:
        return ""
    motivo = detalhe.get("reason")
    if motivo in STATUS_POR_MOTIVO:
        return STATUS_POR_MOTIVO[motivo]
    total = len(detalhe.get("sources") or [])
    score = detalhe.get("retrieval_score")
    texto_score = f" Melhor relevância: {score:.2f}." if isinstance(score, float) else ""
    return f"✅ Resposta fundamentada em {total} fonte(s).{texto_score}"


def responder(mensagem, historico, session_id):
    historico = list(historico or [])
    mensagem = (mensagem or "").strip()

    if not mensagem:
        return historico, "", [], "_Nenhum trecho recuperado._", ""

    try:
        detalhe = get_engine().analyze_detailed(mensagem, session_id=session_id)
    except RuntimeError as erro:
        # Ex.: OLLAMA_API_KEY ausente. A mensagem não contém credenciais.
        detalhe = {
            "output": f"Configuração incompleta: {erro}",
            "reason": "processing_error",
            "sources": [],
            "contexts": [],
        }

    historico.append({"role": "user", "content": mensagem})
    historico.append({"role": "assistant", "content": detalhe["output"]})

    return (
        historico,
        "",
        _linhas_fontes(detalhe.get("sources")),
        _trechos_markdown(detalhe.get("contexts") or []),
        _status(detalhe),
    )


def nova_conversa(session_id):
    try:
        get_engine().clear_session(session_id)
    except RuntimeError:
        pass
    return [], [], "_Nenhum trecho recuperado._", "", uuid.uuid4().hex


def construir_interface() -> gr.Blocks:
    tema = gr.themes.Soft(primary_hue=gr.themes.colors.red)

    with gr.Blocks(title="GoodWe ChargeGrid Intelligence", theme=tema) as demo:
        gr.Markdown(
            "# ◆ GoodWe ChargeGrid Intelligence\n"
            "Assistente com **RAG**: responde apenas com base na documentação "
            "indexada e **cita a fonte** (documento e página)."
        )
        session_id = gr.State(lambda: uuid.uuid4().hex)

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", height=480, label="Conversa")
                status = gr.Markdown("")
                with gr.Row():
                    entrada = gr.Textbox(
                        placeholder="Pergunte sobre recarga, potência, instalação, MODBUS…",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    enviar = gr.Button("Enviar", variant="primary", scale=1)
                limpar = gr.Button("Nova conversa")

            with gr.Column(scale=2):
                gr.Markdown("### Fontes citadas")
                tabela = gr.Dataframe(
                    headers=["Documento", "Página", "Relevância"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    value=[],
                )
                with gr.Accordion("Trechos recuperados", open=False):
                    trechos = gr.Markdown("_Nenhum trecho recuperado._")

        saidas = [chat, entrada, tabela, trechos, status]
        enviar.click(responder, [entrada, chat, session_id], saidas, api_name="responder")
        entrada.submit(responder, [entrada, chat, session_id], saidas, api_name=False)
        limpar.click(
            nova_conversa,
            [session_id],
            [chat, tabela, trechos, status, session_id],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    construir_interface().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )