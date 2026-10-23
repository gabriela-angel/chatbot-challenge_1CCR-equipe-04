
import os

from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)
from langchain_ollama import ChatOllama

from src.chain.builder import build_chain
from src.chain.memoria import (
    SessionMemoryManager,
    TokenBufferChatMessageHistory,
)
from src.guardrails.scope_validator import validate_scope
from src.guardrails.moderation import (
    check_prompt_injection,
    check_safety_risk,
)
from src.schemas.consulta_recarga import ConsultaRecarga

load_dotenv()


class MissionEngine:
    """Motor conversacional com memória, guardrails e RAG."""

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 800,
        max_memory_tokens: int = 2500,
        use_rag: bool = True,
        rag_k: int = 4,
        rag_min_score: float = 0.20,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_memory_tokens = max_memory_tokens
        self.use_rag = use_rag
        self.rag_k = rag_k
        self.rag_min_score = rag_min_score

        self._retriever = None

        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY não encontrada no arquivo .env."
            )

        self.llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.max_tokens,
            base_url="https://ollama.com",
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {api_key}"
                }
            },
        )

        # Mantém a memória por sessão existente na Sprint 3.
        self.memory_manager = SessionMemoryManager(
            llm=self.llm,
            max_token_limit=self.max_memory_tokens,
        )

        # Chain anterior: mantida para compatibilidade e comparação.
        self.chain = build_chain(llm=self.llm)
        self.chain_with_history = self._add_session_history(
            self.chain
        )

        # Chain RAG: só é construída quando solicitada.
        self.rag_chain_with_history = None

        if self.use_rag:
            from src.rag.prompt_rag import build_rag_chain

            rag_chain = build_rag_chain(llm=self.llm)
            self.rag_chain_with_history = (
                self._add_session_history(rag_chain)
            )

    def _add_session_history(self, chain):
        """Adiciona histórico à chain sem alterar o schema de saída."""

        chain_for_history = (
            chain
            | RunnableLambda(self._prepare_history_output)
        )

        return RunnableWithMessageHistory(
            chain_for_history,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

    def _get_session_history(self, session_id: str):
        memory = self.memory_manager.get_memory(session_id)
        return TokenBufferChatMessageHistory(memory)

    @staticmethod
    def _prepare_history_output(result):
        """Converte a saída Pydantic para o formato da memória."""

        if hasattr(result, "model_dump"):
            data = result.model_dump()
            return {
                "output": result.resposta,
                "structured": data,
            }

        return {
            "output": str(result),
            "structured": None,
        }

    def _validate_input(
        self,
        user_input: str,
        session_id: str,
    ) -> tuple[bool, str | None]:
        """Executa os guardrails antes de consultar o modelo."""

        allowed, message = check_prompt_injection(user_input)
        if not allowed:
            return False, message

        allowed, message = check_safety_risk(user_input)
        if not allowed:
            return False, message

        has_history = self.memory_manager.has_history(session_id)

        allowed, message = validate_scope(
            user_input,
            has_history=has_history,
        )
        if not allowed:
            return False, message

        return True, None

    def _get_retriever(self):
        """Inicializa o retriever somente quando for necessário."""

        if self._retriever is None:
            from src.rag.retriever import KnowledgeRetriever
            from src.rag.vector_store import open_vector_store

            vector_store = open_vector_store()

            self._retriever = KnowledgeRetriever(
                vector_store=vector_store,
                k=self.rag_k,
                min_score=self.rag_min_score,
            )

        return self._retriever

    @staticmethod
    def _append_sources(
        answer: str,
        sources: list[str],
    ) -> str:
        """Garante que as fontes recuperadas sejam exibidas."""

        missing_sources = [
            source for source in sources
            if source not in answer
        ]

        if not missing_sources:
            return answer

        source_lines = "\n".join(
            f"- {source}" for source in missing_sources
        )

        return (
            f"{answer.rstrip()}\n\n"
            f"**Fontes consultadas:**\n{source_lines}"
        )

    @staticmethod
    def _blocked_result(
        message: str,
        *,
        reason: str,
    ) -> dict:
        return {
            "output": message,
            "structured": None,
            "blocked": True,
            "reason": reason,
            "sources": [],
            "retrieval_score": None,
        }

    def _invoke(
        self,
        user_input: str,
        session_id: str,
    ) -> dict:
        """Executa validações, recuperação e geração da resposta."""

        if not isinstance(user_input, str) or not user_input.strip():
            return self._blocked_result(
                "Digite uma pergunta para continuar.",
                reason="empty_input",
            )

        user_input = user_input.strip()

        allowed, message = self._validate_input(
            user_input,
            session_id,
        )

        if not allowed:
            return self._blocked_result(
                message or "Solicitação bloqueada pelos guardrails.",
                reason="guardrail",
            )

        sources: list[str] = []
        retrieval_score = None

        try:
            if self.use_rag:
                if self.rag_chain_with_history is None:
                    raise RuntimeError("A chain RAG não foi inicializada.")

                retrieval = self._get_retriever().retrieve(user_input)
                retrieval_score = retrieval.best_score

                # Sem evidência relevante, não consultamos o LLM.
                if not retrieval.documents or not retrieval.context.strip():
                    return {
                        **self._blocked_result(
                            "Não encontrei informações suficientemente "
                            "relevantes na base de conhecimento para "
                            "responder com segurança. Tente reformular "
                            "a pergunta ou consulte a equipe de suporte.",
                            reason="no_relevant_context",
                        ),
                        "retrieval_score": retrieval_score,
                    }

                sources = retrieval.sources

                response = self.rag_chain_with_history.invoke(
                    {
                        "input": user_input,
                        "context": retrieval.context,
                    },
                    config={
                        "configurable": {
                            "session_id": session_id,
                        }
                    },
                )
            else:
                # Modo de compatibilidade com a Sprint 3.
                response = self.chain_with_history.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {
                            "session_id": session_id,
                        }
                    },
                )

            answer = str(response.get("output", "")).strip()
            structured = response.get("structured")

            if not answer:
                return {
                    "output": (
                        "Não foi possível gerar uma resposta válida."
                    ),
                    "structured": structured,
                    "blocked": False,
                    "reason": "empty_model_response",
                    "sources": sources,
                    "retrieval_score": retrieval_score,
                }

            if self.use_rag:
                answer = self._append_sources(answer, sources)

                if isinstance(structured, dict):
                    structured = dict(structured)
                    structured["resposta"] = answer

            return {
                "output": answer,
                "structured": structured,
                "blocked": False,
                "reason": None,
                "sources": sources,
                "retrieval_score": retrieval_score,
            }

        except Exception:
            # Não expõe detalhes internos, credenciais ou exceções ao usuário.
            return {
                "output": (
                    "Não foi possível processar a solicitação. "
                    "Verifique a configuração do Ollama, a indexação "
                    "da base de conhecimento e os logs da aplicação."
                ),
                "structured": None,
                "blocked": False,
                "reason": "processing_error",
                "sources": [],
                "retrieval_score": retrieval_score,
            }

    def analyze(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> str:
        """Mantém a API de texto usada pela interface atual."""

        return self._invoke(user_input, session_id)["output"]

    def analyze_structured(
        self,
        user_input: str,
        session_id: str = "default",
    ):
        """Retorna os dados validados pelo schema Pydantic."""

        return self._invoke(user_input, session_id)["structured"]

    def analyze_detailed(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> dict:
        """Retorna resposta, fontes e metadados para avaliação."""

        return self._invoke(user_input, session_id)

    def get_memory_token_count(
        self,
        session_id: str = "default",
    ) -> int:
        return self.memory_manager.get_token_count(session_id)

    def get_history(
        self,
        session_id: str = "default",
    ):
        return self.memory_manager.get_history(session_id)

    def clear_session(
        self,
        session_id: str = "default",
    ) -> None:
        self.memory_manager.clear(session_id)
