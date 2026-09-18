import os

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)
from langchain_core.chat_history import (
    InMemoryChatMessageHistory
)
from langchain_ollama import ChatOllama

from src.chain.builder import build_chain
from src.chain.memoria import SessionMemoryManager
from src.guardrails.scope_validator import validate_scope
from src.guardrails.moderation import check_prompt_injection


load_dotenv()


class MissionEngine:

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 800,
        max_memory_tokens: int = 2500,
    ):

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_memory_tokens = max_memory_tokens

        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY não encontrada "
                "no arquivo .env"
            )

        # ==========================================
        # MODELO
        # ==========================================

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

        # ==========================================
        # MEMÓRIA TOKEN BUFFER
        # ==========================================

        self.memory_manager = SessionMemoryManager(
            llm=self.llm,
            max_token_limit=self.max_memory_tokens,
        )

        # ==========================================
        # CHAIN LCEL
        # ==========================================

        self.chain = build_chain(
            llm=self.llm
        )

        # ==========================================
        # HISTÓRICO DAS SESSÕES
        # ==========================================

        self.histories = {}

        # ==========================================
        # CHAIN PARA O RUNNABLE
        # ==========================================
        #
        # A chain original retorna ConsultaRecarga.
        #
        # O RunnableWithMessageHistory precisa receber
        # uma saída que possa ser armazenada como mensagem.
        #
        # Por isso convertemos o resultado Pydantic
        # para texto antes do gerenciamento da história.
        #

        self.chain_for_history = (
            self.chain
            | RunnableLambda(
                self._convert_result_to_text
            )
        )

        self.chain_with_history = (
            RunnableWithMessageHistory(
                self.chain_for_history,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
        )

    # ==============================================
    # HISTÓRICO DA SESSÃO
    # ==============================================

    def _get_session_history(
        self,
        session_id: str
    ):

        if session_id not in self.histories:

            self.histories[session_id] = (
                InMemoryChatMessageHistory()
            )

        return self.histories[session_id]

    # ==============================================
    # CONVERSÃO DA SAÍDA ESTRUTURADA
    # ==============================================

    @staticmethod
    def _convert_result_to_text(result):

        if hasattr(result, "resposta"):

            return result.resposta

        return str(result)

    # ==============================================
    # MEMÓRIA TOKEN BUFFER
    # ==============================================

    def _save_to_memory(
        self,
        session_id: str,
        user_input: str,
        response: str,
    ):

        self.memory_manager.add_interaction(
            session_id=session_id,
            user_message=user_input,
            ai_message=response,
        )

    # ==============================================
    # ANALYZE
    # ==============================================

    def analyze(
        self,
        user_input: str,
        session_id: str = "default",
    ):

        # ==========================================
        # 1. PROMPT INJECTION
        # ==========================================

        allowed, moderation_message = (
            check_prompt_injection(user_input)
        )

        if not allowed:
            return moderation_message

        # ==========================================
        # 2. ESCOPO
        # ==========================================

        has_history = (
            self.memory_manager.has_history(
                session_id
            )
        )

        allowed, scope_message = (
            validate_scope(
                user_input,
                has_history=has_history,
            )
        )

        if not allowed:
            return scope_message

        # ==========================================
        # 3. EXECUÇÃO DA CHAIN
        # ==========================================

        try:

            response = (
                self.chain_with_history.invoke(
                    {
                        "input": user_input
                    },
                    config={
                        "configurable": {
                            "session_id": session_id
                        }
                    },
                )
            )

            # A saída já foi convertida para texto
            # pelo RunnableLambda.
            response = str(response)

            # ======================================
            # 4. TOKEN BUFFER
            # ======================================

            self._save_to_memory(
                session_id=session_id,
                user_input=user_input,
                response=response,
            )

            return response

        except Exception as e:

            return (
                "Não foi possível processar "
                "a solicitação no momento.\n\n"
                f"Detalhes técnicos: {e}"
            )

    # ==============================================
    # LIMPAR SESSÃO
    # ==============================================

    def clear_session(
        self,
        session_id: str = "default"
    ):

        self.histories.pop(
            session_id,
            None
        )

        self.memory_manager.clear(
            session_id
        )