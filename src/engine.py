import os

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import (
    RunnableWithMessageHistory
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
        # MEMÓRIA POR SESSÃO
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
        # ADAPTADOR PARA HISTÓRICO
        #
        # Mantém a saída estruturada disponível
        # para avaliação, mas fornece uma string
        # para o histórico conversacional.
        # ==========================================

        self.chain_for_history = (
            self.chain
            | RunnableLambda(
                self._prepare_history_output
            )
        )

        # ==========================================
        # RUNNABLE COM MEMÓRIA POR SESSÃO
        # ==========================================

        self.chain_with_history = (
            RunnableWithMessageHistory(
                self.chain_for_history,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="history",
                output_messages_key="output",
            )
        )

    # ==============================================
    # HISTÓRICO DA SESSÃO
    # ==============================================

    def _get_session_history(
        self,
        session_id: str
    ):

        memory = self.memory_manager.get_memory(
            session_id
        )

        return TokenBufferChatMessageHistory(
            memory
        )

    # ==============================================
    # PREPARAÇÃO DA SAÍDA
    # ==============================================

    @staticmethod
    def _prepare_history_output(result):
        """
        Converte o objeto Pydantic em um dicionário.

        O campo 'output' é usado pelo
        RunnableWithMessageHistory para registrar
        somente a resposta textual na memória.

        O campo 'structured' preserva todos os dados
        validados pelo Pydantic.
        """

        if hasattr(result, "model_dump"):

            dados = result.model_dump()

            return {
                "output": result.resposta,
                "structured": dados,
            }

        return {
            "output": str(result),
            "structured": None,
        }

    # ==============================================
    # VALIDAÇÕES DE ENTRADA
    # ==============================================

    def _validate_input(
        self,
        user_input: str,
        session_id: str
    ):
        """
        Executa os guardrails antes de chamar o LLM.

        Retorna:
            (True, None) quando a entrada pode continuar.

        Ou:
            (False, mensagem) quando deve ser recusada.
        """

        # ==========================================
        # 1. PROMPT INJECTION
        # ==========================================

        allowed, moderation_message = (
            check_prompt_injection(user_input)
        )

        if not allowed:
            return False, moderation_message

        # ==========================================
        # 2. RISCOS DE SEGURANÇA
        # ==========================================

        safety_allowed, safety_message = (
            check_safety_risk(user_input)
        )

        if not safety_allowed:
            return False, safety_message

        # ==========================================
        # 3. ESCOPO
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
            return False, scope_message

        return True, None

    # ==============================================
    # EXECUÇÃO PRINCIPAL
    # ==============================================

    def _invoke(
        self,
        user_input: str,
        session_id: str
    ):
        """
        Executa os guardrails e a chain LCEL.

        Retorna um dicionário com:
        - output: resposta textual
        - structured: saída validada pelo Pydantic
        """

        allowed, message = self._validate_input(
            user_input,
            session_id,
        )

        if not allowed:

            return {
                "output": message,
                "structured": None,
                "blocked": True,
            }

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

            return {
                "output": response.get(
                    "output",
                    ""
                ),
                "structured": response.get(
                    "structured"
                ),
                "blocked": False,
            }

        except Exception as e:

            return {
                "output": (
                    "Não foi possível processar "
                    "a solicitação no momento."
                ),
                "structured": None,
                "blocked": False,
                "error": str(e),
            }

    # ==============================================
    # ANALYZE — API PRINCIPAL DO CHATBOT
    # ==============================================

    def analyze(
        self,
        user_input: str,
        session_id: str = "default",
    ):

        result = self._invoke(
            user_input,
            session_id,
        )

        return result["output"]

    # ==============================================
    # ANALYZE STRUCTURED
    # ==============================================

    def analyze_structured(
        self,
        user_input: str,
        session_id: str = "default",
    ):
        """
        Retorna a saída estruturada validada pelo Pydantic.

        Retorna None quando a solicitação é bloqueada
        por algum guardrail ou quando ocorre erro.
        """

        result = self._invoke(
            user_input,
            session_id,
        )

        return result["structured"]

    # ==============================================
    # ANALYZE DETAILED
    # ==============================================

    def analyze_detailed(
        self,
        user_input: str,
        session_id: str = "default",
    ):
        """
        Retorna todas as informações úteis para avaliação.

        Útil para os scripts de eval sem alterar
        o comportamento normal da interface.
        """

        return self._invoke(
            user_input,
            session_id,
        )

    # ==============================================
    # INFORMAÇÕES DA MEMÓRIA
    # ==============================================

    def get_memory_token_count(
        self,
        session_id: str = "default"
    ) -> int:

        return self.memory_manager.get_token_count(
            session_id
        )

    def get_history(
        self,
        session_id: str = "default"
    ):

        return self.memory_manager.get_history(
            session_id
        )

    # ==============================================
    # LIMPAR SESSÃO
    # ==============================================

    def clear_session(
        self,
        session_id: str = "default"
    ):

        self.memory_manager.clear(
            session_id
        )