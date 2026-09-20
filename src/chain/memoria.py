from typing import Dict, List

from langchain.memory import ConversationTokenBufferMemory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from src.utils.tokens import contar_tokens_mensagens


class TokenBufferChatMessageHistory:
    """
    Adaptador entre RunnableWithMessageHistory e
    ConversationTokenBufferMemory.

    O histórico exposto ao RunnableWithMessageHistory
    é uma lista de mensagens LangChain, enquanto o
    ConversationTokenBufferMemory continua sendo
    responsável pelo limite de tokens.
    """

    def __init__(
        self,
        memory: ConversationTokenBufferMemory,
    ):
        self.memory = memory

    @property
    def messages(self) -> List[BaseMessage]:
        return list(
            self.memory.chat_memory.messages
        )

    def add_message(
        self,
        message: BaseMessage,
    ) -> None:

        if isinstance(message, HumanMessage):

            self.memory.chat_memory.add_user_message(
                message.content
            )

        elif isinstance(message, AIMessage):

            self.memory.chat_memory.add_ai_message(
                message.content
            )

        else:

            self.memory.chat_memory.add_message(
                message
            )

        self._prune_if_needed()

    def add_messages(
        self,
        messages: List[BaseMessage],
    ) -> None:

        for message in messages:
            self.add_message(message)

    def clear(self) -> None:

        self.memory.clear()

    def _prune_if_needed(self) -> None:
        """
        Mantém o histórico dentro do limite configurado.

        Algumas versões do LangChain não expõem prune()
        diretamente em ConversationTokenBufferMemory.
        Por isso fazemos a poda explicitamente.
        """

        messages = self.memory.chat_memory.messages

        if not messages:
            return

        try:

            token_count = contar_tokens_mensagens(
                messages
            )

        except Exception:

            return

        while (
            token_count
            > self.memory.max_token_limit
            and len(messages) > 2
        ):

            # Remove o par mais antigo:
            # pergunta do usuário + resposta do modelo.

            del messages[0]

            if messages:
                del messages[0]

            token_count = (
                contar_tokens_mensagens(
                    messages
                )
            )


class SessionMemoryManager:
    """
    Gerencia uma memória independente para cada sessão.
    """

    def __init__(
        self,
        llm,
        max_token_limit: int = 2500,
    ):

        self.llm = llm

        self.max_token_limit = (
            max_token_limit
        )

        self._memories: Dict[
            str,
            ConversationTokenBufferMemory
        ] = {}

    def get_memory(
        self,
        session_id: str,
    ) -> ConversationTokenBufferMemory:

        if session_id not in self._memories:

            self._memories[
                session_id
            ] = ConversationTokenBufferMemory(
                llm=self.llm,
                max_token_limit=(
                    self.max_token_limit
                ),
                return_messages=True,
                memory_key="history",
            )

        return self._memories[
            session_id
        ]

    def get_history(
        self,
        session_id: str,
    ) -> TokenBufferChatMessageHistory:

        return TokenBufferChatMessageHistory(
            self.get_memory(session_id)
        )

    def has_history(
        self,
        session_id: str,
    ) -> bool:

        memory = self.get_memory(
            session_id
        )

        return bool(
            memory.chat_memory.messages
        )

    def get_token_count(
        self,
        session_id: str,
    ) -> int:

        memory = self.get_memory(
            session_id
        )

        return contar_tokens_mensagens(
            memory.chat_memory.messages
        )

    def clear(
        self,
        session_id: str,
    ) -> None:

        if session_id in self._memories:

            self._memories[
                session_id
            ].clear()

    def clear_all(self) -> None:

        for memory in self._memories.values():
            memory.clear()

        self._memories.clear()