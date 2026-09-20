from typing import Dict, List

from langchain.memory import ConversationTokenBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama


class SessionMemoryManager:
    """
    Gerencia uma ConversationTokenBufferMemory independente
    para cada sessão do chatbot.

    Cada sessão possui seu próprio histórico e o histórico
    é limitado pelo número máximo de tokens configurado.
    """

    def __init__(
        self,
        llm: ChatOllama,
        max_token_limit: int = 2500,
    ):
        self.llm = llm
        self.max_token_limit = max_token_limit

        self.memories: Dict[
            str,
            ConversationTokenBufferMemory
        ] = {}

    def get_memory(
        self,
        session_id: str
    ) -> ConversationTokenBufferMemory:

        if session_id not in self.memories:

            self.memories[session_id] = (
                ConversationTokenBufferMemory(
                    llm=self.llm,
                    max_token_limit=self.max_token_limit,
                    return_messages=True,
                    memory_key="history",
                )
            )

        return self.memories[session_id]

    def get_history(
        self,
        session_id: str
    ) -> List[BaseMessage]:

        return list(
            self.get_memory(session_id)
            .chat_memory
            .messages
        )

    def has_history(
        self,
        session_id: str
    ) -> bool:

        return len(
            self.get_history(session_id)
        ) > 0

    def clear(
        self,
        session_id: str
    ):

        if session_id in self.memories:
            self.memories[session_id].clear()

    def get_token_count(
        self,
        session_id: str
    ) -> int:

        """
        Retorna uma estimativa do número de tokens atualmente
        armazenados na memória da sessão.
        """

        memory = self.get_memory(session_id)

        try:
            return memory.llm.get_num_tokens_from_messages(
                memory.chat_memory.messages
            )
        except Exception:
            return 0


class TokenBufferChatMessageHistory(BaseChatMessageHistory):
    """
    Adaptador que permite utilizar ConversationTokenBufferMemory
    diretamente com RunnableWithMessageHistory.

    O RunnableWithMessageHistory trabalha com BaseChatMessageHistory,
    enquanto o projeto precisa manter ConversationTokenBufferMemory
    como mecanismo de controle do limite de tokens.
    """

    def __init__(
        self,
        memory: ConversationTokenBufferMemory
    ):
        self.memory = memory

    @property
    def messages(self) -> List[BaseMessage]:
        return list(
            self.memory.chat_memory.messages
        )

    def add_messages(
        self,
        messages: List[BaseMessage]
    ) -> None:

        self.memory.chat_memory.add_messages(
            messages
        )

        # Aplica o limite definido em max_token_limit.
        self.memory.prune()

    def add_message(
        self,
        message: BaseMessage
    ) -> None:

        self.add_messages([message])

    def clear(self) -> None:

        self.memory.clear()