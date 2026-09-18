from typing import Dict

from langchain.memory import ConversationTokenBufferMemory
from langchain_ollama import ChatOllama


class SessionMemoryManager:
    """
    Gerencia memória independente para cada sessão.
    """

    def __init__(
        self,
        llm: ChatOllama,
        max_token_limit: int = 2000,
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

    def get_history(self, session_id: str):

        return self.get_memory(
            session_id
        ).chat_memory.messages

    def has_history(self, session_id: str) -> bool:

        return len(
            self.get_history(session_id)
        ) > 0

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        ai_message: str,
    ):

        memory = self.get_memory(
            session_id
        )

        memory.save_context(
            {
                "input": user_message
            },
            {
                "output": ai_message
            }
        )

    def clear(self, session_id: str):

        if session_id in self.memories:
            self.memories[session_id].clear()