from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_system_prompt():
    return SystemMessage(
        content="""
You are a helpful assistant. Answer all questions to the best of your ability.
If you don't know the answer, say "I don't know". If you are unsure, say "I'm not sure".
            """
    )
