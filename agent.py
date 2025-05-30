from typing import Annotated, List
from langchain.schema import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model

from langchain.schema import SystemMessage
from system_prompt import get_system_prompt


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0.5)


def chat_bot(state: State):
    messages = state["messages"]
    # Check if the first message is a SystemMessage
    if not messages or not isinstance(messages[0], SystemMessage):
        system_message = get_system_prompt()
        messages = [system_message] + messages

    return {"messages": llm.invoke(state["messages"])}


graph_builder.add_node("chat_bot", chat_bot)
graph_builder.add_edge(START, "chat_bot")
graph_builder.add_edge("chat_bot", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
