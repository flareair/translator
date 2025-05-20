import streamlit as st

from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage

# Create a ChatOllama instance
llm = ChatOllama(model="llama3")

st.title("LLM Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    # Add human message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Generate AI response
    response = llm(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response.content.strip()))

# Show chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    else:
        st.markdown(f"**Llama:** {msg.content}")