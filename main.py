import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage

llm = ChatOllama(model="llama3")
st.title("LLM Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_llm_response(history):
    try:
        response = llm.invoke(history)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"
    
st.session_state.setdefault("input_value", "")

user_input = st.text_input("You:", key="input", value=st.session_state.get("input_value", ""), placeholder="Type your message...")

if st.button("Send") and user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    ai_response = get_llm_response(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.rerun()

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    else:
        st.markdown(f"**Llama:** {msg.content}")