import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage

llm = ChatOllama(model="llama3.2")
st.title("LLM Translator")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False


def get_llm_response(history: list[HumanMessage | AIMessage]) -> str:
    try:
        response = llm.invoke(history)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("human"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input(
    key="user_input",
    placeholder="Type your request...",
    disabled=st.session_state.generation_in_progress,
):
    print("Button clicked")
    print("User prompt: ", st.session_state.user_input)

    st.session_state.generation_in_progress = True

    with st.chat_message("human"):
        st.markdown(st.session_state.user_input)

    st.session_state.chat_history.append(
        HumanMessage(content=st.session_state.user_input)
    )

    with st.chat_message("assistant"):
        with st.empty():
            with st.spinner("Generating response...", show_time=True):
                ai_response = get_llm_response(st.session_state.chat_history)
            st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(content=ai_response))
    st.session_state.generation_in_progress = False
