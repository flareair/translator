import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage

llm = ChatOllama(model="llama3.2")
st.title("LLM Translator")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_llm_response(history):
    try:
        response = llm.invoke(history)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"


input_text = st.text_area(
    "You:",
    key="user_input",
    value=st.session_state.get("input_value"),
    placeholder="Type your message...",
)


def onButtonClick():
    if input_text == "":
        return

    print("Button clicked")
    print("User prompt: ", st.session_state.user_input)

    st.session_state.chat_history.append(HumanMessage(content=input_text))
    ai_response = get_llm_response(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    st.session_state.user_input = ""  # Clean up the input field


st.button("Send", on_click=onButtonClick)

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    else:
        st.markdown(f"**Llama:** {msg.content}")
