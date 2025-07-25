import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from llm import create_invocation_chain

invocation_chain = create_invocation_chain()

st.title("LLM Translator")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("human"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input(key="user_input", placeholder="Type your request..."):
    print("Button clicked")
    print("User prompt: ", st.session_state.user_input)

    with st.chat_message("human"):
        st.markdown(st.session_state.user_input)

    st.session_state.chat_history.append(
        HumanMessage(content=st.session_state.user_input)
    )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Generating response...", show_time=True):
            ai_response = ""

            # Use the stream method for token-wise generation
            try:
                for i, chunk in enumerate(
                    invocation_chain.stream({"messages": st.session_state.chat_history})
                ):
                    token = getattr(chunk, "content", "")
                    ai_response += token
                    response_placeholder.markdown(ai_response)
            except Exception as e:
                ai_response = f"Error: {e}"
                response_placeholder.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(content=ai_response))
