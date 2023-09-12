"""
Web-based Chat App using Streamlit, LLamaIndex and OpenaI.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
import streamlit as st
from llama_index import VectorStoreIndex, load_index_from_storage, StorageContext
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("OPENAI_API_KEY environment variable not set.")


st.header("Chat with LLamaIndex Docs 💬📚")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about LlamaIndex Documentation!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data() -> VectorStoreIndex:
    """Load VectoreStoreIndex"""
    with st.spinner("Loading Vectore Store Index..."):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
        return index


def display_chat_history(messages):
    """Display previous chat messages."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def generate_assistant_response(prompt, chat_engine):
    """Generate assistant response and update session state."""
    with st.chat_message("assistant"):
        with st.spinner("I am on it..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.response})


# Main 
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# User input
user_input = st.chat_input("Your question")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

# Display previous chat
display_chat_history(st.session_state.messages)

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    generate_assistant_response(user_input, chat_engine)
