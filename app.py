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
        {"role": "assistant", "content": "Try one of the sample questions or ask your own!"}
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

# Sample Questions for User input
user_input_button = None

btn_llama_index = st.session_state.get("btn_llama_index", False)
btn_rag = st.session_state.get("btn_rag", False)
btn_retriever = st.session_state.get("btn_retriever", False)

if st.button("What is LlamaIndex?", type="primary", disabled=btn_llama_index):
    user_input_button = "What is LlamaIndex?"
    st.session_state.btn_llama_index = True

if st.button("how to make my RAG application performant?", type="primary", disabled=btn_rag):
    user_input_button = "how to make my RAG application performant?"
    st.session_state.btn_rag = True

if st.button("how to use a loader to retrieve content from a Notion page?", type="primary", disabled=btn_retriever):
    user_input_button = "how to use a loader to retrieve content from a Notion page?"
    st.session_state.btn_retriever = True

# User input
user_input = st.chat_input("Your question")
if user_input or user_input_button:
    st.session_state.messages.append({"role": "user", "content": user_input or user_input_button})

# Display previous chat
display_chat_history(st.session_state.messages)

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    generate_assistant_response(user_input or user_input_button, chat_engine)
