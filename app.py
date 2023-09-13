"""
Web-based App to chat with LlamaIndex Docs using Streamlit, LLamaIndex and OpenaI.

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


st.header("Chat with 🦙 LlamaIndex Docs 🗂️")

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
            response = query_chatengine(prompt, chat_engine)

            st.info(extract_filenames(response.source_nodes))
            st.write(response.response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.response})
            

@st.cache_data(max_entries=1024, show_spinner=False)
def query_chatengine(prompt, _chat_engine):
    return _chat_engine.chat(prompt)


def extract_filenames(source_nodes):
    src = f"The sources of this response are:\n"
    for item in source_nodes:
        if hasattr(item, "metadata"):
            filename = f"\n'{item.metadata.get('filename')}'\n"
            src += filename
    return src

def sidebar():
    """Configure the sidebar and return the user's preferences."""
    
    with st.sidebar.expander("🔑 OPENAI-API-KEY", expanded=True):
        openai_api_key = st.text_input(label='OPENAI-API-KEY', type='password', key='openai_api_key', label_visibility='hidden').strip()
        if not openai_api_key:
            st.warning('Please enter a valid key')

    with st.sidebar.expander("💲 COST ESTIMATION", expanded=True):
        cost = st.markdown('Cost per 1k tokens: $0.002')

    with st.sidebar.expander("🔧 SETTINGS", expanded=True):
        cache = st.toggle('Cache Results', value=True)
        sources = st.toggle('Display Sources', value=True)
        streaming = st.toggle('Streaming', value=False)

    clear = st.sidebar.button('Clear Messages', type="primary")

    return openai_api_key, cache, sources, streaming, clear

def layout():
    # Main 
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    # Sample Questions for User input
    user_input_button = None

    btn_llama_index = st.session_state.get("btn_llama_index", False)
    btn_retriever = st.session_state.get("btn_retriever", False)
    btn_diff = st.session_state.get("btn_diff", False)
    btn_rag = st.session_state.get("btn_rag", False)

    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        if st.button("explain the basic usage pattern of LlamaIndex", type="primary", disabled=btn_llama_index):
            user_input_button = "explain the basic usage pattern in LlamaIndex"
            st.session_state.btn_llama_index = True
    with col2:
        if st.button("how can I ingest data from the GoogleDocsReader?", type="primary", disabled=btn_retriever):
            user_input_button = "how can I ingest data from the GoogleDocsReader?"
            st.session_state.btn_retriever = True
    with col3:
        if st.button("what's the difference between document & node?", type="primary", disabled=btn_diff):
            user_input_button = "what's the difference between document and node?"
            st.session_state.btn_diff = True
    with col4:
        if st.button("how can I make a RAG application performant?", type="primary", disabled=btn_rag):
            user_input_button = "how can I make a RAG application performant?"
            st.session_state.btn_rag = True

    # User input
    user_input = st.chat_input("Your question")
    if user_input or user_input_button:
        st.session_state.messages.append({"role": "user", "content": user_input or user_input_button})

    # Display previous chat
    display_chat_history(st.session_state.messages)

    # Generate response
    if st.session_state.messages[-1]["role"] != "assistant":
        generate_assistant_response(user_input or user_input_button, chat_engine)

def main():
    """
    Set up user preferences, and layout.
    """
    #st.set_page_config(
    #    page_title="Chat wih LLMs Papers - Powered by LLamaIndex",
    #    page_icon="💬📚",
    #    initial_sidebar_state="expanded",
    #    menu_items={"About": "Built by @dcarpintero with Streamlit & LLamaIndex"},
    #)

    openai_api_key, cache, sources, streaming, clear = sidebar()
    layout()

if __name__ == "__main__":
    main()
