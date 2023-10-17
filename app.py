"""
Streamlit application that integrates with LlamaIndex and OpenAI's GPT-3.5 to create a conversational interface. 
Users can ask questions about LlamaIndex Docs, and the application provides relevant answers. 
The user's OpenAI API key is used to fetch responses from GPT-3.5.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
import streamlit as st
from llama_index import load_index_from_storage, StorageContext, VectorStoreIndex
import openai


st.set_page_config(
    page_title="Chat with LlamaIndex Docs",
    page_icon="🦙",
    initial_sidebar_state="expanded",
    menu_items={"About": "Built by @dcarpintero with Streamlit & LLamaIndex"},
)

if 'input_token_counter' not in st.session_state:
    st.session_state['input_token_counter'] = 0

if 'output_token_counter' not in st.session_state:
    st.session_state['output_token_counter'] = 0

if 'openai_api_key' in st.session_state:
    openai.api_key = st.session_state['openai_api_key']

@st.cache_resource(show_spinner=False)
def load_data() -> VectorStoreIndex:
    """Load VectorStoreIndex from storage."""

    with st.spinner("Loading Vector Store Index..."):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
        return index


def display_chat_history(messages):
    """Display previous chat messages."""

    for message in messages:
        with st.chat_message(message["role"]):
            if st.session_state.with_sources:
                if "sources" in message:
                    st.info(f'The sources of this response are:\n\n {message["sources"]}')
            st.write(message["content"])


def clear_chat_history():
    """"Clear chat history and reset questions' buttons."""

    st.session_state.messages = [
            {"role": "assistant", "content": "Try one of the sample questions or ask your own!"}
        ]
    st.session_state["btn_llama_index"] = False
    st.session_state["btn_retriever"] = False
    st.session_state["btn_diff"] = False
    st.session_state["btn_rag"] = False


def generate_assistant_response(prompt, chat_engine):
    """Generate assistant response and update token counter."""

    with st.chat_message("assistant"):
        with st.spinner("I am on it..."):
            if st.session_state.with_cache:
                response = query_chatengine_cache(prompt, chat_engine)
            else:
                response = query_chatengine(prompt, chat_engine)

            message = {"role": "assistant", "content": response.response, "sources": format_sources(response)}
            if st.session_state.with_sources:
                st.info(f'The sources of this response are:\n\n {message["sources"]}')
            st.write(message["content"])
            
            st.session_state.messages.append(message)
            update_token_counters(response)
            

@st.cache_data(max_entries=1024, show_spinner=False)
def query_chatengine_cache(prompt, _chat_engine):
    """Query chat engine and cache results."""
    return _chat_engine.chat(prompt)


def query_chatengine(prompt, chat_engine):
    """Query chat engine."""	
    return chat_engine.chat(prompt)


def format_sources(response):
    """Format filename, authors and scores of the response source nodes."""
    base = "https://github.com/jerryjliu/llama_index/tree/main/"
    return "\n".join([f"- {base}{source['filename']} (author: '{source['author']}'; score: {source['score']})\n" for source in get_metadata(response)])


def get_metadata(response):
    """Parse response source nodes and return a list of dictionaries with filenames, authors and scores.""" 
    
    sources = []
    for item in response.source_nodes:
        if hasattr(item, "metadata"):
            filename = item.metadata.get('filename').replace('\\', '/')
            author = item.metadata.get('author')
            score = float("{:.3f}".format(item.score))
            sources.append({'filename': filename, 'author': author, 'score': score})
    
    return sources


def update_token_counters(response):
    """Update token counters (1,000 tokens is about 750 words)"""

    # update input token counter
    for item in response.source_nodes:
        st.session_state['input_token_counter'] += round( 0.75 * len(item.text) )

    # update output token counter
    st.session_state['output_token_counter'] += round( 0.75 * len(response.response) )


def sidebar():
    """Configure the sidebar and user's preferences."""
    
    with st.sidebar.expander("🔑 OPENAI-API-KEY", expanded=True):
        st.text_input(label='OPENAI-API-KEY', type='password', key='openai_api_key', label_visibility='hidden').strip()
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    with st.sidebar.expander("💲 GPT3.5 INFERENCE COST", expanded=True):
        i_tokens = st.session_state['input_token_counter']
        o_tokens = st.session_state['output_token_counter']
        st.markdown(f'Input: {i_tokens} tokens')
        st.markdown(f'Output: {o_tokens} tokens')

        i_cost = (i_tokens / 1000) * 0.0015
        o_cost = (o_tokens / 1000) * 0.002
        st.markdown('**Cost Estimation: ${0}**'.format(round(i_cost + o_cost, 5)))
        "[OpenAI Pricing](https://openai.com/pricing)"

    with st.sidebar.expander("🔧 SETTINGS", expanded=True):
        st.toggle('Cache Results', value=True, key="with_cache")
        st.toggle('Display Sources', value=True, key="with_sources")
        st.toggle('Streaming', value=False, disabled=True, key="with_streaming")

    st.sidebar.button('Clear Messages', type="primary", on_click=clear_chat_history) 
    st.sidebar.divider()
    with st.sidebar:
        col_ll, col_gh = st.columns([1, 1])
        with col_ll:
            "[![LlamaIndex Docs](https://img.shields.io/badge/LlamaIndex%20Docs-gray)](https://gpt-index.readthedocs.io/en/latest/index.html)"
        with col_gh:
            "[![Github](https://img.shields.io/badge/Github%20Repo-gray?logo=Github)](https://github.com/dcarpintero/llamaindexchat)"

def layout():
    """"Layout"""

    st.header("Chat with 🦙 LlamaIndex Docs 🗂️")

    # Get Started
    if not openai.api_key:
        st.warning("Hi there! Add your OPENAI-API-KEY on the sidebar field to get started!\n\n", icon="🚨")
        st.stop()

    # Load Index
    index = load_data()
    if index:
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

    # System Message
    if "messages" not in st.session_state:    
        st.session_state.messages = [
            {"role": "assistant", "content": "Try one of the sample questions or ask your own!"}
        ]

    # User input
    user_input = st.chat_input("Your question")
    if user_input or user_input_button:
        st.session_state.messages.append({"role": "user", "content": user_input or user_input_button})

    # Display previous chat
    display_chat_history(st.session_state.messages)

    # Generate response
    if st.session_state.messages[-1]["role"] != "assistant":
        try:
            generate_assistant_response(user_input or user_input_button, chat_engine)
        except Exception as ex:
            st.error(str(ex))
        

def main():
    """Set up user preferences, and layout"""

    sidebar()
    layout()

if __name__ == "__main__":
    main()
