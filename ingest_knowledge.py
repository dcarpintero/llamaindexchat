"""
Knowledge Ingestion of the markdown files in the docs folder of the llama_index repository:
- https://github.com/jerryjliu/llama_index

Built with LlamaIndex, GithubRepositoryReader, and OpenAI.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
from llama_index import download_loader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from dotenv import load_dotenv
import json
import openai
import os
import logging


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not set.")
    
    logging.info("Environment variables loaded.")
    return {"OPENAI_API_KEY": api_key, "GITHUB_TOKEN": github_token}


def initialize_github_loader(github_token: str) -> GithubRepositoryReader:
    """Initialize GithubRepositoryReader"""	

    download_loader("GithubRepositoryReader")
    github_client = GithubClient(github_token)

    loader = GithubRepositoryReader(
        github_client,
        owner =                  "jerryjliu",
        repo =                   "llama_index",
        filter_directories =     (["docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                False,
        concurrent_requests =    10,
    )

    return loader


def load_and_index_data(loader: GithubRepositoryReader) -> VectorStoreIndex:
    """Load and Index Knowledge Base from GitHub Repository"""

    logging.info("Loading data from Github: %s/%s", loader._owner, loader._repo)
    docs = loader.load_data(branch="main")
    for doc in docs:
        logging.info(doc.extra_info)
        doc.metadata = {'filename': doc.extra_info['file_name'], 'author': "LlamaIndex"}
        
    logging.info("Creating ServiceContext...")
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="You are a specialized AI trained in the usage of LlamaIndex.",
        )
    )

    logging.info("Indexing data...")
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
    logging.info("Persisting index on ./storage...")
    index.storage_context.persist(persist_dir="./storage")
        
    logging.info("Data-Knowledge ingestion process is completed (OK)")
    return index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()
        openai.api_key = env_vars['OPENAI_API_KEY']

        loader = initialize_github_loader(env_vars['GITHUB_TOKEN'])
        index = load_and_index_data(loader)
    except Exception as ex:
        logging.error("Unexpected Error: %s", ex)
        raise ex