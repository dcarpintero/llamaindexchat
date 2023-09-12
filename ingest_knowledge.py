"""
Knowledge Ingestion using LlamaIndex, GithubRepositoryReader, and OpenAI.

It will index all the markdown files in the docs folder of the llama_index repository.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
from llama_index import download_loader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from dotenv import load_dotenv
import openai
import os
import logging

def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not api_key or not github_token:
        raise EnvironmentError("Missing environment variables.")
    
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not set.")
    
    logging.info("Environment variables loaded.")
    return {"OPENAI_API_KEY": api_key, "GITHUB_TOKEN": github_token}

def index_knowledge() -> VectorStoreIndex:
    """Load and Index Knowledge Base"""

    env_vars = load_environment_vars()
    openai.api_key = env_vars['OPENAI_API_KEY']
    
    download_loader("GithubRepositoryReader")
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

    loader = GithubRepositoryReader(
        github_client,
        owner =                  "jerryjliu",
        repo =                   "llama_index",
        filter_directories =     (["docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                False,
        concurrent_requests =    10,
    )

    try:
        logging.info("Loading data from Github: %s/%s", loader._owner, loader._repo) 

        docs = loader.load_data(branch="main")
        for doc in docs:
            logging.info(doc.extra_info)
            doc.metadata = {'filename': doc.extra_info['file_name'], 'author': "LlamaIndex"}

    except Exception as e:
        logging.error("Error loading data from Github: %s", e)
        return None
    
    try:
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
        
        logging.info("Persisting index")
        index.storage_context.persist(persist_dir="./storage")
        
        logging.info("Data-Knowledge ingestion completed (OK)")
    except Exception as e:
        logging.error("Error indexing or persisting data: %s", e)
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    index_knowledge()