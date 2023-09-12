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


def index_knowledge() -> VectorStoreIndex:
    """Load and Index Knowledge Base"""

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise Exception("OPENAI_API_KEY environment variable not set.")

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

    docs = loader.load_data(branch="main")

    for doc in docs:
        print(doc.extra_info)

    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="You are a specialized AI trained in the usage of LlamaIndex.",
        )
    )

    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir="./vectorstorage")

if __name__ == "__main__":
    index_knowledge()