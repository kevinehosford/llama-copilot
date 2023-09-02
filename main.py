import pickle
import os

from llama_index import download_loader, GPTVectorStoreIndex, StorageContext, load_index_from_storage
download_loader("GithubRepositoryReader")

from llama_hub.github_repo import GithubClient, GithubRepositoryReader

docs_file = "docs.pkl"
index_dir = "index_persist"
owner = "<owner>"
repo = "<repo>"

index = None

if os.path.exists(index_dir):
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

if index is None:
    docs = None
    if os.path.exists(docs_file):
        with open(docs_file, "rb") as f:
            docs = pickle.load(f)

    if docs is None:
        github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
        loader = GithubRepositoryReader(
            github_client,
            owner =                  owner,
            repo =                   repo,
            filter_directories =     (["src"], GithubRepositoryReader.FilterType.INCLUDE),
            filter_file_extensions = ([".tsx"], GithubRepositoryReader.FilterType.INCLUDE),
            verbose =                True,
            concurrent_requests =    10,
        )

        docs = loader.load_data(branch="main")

        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

    index = GPTVectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=index_dir)

query_engine = index.as_query_engine()

query = "Which is the saddest file. Explain why it is a sad file."

print(query)

response = query_engine.query(query)

print(response)