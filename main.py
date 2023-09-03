import pickle
import os
import argparse

from llama_index import download_loader, GPTVectorStoreIndex, StorageContext, load_index_from_storage
download_loader("GithubRepositoryReader")

from llama_hub.github_repo import GithubClient, GithubRepositoryReader

parser = argparse.ArgumentParser(description="Description of your program")

parser.add_argument("-o", "--owner", type=str, help="Owner of the repository")
parser.add_argument("-r", "--repo", type=str, help="Name of the repository")
parser.add_argument("-b", "--branch", type=str, help="Name of the branch", default="main")
parser.add_argument("-q", "--query", type=str, help="Repo query", default="What's the saddest file in the repo?")

args = parser.parse_args()

owner = args.owner
repo = args.repo
branch = args.branch
query = args.query

print(f"Owner: {owner}")
print(f"Repo: {repo}")
print(f"Branch: {branch}")
print(f"Query: {query}")

docs_file = "docs.pkl"
index_dir = "index_persist"

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
        # raise error if owner and repo are none
        if owner is None or repo is None:
            raise ValueError("Owner and repo cannot be None")

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

        docs = loader.load_data(branch)

        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

    index = GPTVectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=index_dir)

query_engine = index.as_query_engine()

print(query)

response = query_engine.query(query)

print(response)