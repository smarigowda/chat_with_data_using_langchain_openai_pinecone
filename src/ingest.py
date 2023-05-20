import os
import pinecone
from langchain.document_loaders import DirectoryLoader

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")


def main():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    loader = DirectoryLoader("./docs", glob="**/*.pdf")
    docs = loader.load()
    pass


if __name__ == "__main__":
    main()
