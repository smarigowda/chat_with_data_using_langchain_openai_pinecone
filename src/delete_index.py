import os
import pinecone

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")


def main():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    pinecone.delete_index(pinecone_index_name)
    pass


if __name__ == "__main__":
    main()
