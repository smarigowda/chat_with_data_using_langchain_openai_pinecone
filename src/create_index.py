import os
import pinecone

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")


def main():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    pinecone.create_index("pinecone-demo-index", metric="cosine", dimension=1056)
    pass


if __name__ == "__main__":
    main()
