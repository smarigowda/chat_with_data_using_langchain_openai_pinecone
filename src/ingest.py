import os
import pinecone
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
chunk_size = os.environ.get("CHUNK_SIZE")
chunk_overlap = os.environ.get("CHUNK_OVERLAP")


def main():
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    # loader = DirectoryLoader("./docs", glob="**/*.pdf")
    # loader = PyPDFLoader("./docs/pdfs/us_reports_morse_v_frederick.pdf")
    loader = UnstructuredPDFLoader("./docs/pdfs/test.pdf")

    docs = loader.load()
    # pages = loader.load_and_split()
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=["\n", "\r\n", "\n\n", " "],
    # )
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents=docs)

    # texts = text_splitter.split_documents(documents=docs)
    pass


if __name__ == "__main__":
    main()
