import os
import glob
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader_mappings = {
    ".pdf": (PDFMinerLoader, {}),
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {}),
}

load_dotenv()

chunk_size = os.environ.get("CHUNK_SIZE")
chunk_overlap = os.environ.get("CHUNK_OVERLAP")


# load a single document from file_path
def load_sigle_document(file_path: str) -> Document:
    file_extension = "." + file_path.rsplit(".", 1)[-1]
    if file_extension in loader_mappings:
        loader_class, loader_args = loader_mappings[file_extension]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]
    raise Exception(f"Unsupported file extension {file_extension}")


# load all documents from source_dir
def load_all_documents(source_dir: str) -> list[Document]:
    all_files = []
    for ext in loader_mappings:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_sigle_document(file_path) for file_path in all_files]


# main function
def main():
    # load documents and split into chunks
    source_dir = os.getenv("SOURCE_DIR")
    # chunk_size = 500
    # chunk_overlap = 50
    documents = load_all_documents(source_dir)
    print(f"Loaded {len(documents)} documents form {source_dir} directory")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text. Max chunk size is {chunk_size}")
    pass


if __name__ == "__main__":
    main()
