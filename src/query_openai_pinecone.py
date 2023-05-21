import os
import glob
import pinecone
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
index_name = os.environ.get("PINECONE_INDEX_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")


# main function
def main():
    # initialize pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # load an existing index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.9), docsearch.as_retriever(), memory=memory
    )
    query = "What is this legal case about ?"
    result = qa({"question": query})
    print(result)
    pass


if __name__ == "__main__":
    main()
