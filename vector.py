from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

vector_store = Chroma(
    collection_name="emergency_plans_catalunya",
    persist_directory=db_location,
    embedding_function=embeddings
)

add_documents = vector_store._collection.count() == 0

if add_documents:
    # Carga todos los PDFs de la carpeta "assets"
    loader = PyPDFDirectoryLoader("assets/")
    raw_documents = loader.load()  # cada página del PDF = un Document

    # Chunking: divide el texto en fragmentos manejables con overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    ids = [str(i) for i in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)