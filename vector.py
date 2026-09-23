from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

BATCH_SIZE = 20

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

vector_store = Chroma(
    collection_name="emergency_plans_catalunya",
    persist_directory=db_location,
    embedding_function=embeddings
)

add_documents = vector_store._collection.count() == 0

def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


if add_documents:
    loader = PyPDFDirectoryLoader("assets/")
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    ids = [str(i) for i in range(len(documents))]

    for doc_batch, id_batch in zip(batch(documents, BATCH_SIZE), batch(ids, BATCH_SIZE)):
        print(f"Indexando lote de {len(doc_batch)} chunks...")
        vector_store.add_documents(documents=doc_batch, ids=id_batch)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)