from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

llm = OllamaLLM(model="llama3.2")

template = """You are a helpful assistant that answers questions about Catalonia's emergency plans (PLASEQCAT, INFOCAT, INUNCAT, etc.), based only on the provided excerpts.

    Here are relevant excerpts from the emergency plan document(s):
    {excerpts}

    Question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

while True:

    question = input("Ask a question about the emergency plan (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    documents = retriever.invoke(question)

    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in documents
    )

    answer = chain.invoke({"question": question, "excerpts": context})

    print(f"Answer: {answer}")