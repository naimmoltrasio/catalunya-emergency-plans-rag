from rag import ask

while True:

    question = input("Ask a question about the emergency plan (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    answer = ask(question)

    print(f"Answer: {answer}")
