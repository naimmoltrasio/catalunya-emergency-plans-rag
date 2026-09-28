from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask

app = FastAPI(title="Catalunya Emergency Plans RAG")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask_question(payload: Question) -> Answer:
    return Answer(answer=ask(payload.question))
