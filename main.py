from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import ask

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    answer, pages = ask(request.question)
    return {"answer": answer, "pages": pages}