import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response

app = FastAPI()

allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://hodmartins.com",
    "https://www.hodmartins.com",
    "https://www.myportfolio.hodmartins.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/ask")
def ask_preflight():
    return Response(status_code=204)

class AskRequest(BaseModel):
    question: str

qa_chain = None
startup_error = None

@app.on_event("startup")
def startup():
    global qa_chain, startup_error
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable is missing")

        embeddings = OpenAIEmbeddings(api_key=key)

        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=key
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        print("RAG API ready")

    except Exception as e:
        startup_error = str(e)
        print(f"Startup failed: {startup_error}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "qa_chain_ready": qa_chain is not None,
        "startup_error": startup_error
    }

@app.post("/ask")
def ask(payload: AskRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail=startup_error)
    return {"answer": qa_chain.run(payload.question)}
