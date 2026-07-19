import os
import time
import logging
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from contextlib import asynccontextmanager



logger = logging.getLogger("uvicorn.error")

def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class AskRequest(BaseModel):
    question: str


# Globals initialized at startup
retriever = None
llm = None
prompt = None
startup_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm, prompt, startup_error
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is missing"
            )

        embeddings = OpenAIEmbeddings(api_key=key)

        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.5,
            api_key=key,
            streaming=True
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a warm, friendly, and professional restaurant "
                "assistant for Global Kitchen Restaurant. "
                "Answer in 2 to 5 short sentences. Be polite and helpful. "
                "Use ONLY the provided context. "
                "Do not guess or invent menu items, prices, opening hours, "
                "location, or ordering details. "
                "If the context does not contain the answer, say clearly "
                "that the information is not available."
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            )
        ])

        logger.info("RAG API ready")

    except Exception as e:
        startup_error = str(e)
        logger.exception(f"Startup failed: {startup_error}")
        raise

    yield


app = FastAPI(lifespan=lifespan)

allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://hodmartins.com",
    "https://www.hodmartins.com",
    "https://myportfolio.hodmartins.com",
    "https://www.myportfolio.hodmartins.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    stream_ready = (
        retriever is not None
        and llm is not None
        and prompt is not None
    )

    return {
        "status": "ok" if stream_ready else "error",
        "stream_ready": stream_ready,
        "startup_error": startup_error
    }


@app.post("/ask-stream")
async def ask_stream(payload: AskRequest):
    if retriever is None or llm is None or prompt is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "RAG service is not ready."
        )

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    request_id = str(int(time.time() * 1000))
    request_start = time.perf_counter()

    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    }

    greeting_starts = (
        "hi ",
        "hello ",
        "hey ",
        "good morning ",
        "good afternoon ",
        "good evening "
    )

    def sse_message(text: str) -> EventSourceResponse:
        async def generator():
            yield {"data": text}
            yield {"data": "[DONE]"}

        return EventSourceResponse(generator())

    question_lower = question.lower()

    # Handle greetings without calling the retriever or LLM
    if (
        question_lower in greetings
        or question_lower.startswith(greeting_starts)
    ):
        return sse_message(
            "Hello! Welcome to Global Kitchen Restaurant. "
            "How can I help you today?"
        )

    # Retrieve relevant restaurant documents asynchronously
    retrieval_start = time.perf_counter()
    docs = await retriever.ainvoke(question)
    retrieval_ms = ms(retrieval_start)

    logger.info(
        f"RAG_RETRIEVE request_id={request_id} "
        f"docs_count={len(docs)} "
        f"question={question!r}"
    )

    if not docs:
        return sse_message(
            "I could not find relevant restaurant information for that "
            "question. Please ask about our menu, prices, opening hours, "
            "location, or ordering."
        )

    # Combine retrieved document text into one context string
    context = "\n\n".join(
        doc.page_content.strip()
        for doc in docs
        if getattr(doc, "page_content", "").strip()
    )

    if not context:
        return sse_message(
            "I could not find usable restaurant information for that "
            "question. Please ask about our menu, prices, opening hours, "
            "location, or ordering."
        )

    prompt_start = time.perf_counter()

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    prompt_ms = ms(prompt_start)

    async def event_generator():
        first_token_ms = None
        generation_start = time.perf_counter()
        chunk_count = 0

        try:
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", "") or ""

                if token:
                    if first_token_ms is None:
                        first_token_ms = ms(request_start)

                    chunk_count += 1
                    yield {"data": token}
                    await asyncio.sleep(0)

            generation_ms = ms(generation_start)
            total_ms = ms(request_start)

            logger.info(
                f"RAG_TIMING request_id={request_id} "
                f"retrieval_ms={retrieval_ms:.0f} "
                f"prompt_ms={prompt_ms:.0f} "
                f"ttft_ms="
                f"{first_token_ms if first_token_ms is not None else -1:.0f} "
                f"generation_ms={generation_ms:.0f} "
                f"total_ms={total_ms:.0f} "
                f"context_chars={len(context)} "
                f"docs_count={len(docs)} "
                f"chunks={chunk_count}"
            )

            yield {"data": "[DONE]"}

        except Exception as error:
            logger.exception(
                f"RAG_ERROR request_id={request_id} error={error}"
            )

            yield {
                "data": (
                    "Sorry, something went wrong while generating "
                    "the response."
                )
            }
            yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())