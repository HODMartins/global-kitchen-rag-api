import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_FILE_PATH = "global_kitchen_restaurant.txt"
PERSIST_DIR = "chroma_db"


openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is missing. "
        "Set it before running index.py."
    )


# Read the text file
text = Path(DATA_FILE_PATH).read_text(encoding="utf-8")

raw_documents = [
    Document(
        page_content=text,
        metadata={"source": DATA_FILE_PATH}
    )
]


# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

documents = text_splitter.split_documents(raw_documents)

print(f"Split into {len(documents)} chunks")


# Create the embedding model
embeddings = OpenAIEmbeddings(
    api_key=openai_api_key
)


# Build and save the Chroma vector database
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

print(f"Chroma index built and saved to {PERSIST_DIR}/")