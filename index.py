import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

DATA_FILE_PATH = "global_kitchen_restaurant.txt"
PERSIST_DIR = "chroma_db"

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY env var is missing. Set it before running index.py.")

# ---- Load and split documents ----
raw_documents = TextLoader(DATA_FILE_PATH, encoding="utf-8").load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
).split_documents(raw_documents)

print(f"Split into {len(documents)} chunks")

# ---- Build and persist Chroma ----
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

vector_store.persist()

print("Chroma index built and saved to chroma_db/")
