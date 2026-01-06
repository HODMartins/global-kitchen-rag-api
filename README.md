# global-kitchen-rag-api
AI agent using RAG for a restaurant. 

# Global Kitchen RAG API

A retrieval augmented generation (RAG) API that answers restaurant questions using grounded knowledge from menu and operational documents.

## Features
- Answers questions about menu items, opening hours, dietary options, and service information
- Retrieves relevant context from documents before generating responses
- Designed with guardrails to reduce hallucinations
- Easy local run and deployment friendly

## Tech Stack
- Python
- FastAPI (or Flask) + Uvicorn/Gunicorn
- Vector search (your chosen store)
- OpenAI API

## Endpoints
- GET /health
  - Returns service health status
- POST /ask
  - Accepts a JSON payload with a user question and returns an answer

### Example Request
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are your opening hours?\"}"
