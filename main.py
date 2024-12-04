import os
import time
import asyncio
import json
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from rag_2_local_pg_dk import DanishRAGSystem  # Import the RAG system

# Initialize the FastAPI app
app = FastAPI(title="Danish RAG System API")

# Initialize the RAG system
rag = DanishRAGSystem(
    embeddings_url=os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000"),
    llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8001"),
    api_key=os.getenv("LOCAL_API_KEY", "dummy-key")
)

# Data Models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5


@app.post("/v1/load_document")
async def load_document(file: UploadFile = File(...)):
    """Endpoint to load a document into the RAG system."""
    try:
        # Save uploaded file temporarily
        temp_path = f"{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        
        # Load the document into the RAG system
        rag.load_document(temp_path)
        os.remove(temp_path)  # Clean up after loading
        
        return {"message": f"Document {file.filename} loaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/query")
async def query_rag(request: QueryRequest):
    """Endpoint to query the RAG system."""
    try:
        response = rag.query(request.question, k=request.k)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print(request)
    """Endpoint for chat completions."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Combine all chat history
    conversation_context = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in request.messages])

    # Extract the latest user query
    question = request.messages[-1].content

    # Construct the query with all chat history
    try:
        response = rag.query(f"{conversation_context}\n\nUser: {question}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {e}")

    # Respond in the OpenAI-compatible format
    if request.stream:
        return StreamingResponse(
            _resp_async_generator(response),
            media_type="application/x-ndjson"
        )
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": response}}],
    }


async def _resp_async_generator(text_resp: str):
    """Simulate token streaming for streaming responses."""
    tokens = text_resp.split(" ")
    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": "/test",
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
