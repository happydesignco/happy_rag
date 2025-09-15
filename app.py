# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from chains.conversational_chain import build_chain
from chains.classify_intent import classify_intent
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain, memory = build_chain()

class QueryInput(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: QueryInput):
    user_input = query.query.strip()

    intent = classify_intent(user_input)

    if intent == "question":
        result = qa_chain.invoke({"question": user_input})
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }

    # fallback response
    history = memory.chat_memory.messages[-8:]
    return {
        "answer": "Sorry, I wasn't able to find a good answer for that.",
        "sources": []
    }

@app.get("/")
async def root():
    return {"message": "Happy RAG is running."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)