# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import build_chain, classify_intent, fallback_chain, format_history_for_prompt, convert_history_to_messages
from langchain_core.messages import HumanMessage, AIMessage
import os


app = FastAPI()

# Set up chain and memory
qa_chain, memory = build_chain()

class QueryInput(BaseModel):
    query: str
    history: list[dict] = []

@app.post("/chat")
async def chat(query: QueryInput):
    user_input = query.query.strip()
    chat_history = convert_history_to_messages(query.history)

    # Update memory
    memory.chat_memory.messages = chat_history

    # Classify the user's intent
    intent = classify_intent(user_input)

    if intent == "question":
        result = qa_chain.invoke({"question": user_input})
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }

    # Otherwise, use fallback LLM to respond conversationally
    history_str = format_history_for_prompt(memory.chat_memory.messages)
    response = fallback_chain.invoke({
        "history": history_str,
        "input": user_input
    })

    return {
        "answer": response.content,
        "sources": []  # fallback response has no document sources
    }

@app.get("/")
async def root():
    return {"message": "Happy RAG is running."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)