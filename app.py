from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import fallback_chain, classify_intent
from chains.conversational_chain import build_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain, memory = build_chain()

class ChatRequest(BaseModel):
    history: list[dict]
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.question
    history = request.history

    # Update memory
    for turn in history:
        if "user" in turn:
            memory.chat_memory.add_user_message(turn["user"])
        if "ai" in turn:
            memory.chat_memory.add_ai_message(turn["ai"])
    memory.chat_memory.add_user_message(user_input)

    # Classify user intent
    intent = classify_intent(user_input)
    print(f"Intent: {intent}")

    try:
        if intent == "question":
            # Use the RAG pipeline
            result = qa_chain.invoke({"question": user_input})
            memory.chat_memory.add_ai_message(result["answer"])
            return {
                "answer": result["answer"],
                "sources": [
                    d.metadata.get("source", "Unknown") for d in result.get("source_documents", [])
                ]
            }

        elif intent == "meta":
            # Use fallback LLM chain to answer meta/conversational questions
            recent_messages = memory.chat_memory.messages[-8:]
            response = fallback_chain.invoke({
                "history": recent_messages,
                "input": user_input
            })
            memory.chat_memory.add_ai_message(response.content)
            return {
                "answer": response.content,
                "sources": []
            }

        else:  # non-question or small talk
            return {
                "answer": "Got it!",
                "sources": []
            }

    except Exception as e:
        print("⚠️ Error in chat endpoint:", e)
        return {
            "answer": "⚠️ Something went wrong. Please try again.",
            "sources": []
        }