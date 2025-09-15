# query.py

from chains.conversational_chain import build_chain
from chains.classify_intent import classify_intent
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def convert_history_to_messages(history):
    messages = []
    for turn in history:
        messages.append(HumanMessage(content=turn["user"]))
        if "ai" in turn:
            messages.append(AIMessage(content=turn["ai"]))
    return messages

def get_last_user_question(memory):
    for m in reversed(memory.chat_memory.messages):
        if m.type == "human":
            return m.content
    return "I don’t have your previous question."

# Use a general-purpose chat LLM for fallback responses
fallback_llm = ChatOpenAI(model="gpt-4o")

# LLM chain to generate contextual fallback responses
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond conversationally."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])
fallback_chain = fallback_prompt | fallback_llm

def format_history_for_prompt(messages):
    formatted = []
    for msg in messages[-8:]:  # Only keep last 8 turns
        if msg.type == "human":
            formatted.append(f"User: {msg.content}")
        elif msg.type == "ai":
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)

def main():
    qa_chain, memory = build_chain()

    print("🤖 Ask me anything about your docs. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # 1. Classify user intent
        intent = classify_intent(query)

        # 2. If it's a doc-related question, use the RAG chain
        if intent == "question":
            result = qa_chain.invoke({"question": query})
            print("\nAI:", result["answer"])
            print("\n📚 Sources:")
            for doc in result["source_documents"]:
                print("—", doc.metadata.get("source", "Unknown"))
            print("\n---")
            continue

        # 3. Otherwise, use the LLM to generate a contextual response
        messages = memory.chat_memory.messages[-8:]  # last few messages
        response = fallback_chain.invoke({
            "history": messages,
            "input": query
        })

        print("\nAI:", response.content)
        print("\n📚 Sources: (none — not a document-related question)")
        print("\n---")

if __name__ == "__main__":
    main()