# chains/classify_intent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create an LLM instance (you can switch to gpt-3.5-turbo if preferred)
intent_llm = ChatOpenAI(model="gpt‑3.5‑turbo", temperature=0)

# Define the prompt for classifying intent
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that classifies user input into one of the following types:\n"
        "- 'question': factual or knowledge-based query, usually answered with info from documents or general knowledge\n"
        "- 'meta': any message about the conversation itself, such as 'what did I just say?', 'what's my name?', 'what did you mean earlier?', or requests about the assistant’s behavior\n"
        "- 'non-question': comments or other input that doesn’t ask a question\n"
        "Respond with just the category name: 'question', 'meta', or 'non-question'."
    )),
    ("human", "{input}")
])

# Chain together the prompt and model
intent_chain = intent_prompt | intent_llm

# Function to call from other files
def classify_intent(query: str) -> str:
    return intent_chain.invoke({"input": query}).content.strip().lower()