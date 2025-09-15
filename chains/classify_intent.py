# chains/classify_intent.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create an LLM instance (you can switch to gpt-3.5-turbo if preferred)
intent_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the prompt for classifying intent
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that classifies user input into one of the following types:\n"
        "- 'question' if the user is asking something they want an answer to\n"
        "- 'meta' if the user is talking about the conversation, their previous message, or how the AI should behave\n"
        "- 'non-question' if it's just a comment or something that doesn't require an answer\n"
        "Respond with just the category name: 'question', 'meta', or 'non-question'."
    )),
    ("human", "{input}")
])

# Chain together the prompt and model
intent_chain = intent_prompt | intent_llm

# Function to call from other files
def classify_intent(query: str) -> str:
    return intent_chain.invoke({"input": query}).content.strip().lower()