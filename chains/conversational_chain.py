# chains/conversational_chain.py

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME, MODEL_NAME, TEMPERATURE

def load_vectorstore():
    client = QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

def build_chain():
    retriever = load_vectorstore().as_retriever()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    ), memory