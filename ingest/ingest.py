import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load .env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY") or None  # Optional

collection_name = "rag-docs"

# Load files from /data folder
print("🔍 Loading documents...")
loader = DirectoryLoader("./data", glob="**/*", loader_cls=UnstructuredFileLoader)
documents = loader.load()

# Split documents into chunks
print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Create OpenAI Embeddings
print("🧠 Creating embeddings...")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Connect to Qdrant (will create collection if needed)
print("📦 Uploading to Qdrant...")
client = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key
)

# Create collection if not exists
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Upload chunks
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="https://807ff1fe-30da-4c6c-b5f9-877579692dc2.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name=collection_name,
)

print(f"✅ Ingested {len(chunks)} chunks into Qdrant.")