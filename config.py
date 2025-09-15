# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Shared configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag-docs")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))