from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Custom helpers
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Load .env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Step 1: Load and process PDF data
extracted_data = load_pdf_file(data='data/')     # Your custom PDF loader
texts = text_split(extracted_data)               # Chunking the text
embeddings = download_hugging_face_embeddings()  # Embedding model

# Step 2: Initialize Pinecone client (v3+)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "testbot"

# Step 3: Create index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Depends on your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Step 4: Upload vectors to Pinecone using LangChain wrapper
docsearch = LangchainPinecone.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY
)

print("✅ Documents successfully embedded and uploaded to Pinecone.")

