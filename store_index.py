from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")

#print(PINECONE_API_KEY)
extracted_data=load_pdf_file(data='C:/Users/dheer/Chatbot-/data/')
texts=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()




# Initialize Pinecone client
pc = Pinecone(api_key="pcsk_4UboeU_JB8iqzdQjRuW8aDhH8CSRTaPkjcnLqbHZypRLx3mLoF46QUkqFUMgTqAtACxxxA")

index_name = "testbot"

# ✅ Check if index already exists
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    



docsearch=PineconeVectorStore.from_documents(
    documents=texts,
    index_name=index_name,
    embedding=embeddings,
)


