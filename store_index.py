from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os 

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")

print(PINECONE_API_KEY)