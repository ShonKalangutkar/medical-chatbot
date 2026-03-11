from dotenv import load_dotenv
import os

from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Load and process documents
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)


# Load embeddings
embeddings = download_embeddings()


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # required for MiniLM embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


# Connect to index
index = pc.Index(index_name)


# Store documents in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Index created and documents stored successfully.")