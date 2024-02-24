import os
from typing import Union
from fastapi import FastAPI , Query
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client.models import VectorParams, Distance
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
load_dotenv()

origins = [
    "http://localhost:3000",  # Allow frontend to connect from this origin
    # Add more origins if needed
]
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create a Qdrant client
qdrant_client = QdrantClient(
    url= os.environ.get("QDRANT_DB_URL"), 
    api_key=os.environ.get("QDRANT_API_KEY"),
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/create_collection")
def create_collection():
    qdrant_client.recreate_collection(
        collection_name="my_documents",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# TODO: Add a file upload endpoint
class UploadFile(BaseModel):
    file_url:str

@app.post("/upload")
def upload_file(file: UploadFile):
    pdf_loader = PyPDFLoader(file.file_url)
    # print(file.file_url)
    documents = pdf_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAIAPI_KEY"))
    
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        url=os.environ.get("QDRANT_DB_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        collection_name="my_documents"
    )

    return {"message": "Document Upserted to VectorDB"}



@app.get("/query")
def search_query(query:str = Query(...)):
    # search for the query in the Qdrant
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAIAPI_KEY"))
    query_embeddings = embeddings.embed_query(query)
    res = qdrant_client.search(
        collection_name="my_documents",
        query_vector=query_embeddings,
        limit=1
    )

    query_params = "The query is: " + query
    return {"query": query_params , "res": res}