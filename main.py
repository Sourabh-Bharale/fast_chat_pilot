import os
from typing import Union
from fastapi import FastAPI , Query
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# create a Qdrant client
qdrant_client = QdrantClient(
    url= os.environ.get("QDRANT_DB_URL"), 
    api_key=os.environ.get("QDRANT_API_KEY"),
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


# TODO: Add a file upload endpoint
# @app.post("/upload")
# def upload_file(url:str = Query(...)):



@app.get("/query")
def search_query(query:str = Query(...)):
    # search for the query in the Qdrant
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    query_embeddings = embeddings.embed_query(query)
    res = qdrant_client.search(
        collection_name="my_documents",
        query_vector=query_embeddings,
        limit=5
    )

    query_params = "The query is: " + query
    return {"query": query_params , "res": res}