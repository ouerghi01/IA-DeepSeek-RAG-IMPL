from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json

import uuid
from datetime import datetime
import uvicorn
from fastapi.concurrency import run_in_threadpool
from typing import Annotated
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_experimental.text_splitter import SemanticChunker  
from fastapi import FastAPI 
from fastapi import FastAPI,Request,UploadFile,File,Form

from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import uuid
from datetime import datetime

def initialize_database_session():
    cloud_config = {
    'secure_connect_bundle': 'secure-connect-store-base.zip'
    }

    with open("token.json") as f:
       secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    
    create_table = """
    CREATE TABLE IF NOT EXISTS store_key.vectores (
        partition_id UUID PRIMARY KEY,
        document_text TEXT,  -- Text extracted from the PDF
        document_content BLOB,  -- PDF content stored as binary data
        vector BLOB  -- Store the embeddings (vector representation of the document)
    );
    """
    response_table = """
    CREATE TABLE IF NOT EXISTS store_key.response_table (
        partition_id UUID PRIMARY KEY,
        question TEXT,  
        answer TEXT,
        timestamp TIMESTAMP,
        evaluation BOOLEAN
        
    );
    """
    session.execute(create_table)
    session.execute(response_table)
    return session

def load_pdf_documents(path_file):
    loader = PDFPlumberLoader(path_file)
    docs = loader.load()
    return docs
def create_retriever_from_documents(session, docs):
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"  
    # intfloat/e5-small is a smaller model that can be used for faster inference
    #model_name = "intfloat/e5-small"
    model_kwargs = {'device': 'cpu'}  # Use CPU for inference
    encode_kwargs = {'normalize_embeddings': True}  # Normalizing the embeddings for better comparison
    #HuggingFaceEmbeddings
    hf = HuggingFaceEmbeddings(
        model_name=model_name,  
        model_kwargs=model_kwargs,  
        encode_kwargs=encode_kwargs
    )
    text_splitter = SemanticChunker (
    hf 
    )
    documents = text_splitter.split_documents(docs)
    keyspace = "store_key"
    table="vectores"
    cassandra_store :Cassandra = Cassandra.from_documents(documents=documents, embedding=hf, session=session, keyspace=keyspace, table_name=table)
    retrieval=cassandra_store.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.5})
    return retrieval
def get_last_responses(session):
    query = "SELECT * FROM store_key.response_table LIMIT 10"
    rows = session.execute(query)
    text=""
    for row in rows:
        text+=f"Question: {row.question}\nAnswer: {row.answer}\nTimestamp: {row.timestamp}\nEvaluation: {row.evaluation}\n\n"
    return text
def build_q_a_process(retrieval,model_name="deepseek-r1:1.5b"):
    global session

    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    history_responses_str= get_last_responses(session)
    pp = f"""
    1. Utilise uniquement le contexte fourni ci-dessous pour formuler ta réponse.
    2. Si l'information demandée n'est pas présente dans le contexte, réponds par "Je ne sais pas".
    3. Fournis des réponses concises, ne dépassant pas trois phrases.
    4. Si le contexte mentionne un outil ou une fonctionnalité spécifique d'un site web ou d'une plateforme SaaS, explique son utilisation et son objectif.
    5. Ne rajoute aucune information qui ne soit pas incluse dans le contexte fourni.
    6. Si possible, donne une recommandation pertinente.
    
    """
    prompt = pp + """
    Contexte: {context}
    Question: {question}
    Réponse:
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )

    document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",  
        input_variables=["page_content", "source"]  
    )
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,  
        document_variable_name="context",
        callbacks=None,
        document_prompt=document_prompt  
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,  
        retriever=retrieval,
        verbose=True
    )
    return qa

from pathlib import Path
app = FastAPI()
model_name = "qwen2:0.5b"
UPLOAD_DIR = Path("uploads")  
UPLOAD_DIR.mkdir(exist_ok=True)  
session = None
default_path_file = Path("uploads/data.pdf")
Retrieval = None
qa = None

templates = None

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    global session, Retrieval, qa, templates
    session=await run_in_threadpool(initialize_database_session)
    templates = Jinja2Templates(directory="templates")
    if default_path_file.exists():
        Retrieval = create_retriever_from_documents(session, load_pdf_documents(default_path_file))
        qa = await run_in_threadpool(build_q_a_process,Retrieval, model_name)

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down")
    global session
    if session:
        session.shutdown()
        session = None

@app.get("/")
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def upload_file(file: Annotated[UploadFile, File(description="A file read as UploadFile")]):
    global default_path_file, Retrieval, qa
    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        return {"message": file.filename + " already exists"}
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    default_path_file = file_path
    Retrieval = create_retriever_from_documents(session, load_pdf_documents(default_path_file))
    qa = build_q_a_process(Retrieval, model_name)
    
    return {"message": file.filename + " has been uploaded"}

@app.post("/model_name/") 
async def set_model_name(request: Request, model: str = Form(...)):
    global model_name , qa
    
    model_name = model
    if qa is not None:
        qa = build_q_a_process(Retrieval, model_name)
    return {"message": f"Model name has been set to {model_name}"}
@app.get("/evaluate_response/")
async def evaluate_response(request: Request, partition_id: str, evaluation: bool):
    query=f"UPDATE store_key.response_table SET evaluation={evaluation} WHERE partition_id={partition_id}"
    session.execute(query)
    #return templates.TemplateResponse("index.html", {"request": request})
@app.post("/send_message/")
async def send_message(request: Request, question: str = Form(...)):
    global qa
    if not default_path_file.exists() or qa is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No document uploaded"}
        )
    
    answer = await run_in_threadpool(qa.run, question)
    if session is not None:
        partition_id = uuid.uuid1()
        now=datetime.now()
        session.execute(
        "INSERT INTO store_key.response_table (partition_id, question, answer,timestamp,evaluation) VALUES (%s, %s, %s, %s,false)",
        (partition_id,question, answer,now)
        )
    

    return templates.TemplateResponse(
        "messages.html",
        {"request": request, "answer": answer, "question": question,"partition_id":partition_id,"timestamp":now,"evaluation":False}
    )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# https://github.com/bhattbhavesh91/pdf-qa-astradb-langchain/blob/main/requirements.txt
# https://github.com/michelderu/chat-with-your-data-in-cassandra/blob/main/docker-compose.yml