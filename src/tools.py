from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
from langchain.chains import create_sql_query_chain
import uuid
from datetime import datetime
import uvicorn
from fastapi.concurrency import run_in_threadpool
from typing import Annotated
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.tools.cassandra_database.prompt import QUERY_PATH_PROMPT
from langchain_community.utilities.cassandra_database import CassandraDatabase
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_community.agent_toolkits.cassandra_database.toolkit import (
    CassandraDatabaseToolkit,
)

from langchain_core.messages import HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import initialize_agent, Tool, AgentType
from pathlib import Path
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.tools.cassandra_database.prompt import QUERY_PATH_PROMPT
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
    'secure_connect_bundle': '/home/aziz/IA-DeepSeek-RAG-IMPL/src/secure-connect-store-base.zip'
    }

    CLIENT_ID = "QGKzQShKaYWuLtYJZmLQQWvF"
    CLIENT_SECRET =  "j3guOpw5mORCpd-kqldrxZH8qrmm328slgWa1OvQXv1-RJsGNQLzbX3KRxywcLfcXpAWiOR8GwR6FZXQfXtjR+.oqRDXWUtqk9olEOcUlFXr9qNy-bvegRNWomToLYWQ"
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
    db=CassandraDatabase(session=session)
    return session, db

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
def query_cassandra(session,query):
    rows = session.execute(query)
    return [row for row in rows]
def build_q_a_process(retrieval,model_name="deepseek-r1:1.5b"):
    global session

    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    #history_responses_str= get_last_responses(session)
    def fill_text(r,q):
        return f"Question: {q}\nAnswer: {r}\n"
   
    pp = f"""

    🔹 **Présentation de l'agent** 🔹  
    Je suis un agent intelligent conçu pour fournir des réponses précises et pertinentes en me basant uniquement sur les informations du contexte fourni. Mon objectif est d'offrir des explications claires et concises, tout en respectant les limites des données disponibles.  

    📌 **Directives pour formuler les réponses :**  
    1. Utilise uniquement le contexte fourni ci-dessous pour formuler ta réponse.  
    2. Si l'information demandée n'est pas présente dans le contexte, réponds par "Je ne sais pas".  
    3. Fournis des réponses concises, ne dépassant pas trois phrases.  
    4. Si le contexte mentionne un outil ou une fonctionnalité spécifique d'un site web ou d'une plateforme SaaS, explique son utilisation et son objectif.  
    5. N'ajoute aucune information qui ne soit pas incluse dans le contexte fourni.  
    6. Si possible, donne une recommandation pertinente.  
    """

    prompt = pp + """
    Contexte: {context}
    Question: {question}
    Réponse:
    """

    QA_CHAIN_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

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
def get_schema_description(session):
    schema_info = []
    tables_query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name = 'store_key'"
    tables = session.execute(tables_query)

    for table in tables:
        table_name = table.table_name
        columns_query = f"""
        SELECT column_name, type 
        FROM system_schema.columns 
        WHERE keyspace_name = 'store_key' AND table_name = '{table_name}'
        """
        columns = session.execute(columns_query)
        columns_info = ", ".join([f"{col.column_name} ({col.type})" for col in columns])
        schema_info.append(f"Table: {table_name} | Columns: {columns_info}")

    return "\n".join(schema_info)
