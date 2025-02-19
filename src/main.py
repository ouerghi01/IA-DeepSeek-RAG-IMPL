from tools import *


db =None
tools = None
toolkit = None
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
    global session, Retrieval, qa, templates,db,toolkit
    session,db=await run_in_threadpool(initialize_database_session)
    toolkit = CassandraDatabaseToolkit(db=db)
    tools = toolkit.get_tools()
    
    
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
    print(answer)
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
    uvicorn.run(app, host="0.0.0.0", port=8005)
# https://github.com/bhattbhavesh91/pdf-qa-astradb-langchain/blob/main/requirements.txt
# https://github.com/michelderu/chat-with-your-data-in-cassandra/blob/main/docker-compose.yml