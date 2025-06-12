from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import process_and_store_pdf, retrieve_top_chunks, ask_with_gemini
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_origins=[
        "https://questionyourpdf.lovable.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    process_and_store_pdf(file_location)
    return {"status": "success", "filename": file.filename}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    context = retrieve_top_chunks(question)
    answer = ask_with_gemini(context, question)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "Server is up!"}
