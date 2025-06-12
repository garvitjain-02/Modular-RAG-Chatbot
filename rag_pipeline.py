import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "db"

# Local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
# Setup Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def embed_text(texts):
    return embedding_model.encode(texts).tolist()

def process_and_store_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    from langchain.docstore.document import Document
    chunk_texts = [c.page_content for c in chunks]
    embeddings = embed_text(chunk_texts)

    documents = [Document(page_content=txt) for txt in chunk_texts]
    vectordb = Chroma.from_texts(
        texts=chunk_texts,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        ids=[f"doc-{i}" for i in range(len(chunk_texts))]
    )
    vectordb.persist()

def retrieve_top_chunks(question, k=5):
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=lambda x: embed_text(x)
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join([doc.page_content for doc in docs])

def ask_with_gemini(context, question):
    prompt = f"""
You are a helpful assistant. Based on the context below, answer the user's question.

Context:
{context}

Question:
{question}

Answer:
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
