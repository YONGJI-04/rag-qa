import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import RetrievalQA
import tempfile

load_dotenv()

app = FastAPI(title="RAG Q&A API")

llm = ChatAnthropic(model="claude-sonnet-4-6", anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "running", "message": "RAG Q&A API"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(chunks, embeddings)

    return JSONResponse(content={
        "filename": file.filename,
        "pages": len(docs),
        "chunks": len(chunks),
        "message": "문서가 업로드되었습니다. 이제 질문할 수 있어요!"
    })


@app.post("/ask")
def ask_question(req: QuestionRequest):
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="먼저 PDF를 업로드해주세요")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )

    result = qa_chain.invoke({"query": req.question + "\n한국어로 답변해주세요."})

    return JSONResponse(content={
        "question": req.question,
        "answer": result["result"],
    })
