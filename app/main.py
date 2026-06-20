import os
import uuid
import shutil
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

app = FastAPI(title="RAG QA API", description="문서를 업로드하고 AI에게 질문하세요", version="1.1.0")

UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path("/tmp/rag_chroma")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

SUPPORTED_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
}

@app.get("/")
def root():
    return {"status": "running", "message": "RAG QA API - LangChain + Claude", "supported_formats": list(SUPPORTED_TYPES.values())}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_TYPES and not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail=f"지원 형식: PDF, TXT, MD")

    ext = SUPPORTED_TYPES.get(content_type) or Path(file.filename).suffix
    doc_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{doc_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")

    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    persist_dir = str(CHROMA_DIR / doc_id)
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

    return {"doc_id": doc_id, "filename": file.filename, "format": ext, "chunks": len(chunks), "message": "문서가 업로드되었습니다. doc_id로 질문할 수 있습니다."}

class QuestionRequest(BaseModel):
    doc_id: str
    question: str
    language: Literal["ko", "en"] = "ko"

@app.post("/ask")
def ask_question(req: QuestionRequest):
    persist_dir = str(CHROMA_DIR / req.doc_id)
    if not Path(persist_dir).exists():
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다. 먼저 /upload로 문서를 업로드해주세요.")

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    relevant_docs = vectorstore.similarity_search(req.question, k=3)
    context = "\n\n".join([d.page_content for d in relevant_docs])

    lang = "한국어" if req.language == "ko" else "English"
    prompt = f"""다음 문서 내용을 참고하여 질문에 {lang}로 답변해주세요.

[문서 내용]
{context}

[질문]
{req.question}

문서에 없는 내용이면 솔직히 모른다고 답변해주세요."""

    response = llm.invoke(prompt)
    return {"doc_id": req.doc_id, "question": req.question, "answer": response.content, "sources": len(relevant_docs)}

@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    persist_dir = CHROMA_DIR / doc_id
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        return {"message": "문서가 삭제되었습니다"}
    raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
