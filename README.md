# RAG Q&A

PDF 문서를 업로드하면 **LangChain + ChromaDB**로 벡터 저장하고, **Claude API**가 문서 내용을 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템

---

## 프로젝트 개요

RAG는 대형 언어 모델에 외부 문서 검색 능력을 결합하는 기법입니다. PDF를 청크로 분할 → 벡터 변환 → ChromaDB 저장 → 질문 시 유사 청크 검색 → Claude가 검색된 내용을 바탕으로 답변하는 파이프라인을 구현합니다. Claude가 학습하지 않은 내용도 문서만 있으면 답변 가능합니다.

---

## 아키텍처

```
[ 문서 업로드 단계 ]
PDF 파일 업로드
        ↓
PyPDF - 텍스트 추출
        ↓
LangChain RecursiveCharacterTextSplitter
500자 단위 청크 분할 (50자 오버랩)
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
텍스트 → 벡터 변환
        ↓
ChromaDB - 인메모리 벡터 저장

[ 질문 응답 단계 ]
사용자 질문 입력
        ↓
질문 벡터화 → ChromaDB 유사도 검색 (Top 3)
        ↓
검색된 청크 + 질문 → Claude API
        ↓
문서 기반 한국어 답변 반환
```

---

## 사용 기술 스택

| 기술 | 역할 |
|------|------|
| **LangChain** | RAG 파이프라인 오케스트레이션 |
| **ChromaDB** | 인메모리 벡터 데이터베이스 |
| **HuggingFace Embeddings** | all-MiniLM-L6-v2 텍스트 벡터화 |
| **Claude API** (claude-sonnet-4-6) | 최종 답변 생성 |
| **PyPDF** | PDF 텍스트 추출 |
| **FastAPI** | REST API 서버 |

---

## 사용 순서

### 1단계: PDF 업로드

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**응답:**

```json
{
  "filename": "document.pdf",
  "pages": 24,
  "chunks": 87,
  "message": "문서가 업로드되었습니다. 이제 질문할 수 있어요!"
}
```

### 2단계: 질문

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "이 문서에서 핵심 결론은 무엇인가요?"}'
```

**응답:**

```json
{
  "question": "이 문서에서 핵심 결론은 무엇인가요?",
  "answer": "문서에 따르면 핵심 결론은 ..."
}
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 서버 상태 확인 |
| `POST` | `/upload` | PDF 업로드 + 벡터 DB 저장 |
| `POST` | `/ask` | 문서 기반 질문 답변 |
| `GET` | `/docs` | Swagger UI |

---

## 주의사항

- 현재 벡터 저장소는 **인메모리** 방식으로 서버 재시작 시 초기화됨
- PDF만 지원 (txt, docx 등은 추후 확장 가능)

---

## 실행 방법

```bash
cp .env.example .env
pip install -r requirements.txt
cd app && uvicorn main:app --host 0.0.0.0 --port 8008
```

## 환경 변수

| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API 키 |
| `HF_TOKEN` | HuggingFace API 토큰 (임베딩 모델용) |
