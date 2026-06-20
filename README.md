# RAG Q&A

PDF 문서를 업로드하면 문서 내용을 기반으로 Claude가 질문에 답변하는 RAG 시스템

## 아키텍처

```
PDF 파일 업로드
        ↓
PyPDF - 텍스트 추출
        ↓
LangChain - 청크 분할 (500자 단위)
        ↓
HuggingFace Embeddings - 벡터 변환
        ↓
ChromaDB - 벡터 저장
        ↓
질문 입력 → 유사 청크 검색 → Claude 답변 생성
```

## 사용 기술

| 기술 | 역할 |
|------|------|
| LangChain | RAG 파이프라인 구성 |
| ChromaDB | 벡터 데이터베이스 |
| HuggingFace Embeddings | 텍스트 벡터화 (all-MiniLM-L6-v2) |
| Claude API | 최종 답변 생성 |
| PyPDF | PDF 텍스트 추출 |

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 확인 |
| POST | `/upload` | PDF 업로드 + 벡터 저장 |
| POST | `/ask` | 문서 기반 질문 답변 |
| GET | `/docs` | Swagger UI |

## 사용 순서

```bash
# 1. PDF 업로드
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# 2. 질문
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "이 문서의 핵심 내용은 무엇인가요?"}'
```

## 실행 방법

```bash
cp .env.example .env
pip install -r requirements.txt
cd app && uvicorn main:app --host 0.0.0.0 --port 8008
```

## 환경 변수

```
ANTHROPIC_API_KEY=   # Anthropic Claude API 키
HF_TOKEN=            # HuggingFace API 토큰 (임베딩용)
```
