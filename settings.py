"""프로젝트 공통 설정 — 인덱서(② indexer)와 챗봇·평가(④ app, rag, eval)가 함께 쓰는 값.

문서 임베딩(indexer/embed.py)과 질문 임베딩(rag/retriever.py)은 반드시 같은 모델이어야
같은 벡터 공간에서 비교할 수 있으므로, 모델 이름을 이 파일 한 곳에서만 정의한다.
(모델 선택 근거: report/embedding_decision.md)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 임베딩
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# 저장소
DOCS_DB = ROOT / "store" / "docs.db"
CHROMA_DIR = ROOT / "store" / "chroma"


def collection_name(config: str) -> str:
    """청킹 설정(A/B)별 Chroma 컬렉션 이름."""
    return f"chunks_{config}"
