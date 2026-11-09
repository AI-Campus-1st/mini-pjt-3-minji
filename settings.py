"""프로젝트 공통 설정 — 인덱서(② indexer)와 챗봇·평가(④ app, rag, eval)가 함께 쓰는 값.

문서 임베딩(indexer/embed.py)과 질문 임베딩(rag/retriever.py)은 반드시 같은 모델이어야
같은 벡터 공간에서 비교할 수 있으므로, 모델 이름을 이 파일 한 곳에서만 정의한다.
(모델 선택 근거: report/embedding_decision.md)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")   # OPENAI_API_KEY, (선택) LLM_MODEL

# 임베딩
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# 저장소
DOCS_DB = ROOT / "store" / "docs.db"
CHROMA_DIR = ROOT / "store" / "chroma"


# 답변 생성 (rag/chain.py)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # .env 에 LLM_MODEL=... 로 바꿀 수 있다
# 1차 자료: 발행 주체가 직접 낸 문서 (뉴스 = 언론 보도). 검색 균형(retriever)과 출처 표기(prompts)에 사용
PRIMARY_DOC_TYPES = ("dart", "press", "report")
CHUNK_CONFIG = "B"        # 챗봇이 쓰는 청킹 설정 (Day 3 측정에서 Hit@3 우세, report/eval_baseline.md)


def keyword_db(config: str) -> Path:
    """설정(A/B)별 키워드(BM25) 색인 파일. 인덱서가 쓰고 검색기가 읽는다."""
    return ROOT / "store" / f"keyword_{config}.db"


def collection_name(config: str) -> str:
    """청킹 설정(A/B)별 Chroma 컬렉션 이름."""
    return f"chunks_{config}"
