"""검색기 — 질문을 임베딩해 Chroma 에서 가까운 청크를 찾는다.

챗봇(app/)과 평가 하네스(eval/run_eval.py)가 **같은 검색기**를 쓴다 (가이드 4-2).
조회 전용: 문서 임베딩 생성·인덱스 쓰기를 하지 않는다. 질문 임베딩만 만든다.

사용 예
    from rag.retriever import Retriever
    r = Retriever(config="B")
    hits = r.search("한미반도체의 주요 제품은?", k=5)                     # 청크 단위 결과
    docs = r.search_docs("한미반도체의 주요 제품은?", k_docs=5)           # 문서 단위(같은 문서 청크는 1개로)
    hits = r.search("...", k=5, doc_types=["dart"], year_from=2025)      # 필터 (가이드 4-7)

반환 항목(Hit): chunk_id, doc_id, doc_type, source, title, url, published_at, section, page, text, score
- score = 코사인 유사도 (1 - 코사인 거리). 높을수록 질문과 가깝다. 거절 임계값 설정에 사용 (Day 4).
"""
import json
import os
from dataclasses import asdict, dataclass

import chromadb
from dotenv import load_dotenv

from settings import CHROMA_DIR, EMBED_MODEL, ROOT, collection_name

QUERY_CACHE = ROOT / "data" / "cache" / "query_embeddings.json"   # 같은 질문은 다시 임베딩하지 않음


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_type: str
    source: str
    title: str
    url: str
    published_at: str
    section: str | None
    page: int | None
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class QueryEmbedder:
    """질문 → 벡터 (OpenAI, settings.EMBED_MODEL). 디스크 캐시로 같은 질문의 재호출을 막는다."""

    def __init__(self, use_cache: bool = True):
        from openai import OpenAI
        load_dotenv(ROOT / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(".env 에 OPENAI_API_KEY 가 없습니다.")
        self.client = OpenAI()
        self.use_cache = use_cache
        self.cache = json.loads(QUERY_CACHE.read_text(encoding="utf-8")) if use_cache and QUERY_CACHE.exists() else {}

    def __call__(self, text: str) -> list[float]:
        key = f"{EMBED_MODEL}::{text}"
        if key not in self.cache:
            self.cache[key] = self.client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
            if self.use_cache:
                QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
                QUERY_CACHE.write_text(json.dumps(self.cache), encoding="utf-8")
        return self.cache[key]


def build_where(doc_types: list[str] | None = None, year_from: int | None = None,
                year_to: int | None = None) -> dict | None:
    """필터 조건 → Chroma where 절. 조건이 없으면 None."""
    conds = []
    if doc_types:
        conds.append({"doc_type": {"$in": list(doc_types)}})
    if year_from is not None:
        conds.append({"year": {"$gte": int(year_from)}})
    if year_to is not None:
        conds.append({"year": {"$lte": int(year_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class Retriever:
    """청킹 설정(config) 하나의 Chroma 컬렉션을 검색한다.

    Args:
        config: "A"(고정 길이) 또는 "B"(구조 기반)
        embed_fn: 질문 → 벡터 함수 (기본 QueryEmbedder. 테스트에서 주입)
    """

    def __init__(self, config: str = "B", embed_fn=None, chroma_dir=CHROMA_DIR):
        self.config = config
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.col = client.get_collection(collection_name(config))   # 없으면 오류 → 인덱서를 먼저 실행해야 함
        model = (self.col.metadata or {}).get("embed_model")
        if model and model != EMBED_MODEL:
            raise RuntimeError(f"인덱스 모델({model})과 질문 임베딩 모델({EMBED_MODEL})이 다릅니다.")
        self.embed = embed_fn or QueryEmbedder()

    def search(self, query: str, k: int = 5, doc_types=None, year_from=None, year_to=None) -> list[Hit]:
        """질문과 가까운 청크 k개 (유사도 높은 순)."""
        res = self.col.query(query_embeddings=[self.embed(query)], n_results=k,
                             where=build_where(doc_types, year_from, year_to),
                             include=["metadatas", "documents", "distances"])
        hits = []
        for cid, meta, doc, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]):
            hits.append(Hit(chunk_id=cid, doc_id=meta["doc_id"], doc_type=meta["doc_type"],
                            source=meta.get("source", ""), title=meta.get("title", ""), url=meta.get("url", ""),
                            published_at=meta.get("published_at", ""), section=meta.get("section"),
                            page=meta.get("page"), text=doc, score=round(1 - dist, 4)))
        return hits

    def search_docs(self, query: str, k_docs: int = 5, fetch_k: int = 50, **filters) -> list[Hit]:
        """문서 단위 결과: 청크 fetch_k 개를 가져와 같은 문서는 가장 점수 높은 청크 1개만 남긴다.

        평가(Hit@k 는 '정답 문서'가 상위 k개 문서 안에 있는지)와,
        검색 결과가 한 문서의 여러 청크로 채워지는 것을 막는 데 쓴다.
        """
        seen, out = set(), []
        for h in self.search(query, k=fetch_k, **filters):
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                out.append(h)
                if len(out) == k_docs:
                    break
        return out
