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
import sqlite3
from dataclasses import asdict, dataclass

import chromadb
from dotenv import load_dotenv

from settings import CHROMA_DIR, EMBED_MODEL, PRIMARY_DOC_TYPES, ROOT, collection_name

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


class KeywordIndex:
    """BM25 키워드 검색 (store/keyword_{config}.db, indexer/keyword.py 가 만든 FTS5 색인). 조회 전용.

    질문을 색인과 같은 방식으로 토큰화(한글 바이그램 + 영문·숫자 단어)하고, 토큰을 OR 로 묶어
    BM25 점수 순으로 청크 id 를 돌려준다. 색인이 없으면 available=False (하이브리드 검색을 쓸 수 없음).
    """

    # 질문에 공통으로 붙는 말 → 어느 문서에나 있어 검색을 흐린다 (질문 쪽에서만 제외)
    QUESTION_WORDS = ["무엇인가", "어떻게", "어떤", "얼마인가", "얼마나", "얼마", "각각", "어디", "언제", "몇", "인가",
                      "하는가", "했는가", "되는가", "되었나", "됐나", "있는가", "이며", "에서", "따르면", "대해"]

    def __init__(self, config: str = "B"):
        from indexer.keyword import keyword_db, tokenize   # 같은 토큰화 함수 (색인·검색 일치 보장)
        self._tokenize = tokenize
        self.config = config
        self._stop = {t for w in self.QUESTION_WORDS for t in tokenize(w)}
        path = keyword_db(config)
        self.available = path.exists()
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False) if self.available else None

    def _match(self, query: str) -> str | None:
        """질문 → FTS5 MATCH 식 (토큰 OR). 공통 질문어 제외, 남는 토큰이 없으면 None."""
        toks = [t for t in dict.fromkeys(self._tokenize(query)) if t not in self._stop]
        return " OR ".join(f'"{t}"' for t in toks) if toks else None

    def search_in_doc(self, query: str, doc_id: str, k: int = 3) -> list[str]:
        """한 문서 안에서 BM25 상위 k개 청크 id. 긴 공시·보도자료에서 답이 든 표 청크를 찾는 데 쓴다."""
        m = self._match(query) if self.available else None
        if not m:
            return []
        return [r[0] for r in self.conn.execute(
            "SELECT chunk_id FROM fts WHERE fts MATCH ? AND chunk_id LIKE ? ORDER BY bm25(fts) LIMIT ?",
            (m, f"{self.config}:{doc_id}:%", k))]

    def search(self, query: str, k: int = 50, doc_types=None, year_from=None, year_to=None) -> list[str]:
        """BM25 상위 k개 청크 id (좋은 순)."""
        m = self._match(query) if self.available else None
        if not m:
            return []
        sql = "SELECT chunk_id FROM fts WHERE fts MATCH ?"
        args: list = [m]
        if doc_types:
            sql += f" AND doc_type IN ({','.join('?' * len(doc_types))})"; args += list(doc_types)
        if year_from is not None:
            sql += " AND year >= ?"; args.append(int(year_from))
        if year_to is not None:
            sql += " AND year <= ?"; args.append(int(year_to))
        sql += " ORDER BY bm25(fts) LIMIT ?"; args.append(k)   # bm25() 는 작을수록 관련도 높음
        return [r[0] for r in self.conn.execute(sql, args)]


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

    @property
    def keyword(self) -> KeywordIndex:
        """키워드 색인은 하이브리드 검색을 처음 쓸 때 연다."""
        if not hasattr(self, "_keyword"):
            self._keyword = KeywordIndex(self.config)
        return self._keyword

    def search_hybrid(self, query: str, k: int = 50, rrf_k: int = 60, **filters) -> list[Hit]:
        """의미 검색 상위 k + BM25 상위 k 를 RRF(순위 역수 합산)로 합친 청크 목록 (좋은 순).

        RRF 점수 = Σ 1 / (rrf_k + 순위). 점수 단위가 다른 두 검색을 순위만으로 공정하게 섞는다.
        Hit.score 는 계속 '질문과의 코사인 유사도'로 둔다 (거절 판단·화면 표시가 같은 의미를 갖도록).
        키워드로만 찾은 청크는 Chroma 에서 저장된 벡터를 꺼내 유사도를 계산한다.
        """
        qv = self.embed(query)
        dense = self.search(query, k=k, **filters)
        kw_ids = self.keyword.search(query, k=k, **filters)
        fused: dict[str, float] = {}
        for rank, h in enumerate(dense, 1):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0) + 1 / (rrf_k + rank)
        for rank, cid in enumerate(kw_ids, 1):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)
        hits = {h.chunk_id: h for h in dense}
        hits.update(self._hits_by_ids([c for c in kw_ids if c not in hits], qv))
        return [hits[cid] for cid in sorted(fused, key=lambda c: -fused[c]) if cid in hits]

    def _hits_by_ids(self, ids: list[str], qv) -> dict[str, Hit]:
        """청크 id → Hit. Chroma 에 저장된 벡터로 질문과의 코사인 유사도를 계산한다 (키워드로 찾은 청크용)."""
        if not ids:
            return {}
        import numpy as np
        got = self.col.get(ids=ids, include=["metadatas", "documents", "embeddings"])
        q = np.asarray(qv, dtype=float); q /= np.linalg.norm(q)
        out = {}
        for cid, meta, doc, emb in zip(got["ids"], got["metadatas"], got["documents"], got["embeddings"]):
            e = np.asarray(emb, dtype=float)
            out[cid] = Hit(chunk_id=cid, doc_id=meta["doc_id"], doc_type=meta["doc_type"],
                           source=meta.get("source", ""), title=meta.get("title", ""), url=meta.get("url", ""),
                           published_at=meta.get("published_at", ""), section=meta.get("section"),
                           page=meta.get("page"), text=doc, score=round(float(q @ (e / np.linalg.norm(e))), 4))
        return out

    def add_keyword_chunks(self, query: str, groups: list[list[Hit]], n: int, cap: int) -> list[list[Hit]]:
        """문서 안 키워드 청크 보강: 고른 문서마다 BM25 상위 n개 청크를 넣는다 (문서 선택은 그대로).

        한 문서의 청크 순서 = [의미 검색 1위, 키워드 청크들, 나머지 의미 검색 청크] 를 cap 개까지.
        의미 검색 1위 청크는 항상 남겨 기존 근거를 잃지 않게 한다.
        """
        if n <= 0 or not self.keyword.available:
            return groups
        qv = self.embed(query)
        out = []
        for g in groups:
            have = {h.chunk_id for h in g}
            kw = [c for c in self.keyword.search_in_doc(query, g[0].doc_id, k=n + len(g)) if c not in have][:n]
            kw_hits = self._hits_by_ids(kw, qv)
            merged = [g[0]] + [kw_hits[c] for c in kw if c in kw_hits] + g[1:]
            out.append(merged[:cap])
        return out

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

    @staticmethod
    def _group(hits: list[Hit], k_docs: int, chunks_per_doc: int) -> list[list[Hit]]:
        """청크 목록(점수 순) → 문서별 묶음. 상위 k_docs 문서, 문서마다 최대 chunks_per_doc 청크."""
        groups: dict[str, list[Hit]] = {}
        for h in hits:
            if h.doc_id in groups:
                if len(groups[h.doc_id]) < chunks_per_doc:
                    groups[h.doc_id].append(h)
            elif len(groups) < k_docs:
                groups[h.doc_id] = [h]
        return list(groups.values())   # dict 는 넣은 순서 유지 → 문서 1위 점수 순

    def search_docs_grouped(self, query: str, k_docs: int = 5, fetch_k: int = 50, chunks_per_doc: int = 1,
                            primary_min: int = 0, primary_margin: float | None = None, hybrid: bool = False,
                            keyword_chunks: int = 0,
                            doc_types=None, year_from=None, year_to=None) -> list[list[Hit]]:
        """문서 단위 결과 + 문서마다 청크 여러 개: [[문서1 청크들], [문서2 청크들], ...].

        1) 전체에서 검색해 상위 k_docs 문서를 고르고, 상위 fetch_k 청크 안의 같은 문서 청크를
           최대 chunks_per_doc 개까지 묶는다 (Day 5 실험 1 'chunks3').
        2) primary_min > 0 이면 1차 자료(공시·보도자료·보고서)를 최소 primary_min 개 보장한다 (실험 2).
           뉴스가 청크의 93.5% 라 1차 자료가 상위권에서 밀리는 문제 대응.
           - 1차 자료만 따로 검색해, 부족한 수만큼 가장 아래 순위의 뉴스와 바꾼다.
           - primary_margin: 1위 문서 점수보다 이만큼 이상 낮은 1차 자료는 넣지 않는다
             (질문과 무관한 1차 자료가 뉴스 자리를 뺏지 않게). None 이면 제한 없음.
        3) hybrid=True 이면 1)·2)의 검색을 의미+키워드 하이브리드(search_hybrid)로 한다 (심화 확장 실험).
           이때 문서 순서는 RRF 순위를 따르고, 점수 순 재정렬은 하지 않는다.
        4) keyword_chunks > 0 이면 고른 문서마다 문서 안 BM25 청크를 그 수만큼 보강한다 (add_keyword_chunks).
        호출 위치: rag/chain.RAGChain.prepare
        """
        filters = dict(year_from=year_from, year_to=year_to)
        find = self.search_hybrid if hybrid else self.search
        groups = self._group(find(query, k=fetch_k, doc_types=doc_types, **filters), k_docs, chunks_per_doc)
        if primary_min > 0 and groups:
            groups = self._ensure_primary(query, groups, find, fetch_k, k_docs, chunks_per_doc,
                                          primary_min, primary_margin, hybrid, doc_types, filters)
        return self.add_keyword_chunks(query, groups, keyword_chunks, chunks_per_doc)

    def _ensure_primary(self, query, groups, find, fetch_k, k_docs, chunks_per_doc,
                        primary_min, primary_margin, hybrid, doc_types, filters) -> list[list[Hit]]:
        """1차 자료 최소 primary_min 개 보장 (실험 2). 부족하면 가장 아래 순위 뉴스와 바꾼다."""
        allowed = [t for t in (doc_types or PRIMARY_DOC_TYPES) if t in PRIMARY_DOC_TYPES]
        have = sum(g[0].doc_type in PRIMARY_DOC_TYPES for g in groups)
        if not allowed or have >= primary_min:
            return groups                      # 이미 충분하거나, 필터로 1차 자료를 제외한 경우
        top = max(g[0].score for g in groups)   # 하이브리드에서는 1위 문서가 유사도 최고가 아닐 수 있음
        ids = {g[0].doc_id for g in groups}
        cands = [g for g in self._group(find(query, k=fetch_k, doc_types=allowed, **filters),
                                        k_docs, chunks_per_doc)
                 if g[0].doc_id not in ids and (primary_margin is None or g[0].score >= top - primary_margin)]
        for g in cands[:primary_min - have]:
            news_idx = [i for i, x in enumerate(groups) if x[0].doc_type not in PRIMARY_DOC_TYPES]
            if len(groups) >= k_docs and news_idx:
                groups.pop(news_idx[-1])       # 가장 아래 순위 뉴스를 뺀다
            groups.append(g)
        if hybrid:
            return groups                      # RRF 순위 유지 (보장으로 추가된 1차 자료는 뒤에)
        return sorted(groups, key=lambda g: -g[0].score)   # 번호 [n] 이 점수 순이 되도록

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
