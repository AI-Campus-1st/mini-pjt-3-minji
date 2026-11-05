"""② 인덱싱 4단계 — 임베딩 생성 및 벡터 DB(Chroma) 적재.

입력: store/docs.db 의 chunks (config = A / B)
출력: store/chroma/ 의 컬렉션 chunks_A, chunks_B (디스크에 영속화, 가이드 4-5)
      - 벡터 DB 에는 청크와 메타데이터만 저장하고, 원문 전체는 docs.db(documents)에 둔다.

동작
1) 설정별 청크를 읽는다.
2) 이미 컬렉션에 들어 있는 chunk_id 는 건너뛴다 → 중간에 끊겨도 다시 실행하면 이어서 진행.
3) BATCH_SIZE 개씩 OpenAI 임베딩 API 호출 → Chroma 에 추가.
   요청 한도 초과·일시 오류는 기다렸다가 다시 시도한다 (공용 키 사용).
4) 끝나면 청크 수와 컬렉션 수가 같은지 검증하고 collect_log 에 기록한다.

이 모듈만 문서 임베딩을 만든다. 챗봇(app/)은 질문 임베딩만 만들고 인덱스에 쓰지 않는다.

실행 (맥, 프로젝트 루트, OPENAI_API_KEY 필요)
    python -m indexer --only embed --config A
    python -m indexer --only embed --config B
    python -m indexer --only embed --config A --limit 200   # 동작 확인용 소량 실행
"""
import json
import os
import time
from datetime import datetime

import chromadb
from dotenv import load_dotenv

from collector.storage import get_conn
from settings import CHROMA_DIR, EMBED_MODEL, ROOT, collection_name

BATCH_SIZE = 100          # 요청 1회당 청크 수 (청크 최대 약 1,600자 → 요청당 토큰 한도 안쪽)
MAX_RETRIES = 6
META_FIELDS = ["doc_id", "doc_type", "source", "title", "url", "published_at", "year", "section", "page"]


# ---------------------------------------------------------------------------
# 임베딩 함수
# ---------------------------------------------------------------------------
def openai_embedder():
    """텍스트 목록 → 벡터 목록. 요청 한도 초과·일시 오류는 지수적으로 기다렸다가 다시 시도한다."""
    from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env 에 OPENAI_API_KEY 가 없습니다.")
    client = OpenAI()

    def embed(texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
                return [d.embedding for d in resp.data]
            except (RateLimitError, APIConnectionError) as e:
                wait = 5 * 2 ** attempt
                print(f"    {type(e).__name__} → {wait}초 대기 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code >= 500:
                    wait = 5 * 2 ** attempt
                    print(f"    서버 오류 {e.status_code} → {wait}초 대기 후 재시도")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("임베딩 요청이 계속 실패해 중단했습니다. 다시 실행하면 이어서 진행합니다.")
    return embed


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
def get_collection(config: str, chroma_dir=CHROMA_DIR):
    """디스크 영속 Chroma 컬렉션을 연다 (없으면 생성). 거리 = 코사인."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=collection_name(config),
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )


def to_metadata(row: dict) -> dict:
    """Chroma 메타데이터는 None 을 받지 않으므로, 값이 없는 항목은 뺀다."""
    return {k: row[k] for k in META_FIELDS if row.get(k) is not None}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run(config: str = "A", limit: int | None = None, embed_fn=None, chroma_dir=CHROMA_DIR) -> dict:
    """설정 하나의 청크를 임베딩해 Chroma 에 적재하고 통계를 반환한다.

    embed_fn, chroma_dir 은 테스트용 주입 지점 (기본: OpenAI, store/chroma).
    """
    conn = get_conn()
    conn.row_factory = lambda cur, row: {d[0]: v for d, v in zip(cur.description, row)}
    sql = f"select chunk_id, text, {', '.join(META_FIELDS)} from chunks where config=? order by chunk_id"
    rows = conn.execute(sql + (f" limit {int(limit)}" if limit else ""), (config,)).fetchall()
    conn.close()

    col = get_collection(config, chroma_dir)
    done = set()
    for start in range(0, len(rows), 5000):   # 이미 적재된 id 확인 (이어하기)
        ids = [r["chunk_id"] for r in rows[start:start + 5000]]
        done.update(col.get(ids=ids, include=[])["ids"])
    todo = [r for r in rows if r["chunk_id"] not in done]
    total_chars = sum(len(r["text"]) for r in todo)
    print(f"  [{config}] 청크 {len(rows):,}개 중 이미 적재 {len(done):,}개 → 이번에 {len(todo):,}개 ({total_chars:,}자)")

    embed = embed_fn or (openai_embedder() if todo else None)
    t0, n = time.time(), 0
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        vecs = embed([r["text"] for r in batch])
        col.add(ids=[r["chunk_id"] for r in batch], embeddings=vecs,
                documents=[r["text"] for r in batch], metadatas=[to_metadata(r) for r in batch])
        n += len(batch)
        if (start // BATCH_SIZE) % 20 == 0 or n == len(todo):
            el = time.time() - t0
            eta = el / n * (len(todo) - n) if n else 0
            print(f"    {n:,}/{len(todo):,}  경과 {el / 60:.1f}분  남은 시간 약 {eta / 60:.1f}분")

    count = col.count()
    expected = len(rows)
    stats = {"config": config, "model": EMBED_MODEL, "chunks_expected": expected, "collection_count": count,
             "embedded_this_run": n, "seconds": round(time.time() - t0, 1), "complete": count >= expected,
             "limit": limit, "finished_at": datetime.now().isoformat(timespec="seconds")}
    if not limit and count != expected:
        print(f"  ⚠️ 컬렉션 {count:,}개 ≠ 청크 {expected:,}개 — 다시 실행해 이어서 적재하세요.")
    if embed_fn is None:   # 테스트 실행은 기록하지 않음
        conn = get_conn()
        with conn:
            conn.execute("INSERT INTO collect_log VALUES (datetime('now','localtime'), 'embed', ?)",
                         (json.dumps(stats, ensure_ascii=False),))
        conn.close()
    return stats
