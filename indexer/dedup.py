"""② 인덱싱 2단계 — 중복 처리 (가이드 4-4).

입력: store/docs.db 의 parsed_documents (status='ok')
출력: store/docs.db 의 dedup_map — 문서마다 묶음(group_id)과 유지 여부(keep)
      → 청킹은 keep=1 인 문서만 사용한다. 정제 본문·원문은 지우지 않는다.

판별 기준 (Day 2 탐색 결과 근거, report/dedup_decision.md 참고)
| 구분 | 기준 | 대상 |
|---|---|---|
| 완전 중복 | URL 이 같거나 정제 본문이 완전히 같음 | 모든 문서 |
| 재배포본 | 본문 5글자 조각의 자카드 유사도 ≥ threshold, 발행일 차이 ≤ MAX_DAY_GAP 일 | 뉴스만 |

- threshold 기본값 0.8 (보수적). Day 3 평가에서 검색 상위에 같은 내용이 몰리면 0.6 으로 낮춰 재측정한다.
- 발행일 조건: 같은 이슈의 '다른 단계' 기사(예: 상임위 통과 → 본회의 통과)가 하나로 묶이지 않도록.
- 대표 기사: 묶음에서 정제 본문이 가장 긴 기사 (같으면 먼저 발행된 기사) → 정보가 가장 많은 기사를 남긴다.

유사도 계산 방법
- 4.7만 건을 모든 쌍으로 비교하면 약 11억 쌍 → MinHash(128개 해시) 로 문서 '지문'을 만들고,
  LSH(32구간 × 4) 로 비슷할 가능성이 큰 쌍만 후보로 뽑는다. 후보 쌍의 유사도는 지문 일치 비율로 추정한다.
- 비교 범위: 공백을 지운 정제 본문 앞 3,000자 (속도·메모리 절충, 뉴스 평균 길이 약 1,550자)

실행: python -m indexer --only dedup [--dedup-threshold 0.6]
"""
import datetime
import json
import re
import zlib
from collections import Counter, defaultdict

import numpy as np

from collector.storage import get_conn

DEFAULT_THRESHOLD = 0.8
MAX_DAY_GAP = 2
SHINGLE = 5            # 글자 조각 길이
NUM_PERM = 128         # MinHash 해시 개수
BANDS, ROWS = 32, 4    # LSH: 32구간 × 4 = 128 (유사도 약 0.4 이상 쌍을 후보로 잡도록 설정)
COMPARE_CHARS = 3000
SEED = 0

SCHEMA = """
CREATE TABLE IF NOT EXISTS dedup_map (
    doc_id    TEXT PRIMARY KEY,
    doc_type  TEXT NOT NULL,
    group_id  TEXT NOT NULL,   -- 묶음 대표 문서의 doc_id (혼자면 자기 자신)
    keep      INTEGER NOT NULL,-- 1 = 인덱싱 대상, 0 = 중복으로 제외
    reason    TEXT             -- exact_url / exact_body / near_dup / NULL(유지)
);
"""


# ---------------------------------------------------------------------------
# MinHash 지문 · LSH 후보 쌍
# ---------------------------------------------------------------------------
def minhash_signature(text: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """본문 → 128개 최솟값 지문. 두 문서 지문의 일치 비율 ≈ 두 문서의 자카드 유사도."""
    t = re.sub(r"\s+", "", text)[:COMPARE_CHARS]
    x = np.unique(np.fromiter(
        (zlib.crc32(t[j:j + SHINGLE].encode()) for j in range(max(len(t) - SHINGLE + 1, 1))),
        dtype=np.uint64))
    return ((np.outer(x, a) + b) % np.uint64(4294967311)).min(axis=0).astype(np.uint32)


def candidate_pairs(sigs: np.ndarray, max_bucket: int = 300) -> np.ndarray:
    """LSH: 지문을 32구간으로 나눠 한 구간이라도 같은 문서끼리를 후보 쌍으로 모은다."""
    cand = set()
    for band in range(BANDS):
        buckets = defaultdict(list)
        for i, row in enumerate(sigs[:, band * ROWS:(band + 1) * ROWS]):
            buckets[row.tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= max_bucket:   # 지나치게 큰 묶음(상투 문구)은 건너뜀
                cand.update((members[x], members[y]) for x in range(len(members)) for y in range(x + 1, len(members)))
    return np.array(sorted(cand), dtype=np.int64).reshape(-1, 2)


class UnionFind:
    """비슷한 쌍을 이어서 묶음(그룹)을 만든다. A~B, B~C 이면 A·B·C 가 한 묶음."""
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run(threshold: float = DEFAULT_THRESHOLD) -> dict:
    """parsed_documents 를 중복 처리해 dedup_map 에 저장하고 통계를 반환한다."""
    conn = get_conn()
    conn.executescript(SCHEMA)
    docs = conn.execute(
        "select p.doc_id, p.doc_type, d.url, d.published_at, p.body from parsed_documents p "
        "join documents d using(doc_id) where p.status='ok'").fetchall()
    n = len(docs)
    uf, reason = UnionFind(n), {}

    # 1) 완전 중복: URL · 정제 본문 (모든 문서 종류)
    first_by = {}
    for i, (doc_id, _, url, _, body) in enumerate(docs):
        for kind, key in (("exact_url", url), ("exact_body", zlib.crc32(body.encode()))):
            if key in first_by.setdefault(kind, {}):
                j = first_by[kind][key]
                uf.union(i, j)
                # 쌍의 두 문서 모두에 사유를 남긴다 (어느 쪽이 대표가 되든 제외 사유가 정확하도록)
                reason.setdefault(i, kind)
                reason.setdefault(j, kind)
            else:
                first_by[kind][key] = i

    # 2) 재배포본: 뉴스만, MinHash + LSH
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 1 << 31, NUM_PERM).astype(np.uint64)
    b = rng.integers(0, 1 << 31, NUM_PERM).astype(np.uint64)
    news_idx = [i for i, d in enumerate(docs) if d[1] == "news"]
    sigs = np.vstack([minhash_signature(docs[i][4], a, b) for i in news_idx])
    pairs = candidate_pairs(sigs)
    sims = (sigs[pairs[:, 0]] == sigs[pairs[:, 1]]).mean(axis=1) if len(pairs) else np.array([])
    date = lambda i: datetime.date.fromisoformat(docs[i][3][:10])
    merged_pairs = 0
    for (p, q), s in zip(pairs, sims):
        i, j = news_idx[p], news_idx[q]
        if s >= threshold and abs((date(i) - date(j)).days) <= MAX_DAY_GAP:
            uf.union(i, j)
            merged_pairs += 1

    # 3) 묶음별 대표 선정: 본문이 가장 긴 문서, 같으면 먼저 발행된 문서
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    rows, removed = [], Counter()
    for members in groups.values():
        rep = max(members, key=lambda i: (len(docs[i][4]), -date(i).toordinal()))
        for i in members:
            keep = int(i == rep)
            why = None if keep else reason.get(i, "near_dup")
            if not keep:
                removed[f"{docs[i][1]}:{why}"] += 1
            rows.append((docs[i][0], docs[i][1], docs[rep][0], keep, why))

    kept = Counter(r[1] for r in rows if r[3])
    before = Counter(d[1] for d in docs)
    stats = {
        "params": {"threshold": threshold, "max_day_gap": MAX_DAY_GAP, "shingle": SHINGLE,
                   "num_perm": NUM_PERM, "bands_rows": f"{BANDS}x{ROWS}", "compare_chars": COMPARE_CHARS},
        "before": dict(before), "after": dict(kept), "removed": dict(removed),
        "news_candidate_pairs": int(len(pairs)), "news_merged_pairs": merged_pairs,
        "sim_histogram": {f">={t}": int((sims >= t).sum()) for t in (0.9, 0.8, 0.7, 0.6, 0.5)},
        "groups_with_dups": sum(1 for m in groups.values() if len(m) > 1),
    }
    with conn:
        conn.execute("DELETE FROM dedup_map")
        conn.executemany("INSERT INTO dedup_map VALUES (?,?,?,?,?)", rows)
        conn.execute("INSERT INTO collect_log VALUES (datetime('now','localtime'), 'dedup', ?)",
                     (json.dumps(stats, ensure_ascii=False),))
    conn.close()
    return stats
