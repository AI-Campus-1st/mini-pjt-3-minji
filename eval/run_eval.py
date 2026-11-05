"""평가 하네스 — 평가셋으로 검색 품질을 측정한다 (가이드 4-3, 4-8).

챗봇과 같은 검색기(rag/retriever.Retriever)를 호출한다. 결과는 실험 태그별로 eval/results/ 에 저장한다.

측정 지표 (문서 단위: 정답 문서 gold_doc_ids 중 하나가 검색 상위 k개 문서 안에 있는가)
| 지표 | 뜻 |
|---|---|
| Hit@1, Hit@3, Hit@5 | 정답 문서가 상위 k개 안에 있는 질문 비율 (답변 가능 문항만) |
| MRR@10 | 정답 문서 첫 순위의 역수 평균 (10위 밖이면 0) |
| 유형별 Hit@3 | 사실/비교/시점/종합 — 취약 유형 파악 |
| top1 유사도 | 답변 가능 vs 답변 불가 문항의 1위 점수 분포 → 거절 임계값 근거 (Day 4) |
| 상위 5개 쏠림 | 상위 5개 문서 중 서로 유사도 ≥ 0.6 인 쌍의 수 → 중복 처리 기준(0.8 → 0.6) 판단 근거 |

실행 (맥, 프로젝트 루트 — 질문 40개 임베딩에 OpenAI 필요, 캐시되어 두 번째부터는 호출 없음)
    python eval/run_eval.py --tag baseline                 # 설정 A, B 모두
    python eval/run_eval.py --tag baseline --config B      # 하나만
    python eval/run_eval.py --compare baseline             # 저장된 결과 비교표
"""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # 프로젝트 루트 모듈(rag, indexer, settings) 사용

from indexer.dedup import minhash_signature  # noqa: E402  (중복 처리와 같은 유사도 계산)
from rag.retriever import Retriever           # noqa: E402
from settings import DOCS_DB                  # noqa: E402

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
RESULTS = ROOT / "eval" / "results"
K_DOCS, FETCH_K = 10, 50
CROWD_TOP, CROWD_SIM = 5, 0.6


def load_eval_set() -> list[dict]:
    return [json.loads(l) for l in open(EVAL_SET, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# 상위 결과 쏠림: 상위 5개 문서끼리 본문이 얼마나 비슷한가 (중복 처리와 같은 MinHash 유사도)
# ---------------------------------------------------------------------------
class Similarity:
    def __init__(self):
        rng = np.random.default_rng(0)   # indexer/dedup.py 와 같은 해시 설정
        self.a = rng.integers(1, 1 << 31, 128).astype(np.uint64)
        self.b = rng.integers(0, 1 << 31, 128).astype(np.uint64)
        self.conn = sqlite3.connect(DOCS_DB)
        self.cache = {}

    def sig(self, doc_id: str):
        if doc_id not in self.cache:
            body = self.conn.execute("select body from parsed_documents where doc_id=?", (doc_id,)).fetchone()[0]
            self.cache[doc_id] = minhash_signature(body, self.a, self.b)
        return self.cache[doc_id]

    def near_dup_pairs(self, doc_ids: list[str], threshold: float = CROWD_SIM) -> list[tuple]:
        out = []
        for x, y in combinations(doc_ids, 2):
            s = float((self.sig(x) == self.sig(y)).mean())
            if s >= threshold:
                out.append((x, y, round(s, 2)))
        return out


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(config: str, items: list[dict], retriever=None) -> dict:
    retriever = retriever or Retriever(config=config)
    sim = Similarity()
    per_q = []
    for it in items:
        docs = retriever.search_docs(it["q"], k_docs=K_DOCS, fetch_k=FETCH_K)
        gold = set(it["gold_doc_ids"])
        rank = next((i + 1 for i, h in enumerate(docs) if h.doc_id in gold), None)
        top5 = [h.doc_id for h in docs[:CROWD_TOP]]
        pairs = sim.near_dup_pairs(top5)
        per_q.append({
            "id": it["id"], "type": it["type"], "q": it["q"], "rank": rank,
            "top1_score": docs[0].score if docs else None,
            "top5": [{"doc_id": h.doc_id, "doc_type": h.doc_type, "title": h.title[:60], "score": h.score,
                      "gold": h.doc_id in gold} for h in docs[:CROWD_TOP]],
            "near_dup_pairs_top5": pairs,
        })

    answerable = [p for p in per_q if p["type"] != "unanswerable"]
    unans = [p for p in per_q if p["type"] == "unanswerable"]

    def hit(ps, k):
        return round(sum(p["rank"] is not None and p["rank"] <= k for p in ps) / len(ps), 3) if ps else None

    by_type = defaultdict(list)
    for p in answerable:
        by_type[p["type"]].append(p)
    ans_scores = [p["top1_score"] for p in answerable]
    un_scores = [p["top1_score"] for p in unans]
    return {
        "config": config, "n_answerable": len(answerable), "n_unanswerable": len(unans),
        "hit@1": hit(answerable, 1), "hit@3": hit(answerable, 3), "hit@5": hit(answerable, 5),
        "mrr@10": round(sum(1 / p["rank"] for p in answerable if p["rank"]) / len(answerable), 3),
        "hit@3_by_type": {t: hit(ps, 3) for t, ps in by_type.items()},
        "n_by_type": {t: len(ps) for t, ps in by_type.items()},
        "top1_score": {"answerable_mean": round(float(np.mean(ans_scores)), 4),
                       "answerable_min": round(float(np.min(ans_scores)), 4),
                       "unanswerable_mean": round(float(np.mean(un_scores)), 4) if un_scores else None,
                       "unanswerable_max": round(float(np.max(un_scores)), 4) if un_scores else None},
        "crowding_top5": {"questions_with_near_dup": sum(1 for p in per_q if p["near_dup_pairs_top5"]),
                          "near_dup_pairs_total": sum(len(p["near_dup_pairs_top5"]) for p in per_q),
                          "threshold": CROWD_SIM},
        "per_question": per_q,
    }


def save(result: dict, tag: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"retrieval_{tag}_{result['config']}.json"
    result = {"tag": tag, "run_at": datetime.now().isoformat(timespec="seconds"),
              "params": {"k_docs": K_DOCS, "fetch_k": FETCH_K, "crowd_top": CROWD_TOP, "crowd_sim": CROWD_SIM}, **result}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def summary_line(r: dict) -> str:
    return (f"[{r['config']}] Hit@1 {r['hit@1']} | Hit@3 {r['hit@3']} | Hit@5 {r['hit@5']} | MRR {r['mrr@10']} | "
            f"쏠림(상위5 유사 쌍) {r['crowding_top5']['near_dup_pairs_total']}쌍/"
            f"{r['crowding_top5']['questions_with_near_dup']}문항")


def compare(tag: str) -> None:
    rows = {}
    for p in sorted(RESULTS.glob(f"retrieval_{tag}_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows[r["config"]] = r
    if not rows:
        print(f"'{tag}' 결과가 없습니다."); return
    cfgs = list(rows)
    print(f"\n=== 실험 '{tag}' 비교 ===")
    print("지표".ljust(24) + "".join(c.ljust(12) for c in cfgs))
    for k in ["hit@1", "hit@3", "hit@5", "mrr@10"]:
        print(k.ljust(24) + "".join(str(rows[c][k]).ljust(12) for c in cfgs))
    for t in rows[cfgs[0]]["hit@3_by_type"]:
        print(f"Hit@3 [{t}]({rows[cfgs[0]]['n_by_type'][t]})".ljust(24) + "".join(str(rows[c]['hit@3_by_type'].get(t)).ljust(12) for c in cfgs))
    for k in ["answerable_mean", "answerable_min", "unanswerable_mean", "unanswerable_max"]:
        print(f"top1 {k}".ljust(24) + "".join(str(rows[c]['top1_score'][k]).ljust(12) for c in cfgs))
    print("쏠림 쌍/문항".ljust(24) + "".join(f"{rows[c]['crowding_top5']['near_dup_pairs_total']}/{rows[c]['crowding_top5']['questions_with_near_dup']}".ljust(12) for c in cfgs))
    print("\n질문별 정답 첫 순위 (- = 10위 밖, 답변 불가는 1위 점수)")
    print("ID   유형".ljust(24) + "".join(c.ljust(12) for c in cfgs))
    for i, p in enumerate(rows[cfgs[0]]["per_question"]):
        cells = [str(rows[c]["per_question"][i]["top1_score"]) if p["type"] == "unanswerable"
                 else str(rows[c]["per_question"][i]["rank"] or "-") for c in cfgs]
        print(f"{p['id']}  {p['type']}".ljust(24) + "".join(x.ljust(12) for x in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="검색 품질 평가")
    ap.add_argument("--tag", default="baseline", help="실험 이름 (결과 파일명에 사용)")
    ap.add_argument("--config", choices=["A", "B"], nargs="+", default=["A", "B"])
    ap.add_argument("--compare", metavar="TAG", help="저장된 실험 결과 비교표만 출력")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        items = load_eval_set()
        for cfg in args.config:
            r = evaluate(cfg, items)
            print(summary_line(r), "→", save(r, args.tag).relative_to(ROOT))
        compare(args.tag)
