"""실험 2 — 문서 종류 균형 검색: 1차 자료 최소 보장 방식 비교 (검색 단계 지표, Day 5).

챗봇이 실제로 모델에 넘기는 문서 6건(rag/chain.K_DOCS)을 기준으로, 정답 문서가 몇 위에 있는지 비교한다.
같은 Retriever.search_docs_grouped 를 호출하므로 챗봇과 결과가 같다.

| 변형 | 내용 |
|---|---|
| base | 현재 (보장 없음) |
| min{N}_m{M} | 1차 자료 최소 N개 보장, 1위 대비 유사도 격차 M 이내만 |
| fixed3 | 고정 분할: 1차 자료 3 + 뉴스 3 (비교용, 코드에는 없음) |

실행 (질문 임베딩은 data/cache 에 있으므로 OpenAI 호출 없음)
    python eval/exp_balance.py
결과: eval/results/exp_balance.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chain import FETCH_K, K_DOCS            # noqa: E402
from rag.retriever import Retriever              # noqa: E402
from settings import CHUNK_CONFIG, PRIMARY_DOC_TYPES   # noqa: E402

TARGET = ["e01", "e07", "e16", "e18", "e19", "e23", "e28", "e34"]   # 실패 원인 '1차 자료 묻힘' (report/eval_answers.md)
VARIANTS = {"base": dict(primary_min=0)}
for n in (1, 2):
    for m in (0.06, 0.08, 0.10):
        VARIANTS[f"min{n}_m{m:.2f}"] = dict(primary_min=n, primary_margin=m)


def fixed_split(r: Retriever, q: str, n_primary: int = 3) -> list[str]:
    """비교용 고정 분할: 1차 자료 상위 n + 뉴스 상위 (K_DOCS - n), 점수 순 정렬."""
    p = r.search_docs_grouped(q, k_docs=n_primary, fetch_k=FETCH_K, doc_types=list(PRIMARY_DOC_TYPES))
    n = r.search_docs_grouped(q, k_docs=K_DOCS - len(p), fetch_k=FETCH_K, doc_types=["news"])
    return [g[0].doc_id for g in sorted(p + n, key=lambda g: -g[0].score)]


def metrics(ranked: dict[str, list[str]], items: list[dict]) -> dict:
    """ranked: 문항 id → 문서 id 목록(순서대로). 답변 가능 문항만 집계."""
    ans = [it for it in items if it["type"] != "unanswerable"]
    first = {}
    for it in ans:
        g = set(it["gold_doc_ids"])
        first[it["id"]] = next((i for i, d in enumerate(ranked[it["id"]], 1) if d in g), None)
    hit = lambda k: round(sum(1 for v in first.values() if v and v <= k) / len(ans), 3)
    return {
        "hit@1": hit(1), "hit@3": hit(3), "hit@6": hit(K_DOCS),
        "mrr@6": round(sum(1 / v for v in first.values() if v) / len(ans), 3),
        "target_hit@6": sum(1 for q in TARGET if first.get(q)),
        "primary_docs_mean": None, "first_rank": first,
    }


def main() -> None:
    items = [json.loads(l) for l in open(ROOT / "eval" / "eval_set.jsonl", encoding="utf-8") if l.strip()]
    r = Retriever(CHUNK_CONFIG)
    out = {}
    for name, kw in [*VARIANTS.items(), ("fixed3", None)]:
        ranked, nprim = {}, []
        for it in items:
            ids = (fixed_split(r, it["q"]) if kw is None else
                   [g[0].doc_id for g in r.search_docs_grouped(it["q"], k_docs=K_DOCS, fetch_k=FETCH_K, **kw)])
            ranked[it["id"]] = ids
            nprim.append(sum(d.split("_")[0] in PRIMARY_DOC_TYPES for d in ids))
        m = metrics(ranked, items)
        m["primary_docs_mean"] = round(sum(nprim) / len(nprim), 2)
        out[name] = m
    base = out["base"]["first_rank"]
    print(f"{'변형':<14}{'Hit@1':>7}{'Hit@3':>7}{'Hit@6':>7}{'MRR@6':>7}{'대상8':>6}{'1차평균':>7}  얻음 / 잃음(정답이 6위 밖으로)")
    for name, m in out.items():
        fr = m["first_rank"]
        gain = [q for q in fr if fr[q] and not base[q]]
        loss = [q for q in fr if base[q] and not fr[q]]
        worse = [q for q in fr if base[q] and fr[q] and fr[q] > base[q]]
        m.update(gained=gain, lost=loss, rank_worse=worse)
        print(f"{name:<14}{m['hit@1']:>7}{m['hit@3']:>7}{m['hit@6']:>7}{m['mrr@6']:>7}{m['target_hit@6']:>6}"
              f"{m['primary_docs_mean']:>7}  {gain} / {loss} (순위 하락 {worse})")
    (ROOT / "eval" / "results" / "exp_balance.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
