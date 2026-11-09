"""실험 5 — 하이브리드 검색(의미 + BM25 키워드) 비교 (검색 단계, Day 5 심화 확장).

| 변형 | 내용 |
|---|---|
| dense | 현재 최종 설정 (의미 검색만) |
| rrf | 문서 선택부터 의미+BM25 를 RRF 로 섞음 (전체 하이브리드) |
| kw1 / kw2 | 문서 선택은 의미 검색 그대로, 고른 문서 안에서 BM25 청크 1개 / 2개 보강 |

현재 최종 설정(문서 6개 · 문서당 청크 3개 · 1차 자료 최소 1개)에서 검색 방식만 바꿔 비교한다.
챗봇과 같은 RAGChain.prepare 를 호출하므로 실제로 모델에 넘어가는 글과 같다.

| 지표 | 뜻 |
|---|---|
| Hit@1·@3·@6, MRR@6 | 정답 문서가 넘기는 문서 6개 중 몇 위인가 (답변 가능 34문항) |
| 정답 수치 포함률 | 정답 요지의 숫자(연도 제외)가 모델에 넘기는 글에 들어 있는 비율 — 수치 문항만 |
| 검색 시간 | 질문 1개 검색(재작성·생성 제외) 평균 초 |

실행 (질문 임베딩은 data/cache 에 있으면 OpenAI 호출 없음)
    python eval/exp_hybrid.py
결과: eval/results/exp_hybrid.json
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chain import RAGChain                    # noqa: E402
from rag.prompts import format_context            # noqa: E402
from rag.retriever import Retriever               # noqa: E402

TARGET = ["e09", "e14", "e26", "e33"]   # 실패 원인 '답 청크가 의미 검색 밖' (report/evaluation.md 2부 5절)


def key_numbers(answer_key: str) -> list[str]:
    """정답 요지의 숫자 중 확인할 값: 소수(84.6) 또는 2자리 이상 정수. 연도(19xx·20xx)와 쉼표 제외."""
    nums = re.findall(r"\d+(?:\.\d+)?", answer_key.replace(",", ""))
    return list(dict.fromkeys(n for n in nums if ("." in n or len(n) >= 2) and not re.fullmatch(r"(19|20)\d\d", n)))


def run_variant(items, retriever: Retriever, **kw) -> dict:
    # prepare()(검색·거절 판단)만 쓰므로 생성 모델은 호출하지 않는다 → 비용 없는 가짜 모델로 채움
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    rag = RAGChain(retriever=retriever, llm=FakeListChatModel(responses=["-"]), rewrite=False, **kw)
    rows, t_total = {}, 0.0
    for it in items:
        t0 = time.time()
        prep = rag.prepare(it["q"])
        t_total += time.time() - t0
        ctx = format_context(prep.sources).replace(",", "")
        gold = set(it["gold_doc_ids"])
        nums = key_numbers(it["answer_key"]) if it["type"] != "unanswerable" else []
        rows[it["id"]] = {
            "type": it["type"],
            "first_rank": next((s.n for s in prep.sources if s.doc_id in gold), None),
            "nums": nums, "nums_in_ctx": [n for n in nums if n in ctx],
            "status": prep.status, "top_score": prep.top_score,
            "docs": [s.doc_id for s in prep.sources],
        }
    ans = [v for v in rows.values() if v["type"] != "unanswerable"]
    hit = lambda k: round(sum(1 for v in ans if v["first_rank"] and v["first_rank"] <= k) / len(ans), 3)
    numq = [v for v in ans if v["nums"]]
    return {
        "options": kw, "hit@1": hit(1), "hit@3": hit(3), "hit@6": hit(6),
        "mrr@6": round(sum(1 / v["first_rank"] for v in ans if v["first_rank"]) / len(ans), 3),
        "num_coverage": round(sum(len(v["nums_in_ctx"]) for v in numq) / sum(len(v["nums"]) for v in numq), 3),
        "n_num_questions": len(numq),
        "refused_by_gate": [k for k, v in rows.items() if v["status"] != "ok"],
        "search_s_mean": round(t_total / len(items), 3),
        "per_question": rows,
    }


def main() -> None:
    items = [json.loads(l) for l in open(ROOT / "eval" / "eval_set.jsonl", encoding="utf-8") if l.strip()]
    r = Retriever("B")
    if not r.keyword.available:
        sys.exit("키워드 색인이 없습니다. 먼저: python -m indexer --only keyword --config B")
    variants = {"dense": {}, "rrf": {"hybrid": True}, "kw1": {"keyword_chunks": 1}, "kw2": {"keyword_chunks": 2}}
    res = {name: run_variant(items, r, **kw) for name, kw in variants.items()}
    print(f"{'지표':<16}" + "".join(f"{n:>10}" for n in res))
    for k in ["hit@1", "hit@3", "hit@6", "mrr@6", "num_coverage", "search_s_mean"]:
        print(f"{k:<16}" + "".join(f"{res[n][k]:>10}" for n in res))
    bp = res["dense"]["per_question"]
    for name in ["rrf", "kw1", "kw2"]:
        hp = res[name]["per_question"]
        gain = [q for q in bp if hp[q]["first_rank"] and not bp[q]["first_rank"]]
        loss = [q for q in bp if bp[q]["first_rank"] and not hp[q]["first_rank"]]
        res[name].update(gained=gain, lost=loss)
        print(f"[{name}] 정답 문서 새로 찾음 {gain} / 잃음 {loss} / 게이트 거절 {res[name]['refused_by_gate']}")
    print("\n정답 수치가 넘기는 글에 들어갔는지 (dense → kw1 → kw2)")
    for q in bp:
        a, b, c = (res[n]["per_question"][q]["nums_in_ctx"] for n in ("dense", "kw1", "kw2"))
        if bp[q]["nums"] and (a != b or a != c or q in TARGET):
            print(f"  {q}: {a} → {b} → {c}  (정답 수치 {bp[q]['nums']}){' ◀ 대상' if q in TARGET else ''}")
    (ROOT / "eval" / "results" / "exp_hybrid.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
