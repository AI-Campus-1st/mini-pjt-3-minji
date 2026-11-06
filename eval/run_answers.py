"""답변 평가 — 평가셋 전체를 챗봇과 같은 RAG 체인(rag/chain.RAGChain)으로 답하고 기록한다 (가이드 4-8, Day 5).

검색 평가(run_eval.py)가 '정답 문서를 찾았나'를 본다면, 이 스크립트는 '답변이 맞고 근거를 바르게 댔나'를 본다.

자동 지표 (사람 채점 전에 계산 가능한 것)
| 지표 | 뜻 |
|---|---|
| 거절 정확도 | 답변 불가 문항 중 거절한 비율 (높을수록 좋음) |
| 과잉 거절률 | 답변 가능 문항인데 거절한 비율 (낮을수록 좋음) |
| 정답 문서 검색률 | 답변 가능 문항 중 근거로 넘긴 문서(상위 6개)에 정답 문서가 있는 비율 |
| 정답 문서 인용률 | 답변 가능 문항 중 답변이 [n]으로 인용한 문서에 정답 문서가 있는 비율 |
| 응답 시간 | 질문 1개 처리 시간(초) 평균·최대 |

사람 채점 (정확 / 부분 정확 / 오답, 출처 정확성)은 결과 파일을 읽고 eval/grades_{tag}.json 에 기록한다.

실행 (맥, 프로젝트 루트 — OpenAI 호출: 질문당 생성 1회. 40문항 기준 수십 원 수준)
    python eval/run_answers.py --tag baseline              # 전체 40문항
    python eval/run_answers.py --tag baseline --ids e01 e36   # 일부만 (시험 실행)
    python eval/run_answers.py --tag chunks3 --chunks-per-doc 3   # 실험 1: 문서당 청크 3개
    python eval/run_answers.py --tag balance1 --primary-min 1 --primary-margin 0.10   # 실험 2
    python eval/run_answers.py --tag prompt2 --prompt v2   # 실험 3: 프롬프트 v2
    python eval/run_answers.py --tag prompt3 --prompt v3   # 실험 4: 프롬프트 v3 (+ 연도 기준)
중간에 멈춰도 같은 tag 로 다시 실행하면 끝난 문항은 건너뛴다.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chain import RAGChain   # noqa: E402  (챗봇과 같은 체인 — 가이드 4-2 계층 분리)
from settings import CHUNK_CONFIG, LLM_MODEL   # noqa: E402

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
RESULTS = ROOT / "eval" / "results"


def load_eval_set() -> list[dict]:
    return [json.loads(l) for l in open(EVAL_SET, encoding="utf-8") if l.strip()]


def out_path(tag: str) -> Path:
    return RESULTS / f"answers_{tag}.jsonl"


def run(tag: str, ids: list[str] | None = None, config: str = CHUNK_CONFIG,
        chunks_per_doc: int | None = None, primary_min: int | None = None,
        primary_margin: float | None = None, prompt: str | None = None) -> Path:
    """평가 문항을 하나씩 답하고 결과를 한 줄씩 저장한다 (중단 후 이어서 실행 가능)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = out_path(tag)
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    items = [it for it in load_eval_set() if (not ids or it["id"] in ids) and it["id"] not in done]
    # 평가 문항은 대화 맥락이 없는 단일 질문이므로 재작성은 끈다 (불필요한 LLM 호출 방지)
    kw = {k: v for k, v in dict(chunks_per_doc=chunks_per_doc, primary_min=primary_min,
                                primary_margin=primary_margin, prompt_version=prompt).items() if v is not None}
    rag = RAGChain(config=config, rewrite=False, **kw)
    print(f"[{tag}] 모델 {LLM_MODEL} · 설정 {config} · 문서당 청크 {rag.chunks_per_doc} · "
          f"1차 자료 최소 {rag.primary_min}(격차 {rag.primary_margin}) · 프롬프트 {rag.prompt_version} · "
          f"실행 {len(items)}문항 (이미 완료 {len(done)})")
    with open(path, "a", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            t0 = time.time()
            r = rag.answer(it["q"])
            gold = set(it["gold_doc_ids"])
            cited_ids = [s["doc_id"] for s in r["sources"] if s["n"] in r["cited"]]
            row = {
                "id": it["id"], "type": it["type"], "q": it["q"], "answer_key": it["answer_key"],
                "gold_doc_ids": it["gold_doc_ids"],
                **{k: r[k] for k in ["status", "refused", "answer", "cited", "sources"]},
                "gold_retrieved": bool(gold & {s["doc_id"] for s in r["sources"]}),
                "gold_cited": bool(gold & set(cited_ids)),
                "latency_s": round(time.time() - t0, 2),
                "model": LLM_MODEL, "config": config, "chunks_per_doc": rag.chunks_per_doc,
                "primary_min": rag.primary_min, "primary_margin": rag.primary_margin,
                "prompt_version": rag.prompt_version, "run_at": datetime.now().isoformat(timespec="seconds"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()   # 중간에 멈춰도 여기까지는 저장
            mark = "거절" if r["refused"] else f"인용 {r['cited']}"
            print(f"  {i:>2}/{len(items)} {it['id']} {it['type']:<12} {mark:<16} {row['latency_s']}s")
    return path


def summarize(tag: str) -> dict:
    """저장된 답변으로 자동 지표를 계산한다."""
    rows = [json.loads(l) for l in open(out_path(tag), encoding="utf-8")]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    rate = lambda xs, key: round(sum(bool(x[key]) for x in xs) / len(xs), 3) if xs else None
    lat = [r["latency_s"] for r in rows]
    s = {
        "tag": tag, "n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "refusal_accuracy": rate(una, "refused"),
        "false_refusal_rate": rate(ans, "refused"),
        "gold_retrieved_rate": rate(ans, "gold_retrieved"),
        "gold_cited_rate": rate(ans, "gold_cited"),
        "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_max_s": max(lat) if lat else None,
        "false_refusals": [r["id"] for r in ans if r["refused"]],
        "missed_refusals": [r["id"] for r in una if not r["refused"]],
    }
    print(json.dumps(s, ensure_ascii=False, indent=1))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="답변 평가 (RAG 체인 전체)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--ids", nargs="+", help="일부 문항만 (예: e01 e36)")
    ap.add_argument("--config", choices=["A", "B"], default=CHUNK_CONFIG)
    ap.add_argument("--chunks-per-doc", type=int, help="문서마다 넘길 청크 수 (기본: rag/chain.CHUNKS_PER_DOC)")
    ap.add_argument("--primary-min", type=int, help="1차 자료 최소 보장 수 (기본: rag/chain.PRIMARY_MIN)")
    ap.add_argument("--primary-margin", type=float, help="1위 점수 대비 허용 격차 (기본: rag/chain.PRIMARY_MARGIN)")
    ap.add_argument("--prompt", choices=["v1", "v2", "v3"], help="시스템 프롬프트 버전 (기본: rag/prompts.PROMPT_VERSION)")
    ap.add_argument("--summary", action="store_true", help="실행 없이 저장된 결과 요약만")
    args = ap.parse_args()
    if not args.summary:
        run(args.tag, args.ids, args.config, args.chunks_per_doc, args.primary_min, args.primary_margin, args.prompt)
    summarize(args.tag)
