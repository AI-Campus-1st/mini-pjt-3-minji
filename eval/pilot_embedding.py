"""임베딩 모델 비교 파일럿 — 두 모델로 임베딩 → 검색 정확도·속도 비교 (Day 2).

입력 : data/pilot/chunks.jsonl, data/pilot/questions.json  (eval/pilot_build.py 가 생성)
출력 : eval/results/pilot_embedding_<model>.json            (모델별 결과)
캐시 : data/pilot/emb_<model>.npy                          (다시 실행하면 임베딩을 재사용)

실행 (프로젝트 루트, pjt3 환경)
    python eval/pilot_embedding.py --model kure     # KURE-v1 (맥에서 계산, 처음엔 모델 약 2.3GB 다운로드)
    python eval/pilot_embedding.py --model openai   # OpenAI text-embedding-3-small (.env 의 OPENAI_API_KEY)
    python eval/pilot_embedding.py --compare        # 두 결과 비교표

지표 (문서 단위): 청크 검색 결과를 문서 순위로 합친 뒤, 정답 문서가 처음 나오는 순위로 계산
- Hit@k : 상위 k개 문서 안에 정답 문서가 있는 질문 비율 (k = 1, 3, 5)
- MRR   : 정답 문서 첫 순위의 역수 평균 (상위 10위 밖이면 0)
- 답변 불가 문항(q10)은 정확도 계산에서 빼고, 1위 유사도만 기록 (거절 임계값 설정 참고용)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "data" / "pilot"
RESULTS = ROOT / "eval" / "results"
FULL_CORPUS_CHARS = 79_201_469  # 전체 지식베이스 글자 수 (2026-11-04, 중복 제거 전) → 전체 인덱싱 시간 추정용

MODELS = {
    "kure": "nlpai-lab/KURE-v1",
    "openai": "text-embedding-3-small",
}


# ---------------------------------------------------------------------------
# 임베딩 함수: 문자열 목록 → 정규화된 벡터 행렬 (n, dim)
# ---------------------------------------------------------------------------
def make_kure_encoder():
    """KURE-v1 을 불러와 encode 함수를 반환한다. 맥의 GPU(MPS)가 있으면 사용한다."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  장치: {device}")
    model = SentenceTransformer(MODELS["kure"], device=device)

    def encode(texts, show_progress=False):
        return model.encode(texts, batch_size=16, normalize_embeddings=True,
                            show_progress_bar=show_progress, convert_to_numpy=True)
    return encode


def make_openai_encoder():
    """OpenAI 임베딩 encode 함수를 반환한다. 요청 한도 초과 시 기다렸다가 다시 시도한다."""
    from openai import OpenAI, RateLimitError

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env 에 OPENAI_API_KEY 가 없습니다.")
    client = OpenAI()

    def encode(texts, show_progress=False):
        out, batch = [], 100
        for start in range(0, len(texts), batch):
            for attempt in range(6):
                try:
                    resp = client.embeddings.create(model=MODELS["openai"], input=texts[start:start + batch])
                    break
                except RateLimitError:
                    wait = 2 ** attempt * 5
                    print(f"  요청 한도 초과 → {wait}초 대기 후 재시도")
                    time.sleep(wait)
            else:
                raise RuntimeError("요청 한도 초과가 계속되어 중단했습니다.")
            out.extend(d.embedding for d in resp.data)
            if show_progress and (start // batch) % 20 == 0:
                print(f"  {start + len(resp.data):,}/{len(texts):,}")
        vec = np.array(out, dtype=np.float32)
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return encode


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def doc_ranking(scores: np.ndarray, chunk_docs: list[str], top_docs: int = 10) -> list[tuple[str, float]]:
    """청크 유사도를 문서 순위로 합친다 (문서 점수 = 그 문서 청크 중 최고 점수)."""
    ranked, seen = [], set()
    for idx in np.argsort(-scores):
        d = chunk_docs[idx]
        if d not in seen:
            seen.add(d)
            ranked.append((d, float(scores[idx])))
            if len(ranked) == top_docs:
                break
    return ranked


def evaluate(q_vecs, c_vecs, chunk_docs, questions) -> dict:
    """질문별 정답 순위와 Hit@k·MRR 을 계산한다."""
    per_q, ranks = [], []
    for q, qv in zip(questions, q_vecs):
        ranking = doc_ranking(c_vecs @ qv, chunk_docs)
        gold = set(q["gold_doc_ids"])
        rank = next((i + 1 for i, (d, _) in enumerate(ranking) if d in gold), None)
        per_q.append({"id": q["id"], "type": q["type"], "rank": rank,
                      "top1_score": round(ranking[0][1], 4),
                      "top3": [d for d, _ in ranking[:3]]})
        if gold:
            ranks.append(rank)
    n = len(ranks)
    return {
        "n_questions": n,
        "hit@1": round(sum(r is not None and r <= 1 for r in ranks) / n, 3),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
        "hit@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "per_question": per_q,
    }


def run(model_key: str) -> None:
    chunks = [json.loads(l) for l in open(PILOT / "chunks.jsonl", encoding="utf-8")]
    questions = json.load(open(PILOT / "questions.json", encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    n_chars = sum(len(t) for t in texts)
    print(f"[{MODELS[model_key]}] 청크 {len(texts):,}개, {n_chars:,}자")

    encode = make_kure_encoder() if model_key == "kure" else make_openai_encoder()

    cache = PILOT / f"emb_{model_key}.npy"
    if cache.exists():
        c_vecs, elapsed = np.load(cache), None
        print("  캐시된 임베딩 사용 (속도는 이전 결과 파일 참고)")
    else:
        t0 = time.time()
        c_vecs = encode(texts, show_progress=True)
        elapsed = time.time() - t0
        np.save(cache, c_vecs)

    t0 = time.time()
    q_vecs = encode([q["q"] for q in questions])
    q_ms = (time.time() - t0) / len(questions) * 1000

    result = {"model": MODELS[model_key], "chunks": len(texts), "chars": n_chars, "dim": int(c_vecs.shape[1]),
              "query_ms_per_question": round(q_ms, 1), **evaluate(q_vecs, c_vecs, [c["doc_id"] for c in chunks], questions)}
    if elapsed is not None:
        result["index_seconds"] = round(elapsed, 1)
        result["chars_per_second"] = round(n_chars / elapsed)
        result["full_corpus_minutes_est"] = round(FULL_CORPUS_CHARS / (n_chars / elapsed) / 60, 1)
    else:
        prev = RESULTS / f"pilot_embedding_{model_key}.json"
        if prev.exists():
            old = json.load(open(prev, encoding="utf-8"))
            for k in ("index_seconds", "chars_per_second", "full_corpus_minutes_est"):
                if k in old:
                    result[k] = old[k]

    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(RESULTS / f"pilot_embedding_{model_key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, ensure_ascii=False, indent=1))


def compare() -> None:
    rows = {k: json.load(open(RESULTS / f"pilot_embedding_{k}.json", encoding="utf-8"))
            for k in MODELS if (RESULTS / f"pilot_embedding_{k}.json").exists()}
    keys = ["hit@1", "hit@3", "hit@5", "mrr", "index_seconds", "full_corpus_minutes_est", "query_ms_per_question"]
    print("지표".ljust(26) + "".join(k.ljust(14) for k in rows))
    for k in keys:
        print(k.ljust(26) + "".join(str(rows[m].get(k, "-")).ljust(14) for m in rows))
    print("\n질문별 정답 첫 순위 (None = 10위 밖, q10 = 답변 불가: 1위 유사도)")
    for i, q in enumerate(next(iter(rows.values()))["per_question"]):
        cells = []
        for m in rows:
            p = rows[m]["per_question"][i]
            cells.append(f"{p['top1_score']}" if q["type"] == "unanswerable" else str(p["rank"]))
        print(f"  {q['id']} [{q['type']}]".ljust(26) + "".join(c.ljust(14) for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    elif args.model:
        run(args.model)
    else:
        ap.print_help()
