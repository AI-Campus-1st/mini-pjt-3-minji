"""임베딩 모델 비교 파일럿 — 실험용 부분집합 생성 (Day 2).

목적: 전체 인덱싱 전에, 우리 데이터로 두 임베딩 모델(KURE-v1, OpenAI text-embedding-3-small)의
      검색 정확도와 속도를 비교해 모델을 결정한다. (가이드 4-5: 선택 근거 기술, 복수 모델 비교)

구성
- 질문: 대표 질문 10개 (report/representative_questions.md), 정답 문서는 제목·본문 규칙으로 선정
  → 파일럿용 근사 정답이다. 본 평가셋(eval/eval_set.jsonl)은 원문을 읽고 따로 확정한다.
- 문서: 정답 문서 전부 + DART·KIET·보도자료 전부 + 무작위 뉴스 3,000건 (seed 고정)
- 청크: 파일럿 전용 단순 청킹 — '제목\n' + 본문 1,000자 창(겹침 100자)
  (본 청킹 전략은 indexer/chunk.py 에서 따로 비교한다)

출력: data/pilot/chunks.jsonl, data/pilot/questions.json
실행: python eval/pilot_build.py
"""
import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "store" / "docs.db"
OUT = ROOT / "data" / "pilot"
SEED, N_RANDOM_NEWS = 42, 3000
CHUNK, OVERLAP = 1000, 100

# 대표 질문 + 정답 문서 선정 규칙 (SQL WHERE 절). gold 가 비면 '답변 불가' 문항.
QUESTIONS = [
    ("q01", "fact", "과기정통부 'AI 반도체 혁신연구소'로 선정된 대학은 어디이고, 언제 선정됐나?",
     "doc_type='press' and title like '%혁신연구소%'"),
    ("q02", "fact", "한미반도체의 주요 제품은 무엇인가?",
     "doc_type='dart' and source='한미반도체'"),
    ("q03", "fact", "오픈엣지테크놀로지의 주력 사업은 무엇인가?",
     "doc_type='dart' and source like '오픈엣지%'"),
    ("q04", "fact", "국산 NPU 성능지표 'K-Perf'는 무엇이고, 어디에 처음 활용됐나?",
     "doc_type='press' and (title like '%성능지표%' or title like '%K-Perf%')"),
    ("q05", "compare", "삼성전자와 SK하이닉스는 2025년 사업보고서에서 HBM 사업을 어떻게 다르게 설명하나?",
     "doc_type='dart' and source in ('삼성전자','SK하이닉스') and title like '%2025.12%'"),
    ("q06", "compare", "대만과 한국의 반도체 산업 구조는 어떻게 다른가?",
     "(doc_type='report' and title like '%대만%') or (doc_type='news' and title like '%대만%' and (title like '%한국%' or title like '%韓%') and title like '%반도체%' and (title like '%구조%' or title like '%비교%' or title like '%차이%' or title like '%생태계%'))"),
    ("q07", "time", "SK하이닉스의 실적이 크게 성장한 시기는 언제이고, 그 성장을 이끈 기술(제품)은 무엇인가?",
     "(doc_type='dart' and source='SK하이닉스' and (title like '%2024.12%' or title like '%2025.12%')) or (doc_type='news' and title like '%SK하이닉스%' and title like '%HBM%' and (title like '%최대%' or title like '%실적%' or title like '%영업이익%'))"),
    ("q08", "synthesis", "정부의 국산 AI 반도체(NPU) 지원 정책은 2022년 이후 어떻게 전개됐나?",
     "doc_type='press' and (title like '%인공지능 반도체%' or title like '%AI 반도체%' or title like '%인공지능반도체%' or title like '%AI반도체%' or title like '%NPU%' or title like '%신경망%') and (title like '%지원%' or title like '%육성%' or title like '%전략%' or title like '%대책%' or title like '%사업%' or title like '%투자%')"),
    ("q09", "synthesis", "반도체 특별법의 근로시간 예외 조항을 둘러싼 찬반 쟁점은 무엇인가?",
     "doc_type in ('news','press') and title like '%특별법%' and (title like '%52시간%' or title like '%근로시간%' or title like '%주52%' or title like '%노동시간%')"),
    ("q10", "unanswerable", "퓨리오사AI RNGD 칩의 판매 단가는 얼마인가?", None),
]


def chunk_text(title: str, body: str) -> list[str]:
    """제목을 앞에 붙인 1,000자 창(겹침 100자) 청크 목록."""
    body = body.replace("\f", "\n")
    step = CHUNK - OVERLAP
    return [f"{title}\n{body[i:i + CHUNK]}" for i in range(0, max(len(body), 1), step)]


def main() -> None:
    c = sqlite3.connect(DB)
    OUT.mkdir(parents=True, exist_ok=True)

    questions, gold_all = [], set()
    for qid, qtype, q, where in QUESTIONS:
        gold = [r[0] for r in c.execute(f"select doc_id from documents where {where}")] if where else []
        gold_all.update(gold)
        questions.append({"id": qid, "type": qtype, "q": q, "gold_doc_ids": gold})

    random.seed(SEED)
    news_ids = [r[0] for r in c.execute("select doc_id from documents where doc_type='news'")]
    doc_ids = set(random.sample(news_ids, N_RANDOM_NEWS)) | gold_all
    doc_ids |= {r[0] for r in c.execute("select doc_id from documents where doc_type in ('dart','report','press')")}

    n_chunks, n_chars = 0, 0
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for doc_id, doc_type, title, body in c.execute("select doc_id, doc_type, title, body from documents"):
            if doc_id not in doc_ids:
                continue
            for i, text in enumerate(chunk_text(title, body)):
                f.write(json.dumps({"doc_id": doc_id, "doc_type": doc_type, "i": i, "text": text}, ensure_ascii=False) + "\n")
                n_chunks += 1
                n_chars += len(text)
    json.dump(questions, open(OUT / "questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"문서 {len(doc_ids):,}건 / 청크 {n_chunks:,}개 / {n_chars:,}자")
    for q in questions:
        print(f"  {q['id']} [{q['type']}] 정답 문서 {len(q['gold_doc_ids'])}건")


if __name__ == "__main__":
    main()
