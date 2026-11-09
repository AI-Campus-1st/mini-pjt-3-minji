"""키워드 색인 — 하이브리드 검색(의미 검색 + 키워드 검색)용 BM25 색인을 만든다 (Day 5 심화 확장).

왜 필요한가
    의미 검색(임베딩)은 "뜻이 비슷한" 청크를 잘 찾지만, 질문의 특정 단어·숫자(예: "출원 특허", "84.6%")가
    들어 있는 표 청크는 순위가 낮아 놓친다 (e09·e14·e26·e33, report/eval_answers.md).
    BM25 는 질문 단어가 청크에 몇 번, 얼마나 드물게 나오는지로 점수를 매겨 이런 청크를 끌어올린다.

토큰화
    한국어는 "한미반도체의"처럼 조사가 붙어 띄어쓰기 단위로는 단어가 맞지 않는다.
    형태소 분석기(새 의존성) 대신, 한글 연속 구간은 글자 2개씩 자른 바이그램으로,
    영문·숫자는 단어 그대로(소문자, 소수점 숫자는 84.6 → 84d6) 쓴다.
    예) "한미반도체의 TC본더" → 한미 미반 반도 도체 체의 tc 본더

저장
    SQLite 내장 전문 검색(FTS5)을 쓴다 (새 라이브러리 없음). store/keyword_{config}.db
    - fts(tokens, chunk_id UNINDEXED, doc_type UNINDEXED, year UNINDEXED)
    검색 쪽(rag/retriever.KeywordIndex)은 읽기 전용으로 연다.

실행 (프로젝트 루트) — 중단돼도 다시 실행하면 이어서 만든다
    python -m indexer --only keyword --config B
"""
import re
import sqlite3
import time

from settings import DOCS_DB, ROOT

HANGUL_RUN = re.compile(r"[가-힣]+")
TOKEN = re.compile(r"[가-힣]+|[0-9]+(?:\.[0-9]+)?|[a-z0-9]+")
BATCH = 5000


def keyword_db(config: str):
    return ROOT / "store" / f"keyword_{config}.db"


def tokenize(text: str) -> list[str]:
    """한글 구간은 글자 바이그램(1글자면 그대로), 영문·숫자는 단어 그대로. 검색 질문에도 같은 함수를 쓴다."""
    out = []
    for t in TOKEN.findall(text.lower()):
        if HANGUL_RUN.fullmatch(t):
            out.extend([t] if len(t) == 1 else [t[i:i + 2] for i in range(len(t) - 1)])
        else:
            # FTS5 는 '.' 을 구분자로 보므로 소수점 숫자는 84.6 → 84d6 으로 한 토큰 유지
            out.append(t.replace(".", "d"))
    return out


def run(config: str = "B", rebuild: bool = False) -> dict:
    """chunks 테이블(설정 config)의 청크를 토큰화해 FTS5 색인에 넣는다. 이미 넣은 청크는 건너뛴다."""
    path = keyword_db(config)
    if rebuild and path.exists():
        path.unlink()
    out = sqlite3.connect(path)
    # unicode61 + 구분자 없음: 공백으로 이어 붙인 토큰을 그대로 단어로 인식
    out.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5("
                "tokens, chunk_id UNINDEXED, doc_type UNINDEXED, year UNINDEXED, tokenize='unicode61')")
    done = {r[0] for r in out.execute("SELECT chunk_id FROM fts")}
    src = sqlite3.connect(f"file:{DOCS_DB}?mode=ro", uri=True)
    total = src.execute("SELECT count(*) FROM chunks WHERE config = ?", (config,)).fetchone()[0]
    t0, added, buf = time.time(), 0, []
    for cid, dtype, year, title, text in src.execute(
            "SELECT chunk_id, doc_type, year, title, text FROM chunks WHERE config = ?", (config,)):
        if cid in done:
            continue
        # 제목도 함께 색인 (청크 본문에 회사명이 없을 때 대비)
        buf.append((" ".join(tokenize(f"{title or ''} {text}")), cid, dtype, year))
        if len(buf) >= BATCH:
            out.executemany("INSERT INTO fts VALUES (?, ?, ?, ?)", buf); out.commit()
            added += len(buf); buf = []
            print(f"  {len(done) + added:,}/{total:,} ({time.time() - t0:.0f}초)", flush=True)
    if buf:
        out.executemany("INSERT INTO fts VALUES (?, ?, ?, ?)", buf); out.commit(); added += len(buf)
    n = out.execute("SELECT count(*) FROM fts").fetchone()[0]
    out.close(); src.close()
    print(f"  키워드 색인 {config}: {n:,}개 (이번에 추가 {added:,}) → {path.relative_to(ROOT)}")
    if n != total:
        print(f"  ⚠️ 청크 수({total:,})와 다릅니다. 다시 실행하면 이어서 만듭니다.")
    return {"config": config, "indexed": n, "chunks": total}
