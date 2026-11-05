"""② 인덱싱 3단계 — 청킹 (가이드 4-4).

입력: store/docs.db 의 parsed_documents(정제 본문) + dedup_map(keep=1) + documents(메타데이터)
출력: store/docs.db 의 chunks — 설정(config)별 청크 + 메타데이터
      → indexer/embed.py 가 config 별로 임베딩해 Chroma 컬렉션에 적재한다.

비교 설정 (크기는 같게, '나누는 방법'만 다르게 → 결과 차이를 방법 하나로 설명하기 위함)
| 설정 | 나누는 방법 | 청크 앞머리 |
|---|---|---|
| A 고정 길이 (기준선) | 모든 문서를 1,000자씩, 겹침 100자 | 제목 |
| B 문서 구조 기반     | 문서 종류별 자연 경계에서 목표 1,000자로 묶음 | 제목 + 소제목(DART)·쪽 번호(KIET) |

B 의 문서 종류별 규칙
- news  : 1,200자 이하면 기사 통째로 1개, 길면 문장 경계로 묶음 (앞 청크 마지막 문장을 겹침으로 반복)
- press : 개조식 항목(□ ○ ◦ ㅇ ※ ▪ 1. 가. 등) 경계로 묶음
- dart  : '## 소제목' 단위로 나눈 뒤, 길면 줄 경계로 묶음. 청크에 소제목을 붙임
- report: 쪽(\\f) → 문단 경계로 묶음. 청크에 시작 쪽 번호를 붙임
- 공통  : 한 덩어리가 MAX_CHARS 를 넘으면 문장 → 글자 순으로 다시 자른다 (임베딩 입력 한도·검색 정밀도)

실행: python -m indexer --only chunk
"""
import json
import re
from collections import Counter

from collector.storage import get_conn

TARGET_CHARS = 1000      # 목표 청크 길이 (본문 기준, 앞머리 제외)
OVERLAP_CHARS = 100      # 설정 A 겹침
MAX_CHARS = 1400         # 설정 B 에서 한 단위를 미리 쪼개는 기준 (짧은 소제목이 붙으면 약 1,650자까지 가능)
NEWS_WHOLE_MAX = 1200    # 설정 B: 이 길이 이하 뉴스는 통째로

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,   -- {config}:{doc_id}:c{순번}
    config       TEXT NOT NULL,      -- A / B
    doc_id       TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    source       TEXT,
    title        TEXT,
    url          TEXT,
    published_at TEXT,
    year         INTEGER,
    section      TEXT,               -- DART 소제목 (설정 B)
    page         INTEGER,            -- KIET 시작 쪽 번호 (설정 B)
    chunk_index  INTEGER,
    text         TEXT NOT NULL,      -- 임베딩·검색에 쓰는 전체 텍스트 (앞머리 + 본문)
    body_chars   INTEGER             -- 앞머리를 뺀 본문 길이
);
CREATE INDEX IF NOT EXISTS idx_chunks_config ON chunks(config);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

SENT_END = re.compile(r"(?<=[다요음함됨임\.\?\!\"”’])\s+(?=\S)")
PRESS_BULLET = re.compile(r"^\s*(?:[□■○●◦ㅇ◇◆▪▶※\-]|\d+\.|[가-하]\.|\(\d+\)|[①-⑳])\s*")


# ---------------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """한국어 문장 경계(…다. …요. 등 뒤 공백)로 나눈다."""
    return [s for s in SENT_END.split(text) if s.strip()]


def hard_split(text: str, size: int = TARGET_CHARS) -> list[str]:
    """문장으로도 못 나누는 긴 덩어리를 글자 수로 자른다 (최후 수단)."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def pack(units: list[str], target: int = TARGET_CHARS, max_chars: int = MAX_CHARS,
         sep: str = "\n", overlap_last: bool = False) -> list[str]:
    """작은 단위(문장·항목·줄)를 순서대로 이어 붙여 target 근처 길이의 청크로 묶는다.

    - 한 단위가 max_chars 보다 길면 문장 → 글자 순으로 먼저 쪼갠다.
    - overlap_last=True 면 앞 청크의 마지막 단위를 다음 청크 맨 앞에 반복한다 (문맥 연결).
    """
    pieces = []
    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(u) <= max_chars:
            pieces.append(u)
        else:
            for s in split_sentences(u):
                pieces.extend([s] if len(s) <= max_chars else hard_split(s))

    chunks, cur, cur_len = [], [], 0
    for p in pieces:
        if cur and cur_len + len(p) > target and cur_len < target // 4:
            # 지금까지 모은 게 소제목 한 줄처럼 짧으면 떼어 내지 않고 다음 단위와 함께 묶는다
            # ("4) 시장여건 및 경쟁상황" 만 있는 청크 방지 — 청킹 샘플 확인 결과)
            cur.append(p)
            cur_len += len(p) + len(sep)
            continue
        if cur and cur_len + len(p) > target:
            chunks.append(sep.join(cur))
            carry = cur[-1] if overlap_last and len(cur[-1]) < target // 3 else None
            cur, cur_len = ([carry], len(carry)) if carry else ([], 0)
        cur.append(p)
        cur_len += len(p) + len(sep)
    if cur:
        tail = sep.join(cur)
        # 너무 짧은 마지막 조각은 앞 청크에 붙인다 (자투리 청크 방지)
        if chunks and len(tail) < target // 4 and len(chunks[-1]) + len(tail) <= max_chars:
            chunks[-1] = chunks[-1] + sep + tail
        else:
            chunks.append(tail)
    return chunks


# ---------------------------------------------------------------------------
# 설정 A: 고정 길이
# ---------------------------------------------------------------------------
def chunk_fixed(body: str) -> list[dict]:
    body = body.replace("\f", "\n")
    step = TARGET_CHARS - OVERLAP_CHARS
    return [{"body": body[i:i + TARGET_CHARS]} for i in range(0, max(len(body), 1), step)
            if body[i:i + TARGET_CHARS].strip()]


# ---------------------------------------------------------------------------
# 설정 B: 문서 구조 기반
# ---------------------------------------------------------------------------
def chunk_news(body: str) -> list[dict]:
    if len(body) <= NEWS_WHOLE_MAX:
        return [{"body": body}]
    return [{"body": c} for c in pack(split_sentences(body), sep=" ", overlap_last=True)]


def chunk_press(body: str) -> list[dict]:
    blocks, cur = [], []
    for line in body.split("\n"):
        if PRESS_BULLET.match(line) and cur:      # 새 항목이 시작되면 이전 항목을 닫는다
            blocks.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [{"body": c} for c in pack(blocks)]


def chunk_dart(body: str) -> list[dict]:
    out, section, lines = [], None, []

    def flush():
        if section and lines:
            for c in pack(lines):
                out.append({"body": c, "section": section})

    for line in body.split("\n"):
        if line.startswith("## "):
            flush()
            section, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()
    return out


def chunk_report(body: str) -> list[dict]:
    units = []   # (쪽 번호, 문단)
    for page_no, page in enumerate(body.split("\f"), start=1):
        for para in re.split(r"\n\s*\n", page):
            if para.strip():
                units.append((page_no, para.replace("\n", " ").strip()))
    out, cur, cur_len, start_page = [], [], 0, None
    for page_no, para in units:
        if cur and cur_len + len(para) > TARGET_CHARS:
            for c in pack(cur, sep="\n"):
                out.append({"body": c, "page": start_page})
            cur, cur_len, start_page = [], 0, None
        if start_page is None:
            start_page = page_no
        cur.append(para)
        cur_len += len(para) + 1
    if cur:
        for c in pack(cur, sep="\n"):
            out.append({"body": c, "page": start_page})
    return out


STRUCTURED = {"news": chunk_news, "press": chunk_press, "dart": chunk_dart, "report": chunk_report}


def make_text(title: str, piece: dict) -> str:
    """임베딩할 텍스트 = 제목(필수, 가이드 4-4) + 위치 정보(설정 B) + 본문."""
    head = [title]
    if piece.get("section"):
        head.append(f"[{piece['section']}]")
    if piece.get("page"):
        head.append(f"[p.{piece['page']}]")
    return "\n".join(head) + "\n" + piece["body"]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run() -> dict:
    """keep=1 문서를 설정 A, B 로 청킹해 chunks 테이블에 저장하고 통계를 반환한다."""
    conn = get_conn()
    conn.executescript(SCHEMA)
    docs = conn.execute(
        "select d.doc_id, d.doc_type, d.source, d.title, d.url, d.published_at, p.body "
        "from documents d join parsed_documents p using(doc_id) join dedup_map m using(doc_id) "
        "where m.keep=1 and p.status='ok'").fetchall()

    rows, stats = [], {}
    for config in ("A", "B"):
        lens, per_type = [], Counter()
        for doc_id, doc_type, source, title, url, published_at, body in docs:
            pieces = chunk_fixed(body) if config == "A" else STRUCTURED[doc_type](body)
            for k, piece in enumerate(pieces):
                text = make_text(title, piece if config == "B" else {"body": piece["body"]})
                rows.append((f"{config}:{doc_id}:c{k:03d}", config, doc_id, doc_type, source, title, url,
                             published_at, int(published_at[:4]), piece.get("section") if config == "B" else None,
                             piece.get("page") if config == "B" else None, k, text, len(piece["body"])))
                lens.append(len(piece["body"]))
                per_type[doc_type] += 1
        lens.sort()
        q = lambda p: lens[min(int(len(lens) * p), len(lens) - 1)]
        stats[config] = {
            "chunks": len(lens), "by_type": dict(per_type),
            "body_chars_p5_p50_p95": [q(0.05), q(0.5), q(0.95)], "min": lens[0], "max": lens[-1],
            "under_200": sum(l < 200 for l in lens), "over_1400": sum(l > 1400 for l in lens),
        }

    with conn:
        conn.execute("DELETE FROM chunks")
        conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO collect_log VALUES (datetime('now','localtime'), 'chunk', ?)",
                     (json.dumps({"params": {"target": TARGET_CHARS, "overlap_A": OVERLAP_CHARS, "max_B": MAX_CHARS,
                                             "news_whole_max_B": NEWS_WHOLE_MAX}, **stats}, ensure_ascii=False),))
    conn.close()
    return stats
