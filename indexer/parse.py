"""② 인덱싱 1단계 — 텍스트 정제 (보일러플레이트 제거).

입력: store/docs.db 의 documents (원문, 수집 계층이 저장)
출력: store/docs.db 의 parsed_documents (정제 본문) — 원문(documents)은 그대로 둔다.
      → 챗봇의 '원문 확인'은 documents 를, 인덱싱(중복 처리·청킹·임베딩)은 parsed_documents 를 쓴다.

정제 규칙은 Day 2 문서 탐색 결과(반복 줄·반복 구절 빈도)를 근거로 정했다.
| 문서 종류 | 제거 대상 |
|---|---|
| 공통   | 제어 문자(PDF 글꼴 깨짐 잔여물), 연속 공백·빈 줄 |
| news   | 기사 끝 안내문("기사를 더 보고 싶다면?", "기사 섹션 분류 안내", 제보·구독 안내 등), 첫머리 발신지 표기 |
| press  | 머리말(보도자료/보도시점/배포 + 날짜), 담당 부서·연락처 블록, 쪽 번호("- 3 -"), 목차 줄(점선 + 쪽 번호) |
| dart   | 이미지 파일명만 남은 줄("연구개발조직도.jpg"), DART 줄바꿈 표기 `&cr;` → 줄바꿈 |
| report | 쪽마다 반복되는 머리말·꼬리말(숫자만 다른 줄 포함), 쪽 번호만 있는 줄, 목차 줄(점선 + 쪽 번호) |

글자가 깨진 문서(정제 후에도 제어 문자 비율이 GARBLED_RATIO 이상)는 status='garbled' 로 표시해 인덱싱에서 제외한다.

실행: python -m indexer --only parse   (indexer/__main__.py)
"""
import collections
import json
import re

from collector.storage import get_conn

GARBLED_RATIO = 0.05
CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")   # \t \n \f(쪽 구분자) 는 남긴다

# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def normalize_ws(text: str) -> str:
    """줄 안의 연속 공백을 하나로, 빈 줄은 최대 1개로 줄인다 (쪽 구분자 \\f 는 유지)."""
    text = re.sub(r"[ \t 　]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 목차 줄: "<표 2-6> 일본 정부의 반도체 기업 지원 현황·········37" 처럼 점선(5개 이상) + 쪽 번호로 끝나는 줄
TOC_LINE = re.compile(r"[·\.…ㆍ]{5,}\s*\d*\s*$")


def drop_toc_lines(text: str) -> tuple[str, bool]:
    """목차 줄을 지운다. 본문 내용이 없어 검색 결과에 끼어들면 방해만 되기 때문 (청킹 샘플 확인 결과)."""
    lines = text.split("\n")
    kept = [l for l in lines if not TOC_LINE.search(l)]
    return "\n".join(kept), len(kept) < len(lines)


# ---------------------------------------------------------------------------
# news: 줄바꿈 없이 한 덩어리 → 끝부분 안내문을 '시작 지점부터 끝까지' 잘라낸다
# ---------------------------------------------------------------------------
NEWS_TAIL_TRIGGERS = [
    "기사를 더 보고 싶다면", "기사 섹션 분류 안내", "이 기사를 추천합니다",
    "제보가 뉴스로 만들어집니다", "여러분의 제보를 기다", "제보를 기다립니다",
    "모든 얘깃거리를 알려주세요", "구독해주세요", "제보하기", "카톡 라인 jebo23",
    "All rights reserved", "무단전재", "재배포 금지", "Copyright",
]
NEWS_TAIL_WINDOW = 400   # 끝에서 이 글자 수 안에서만 찾는다 (본문 중간 인용을 지우지 않도록)
NEWS_DATELINE = re.compile(r"^\(\s*[^()]{1,12}?\s*(?:뉴스1|연합뉴스|뉴시스)\s*\)\s*")


def clean_news(body: str) -> tuple[str, list[str]]:
    removed = []
    m = NEWS_DATELINE.match(body)
    if m:
        removed.append("dateline")
        body = body[m.end():]
    tail_start = len(body) - NEWS_TAIL_WINDOW
    cut = min((i for t in NEWS_TAIL_TRIGGERS if (i := body.find(t, max(tail_start, 0))) >= 0), default=None)
    if cut is not None:
        removed.append("tail_notice")
        body = body[:cut]
    return body, removed


# ---------------------------------------------------------------------------
# press: 머리말 · 연락처 블록 · 쪽 번호
# ---------------------------------------------------------------------------
PAGE_MARK = re.compile(r"^\s*-\s*\d+\s*-\s*$")
PHONE = re.compile(r"\(?0\d{1,2}-\d{3,4}-\d{4}\)?")


def clean_press(body: str) -> tuple[str, list[str]]:
    removed = []
    lines = body.split("\n")

    # 1) 머리말: 앞 12줄 안의 '배포' 줄과 그다음 날짜 줄까지 제거
    for i, l in enumerate(lines[:12]):
        if l.strip() == "배포":
            end = i + 2 if i + 1 < len(lines) and re.match(r"^\s*\d", lines[i + 1]) else i + 1
            lines = lines[end:]
            removed.append("header")
            break

    # 2) 연락처 블록: '담당 부서' 줄부터, 40줄 안의 마지막 전화번호 줄까지 제거
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() in ("담당 부서", "담당부서"):
            window = lines[i:i + 40]
            last_phone = max((j for j, l in enumerate(window) if PHONE.search(l)), default=None)
            if last_phone is not None:
                i += last_phone + 1
                removed.append("contact_block")
                continue
        out.append(lines[i])
        i += 1

    # 3) 쪽 번호 줄
    n = len(out)
    out = [l for l in out if not PAGE_MARK.match(l)]
    if len(out) < n:
        removed.append("page_marks")
    text, had_toc = drop_toc_lines("\n".join(out))
    if had_toc:
        removed.append("toc_lines")
    return text, removed


# ---------------------------------------------------------------------------
# dart: 이미지 파일명 줄
# ---------------------------------------------------------------------------
IMAGE_LINE = re.compile(r"^\s*\S.*\.(?:jpg|jpeg|png|gif|bmp)\s*$", re.I)


def clean_dart(body: str) -> tuple[str, list[str]]:
    removed = []
    if "&cr;" in body:   # DART 원문 XML 의 줄바꿈 표기가 글자로 남은 경우
        body = body.replace("&cr;", "\n")
        removed.append("cr_entity")
    lines = body.split("\n")
    kept = [l for l in lines if not IMAGE_LINE.match(l)]
    if len(kept) < len(lines):
        removed.append("image_lines")
    return "\n".join(kept), removed


# ---------------------------------------------------------------------------
# report: 쪽마다 반복되는 머리말·꼬리말 (문서별로 자동 탐지)
# ---------------------------------------------------------------------------
def _shape(line: str) -> str:
    """숫자를 지운 줄 모양 — '20 대만 반도체산업 분석' 과 '21 대만 반도체산업 분석' 을 같은 머리말로 본다."""
    return re.sub(r"\d+", "#", line.strip())


def clean_report(body: str, edge_lines: int = 2, min_share: float = 0.15) -> tuple[str, list[str]]:
    pages = body.split("\f")
    cnt = collections.Counter()
    for p in pages:
        ls = [l for l in p.split("\n") if l.strip()]
        cnt.update({_shape(l) for l in ls[:edge_lines] + ls[-edge_lines:]})
    repeated = {s for s, n in cnt.items() if n >= max(3, len(pages) * min_share) and len(s) <= 60}

    removed, new_pages = set(), []
    for p in pages:
        ls = p.split("\n")
        idx = [k for k, l in enumerate(ls) if l.strip()]
        edges = set(idx[:edge_lines] + idx[-edge_lines:])
        kept = []
        for k, l in enumerate(ls):
            if k in edges and (_shape(l) in repeated or re.fullmatch(r"\s*\d{1,3}\s*", l)):
                removed.add("running_header" if _shape(l) in repeated else "page_number")
                continue
            kept.append(l)
        page_text, had_toc = drop_toc_lines("\n".join(kept))
        if had_toc:
            removed.add("toc_lines")
        new_pages.append(page_text)
    return "\f".join(new_pages), sorted(removed)


CLEANERS = {"news": clean_news, "press": clean_press, "dart": clean_dart, "report": clean_report}


def clean(doc_type: str, body: str) -> tuple[str, list[str], str]:
    """문서 한 건을 정제한다. 반환: (정제 본문, 적용된 제거 규칙 목록, 상태 'ok'|'garbled'|'empty')."""
    n_ctrl = len(CTRL.findall(body))
    body, removed = CLEANERS[doc_type](body)
    if n_ctrl:
        removed.append("control_chars")
    if n_ctrl / max(len(body), 1) >= GARBLED_RATIO:
        status = "garbled"
    else:
        status = "ok"
    body = normalize_ws(CTRL.sub("", body))
    if not body:
        status = "empty"
    return body, removed, status


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS parsed_documents (
    doc_id       TEXT PRIMARY KEY,
    doc_type     TEXT NOT NULL,
    body         TEXT NOT NULL,
    chars_before INTEGER,
    chars_after  INTEGER,
    removed      TEXT,       -- 적용된 제거 규칙 (JSON 목록)
    status       TEXT        -- ok / garbled / empty → ok 만 인덱싱
);
"""


def run() -> dict:
    """documents 전체를 정제해 parsed_documents 에 저장하고 통계를 반환한다."""
    conn = get_conn()
    conn.executescript(SCHEMA)
    rows, stats = [], {"by_type": {}, "rules": collections.Counter(), "status": collections.Counter(), "excluded": []}
    for doc_id, doc_type, body in conn.execute("select doc_id, doc_type, body from documents"):
        text, removed, status = clean(doc_type, body)
        rows.append((doc_id, doc_type, text, len(body), len(text), json.dumps(removed), status))
        t = stats["by_type"].setdefault(doc_type, {"docs": 0, "chars_before": 0, "chars_after": 0})
        t["docs"] += 1
        t["chars_before"] += len(body)
        t["chars_after"] += len(text)
        stats["rules"].update(f"{doc_type}:{r}" for r in removed)
        stats["status"][status] += 1
        if status != "ok":
            stats["excluded"].append(f"{doc_id} ({status})")
    with conn:
        conn.execute("DELETE FROM parsed_documents")
        conn.executemany("INSERT INTO parsed_documents VALUES (?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO collect_log VALUES (datetime('now','localtime'), 'parse', ?)",
                     (json.dumps({**stats, "rules": dict(stats["rules"]), "status": dict(stats["status"])}, ensure_ascii=False),))
    conn.close()
    stats["rules"], stats["status"] = dict(stats["rules"]), dict(stats["status"])
    return stats
