"""대한민국 정책브리핑(korea.kr) 보도자료 수집 (정책 1차 자료).

정부 부처가 직접 발표한 보도자료로, 반도체 지원정책·보조금·수출 실적 질문의 1차 근거가 된다.

사이트 조사 결과 (2026-11-03, 리포트 근거)
- robots.txt: 'User-Agent: * / Allow: /' → 수집 허용
- 보도자료 웹페이지 본문에는 "보도자료를 전재하여 제공" 한 줄만 있고,
  실제 내용은 첨부 파일에 있다 → 첨부를 내려받아 텍스트를 추출한다.
- 부처마다 첨부 형식이 다르다: 산업부는 PDF+HWPX, 과기정통부는 HWPX+ODT (PDF 없음)
  → PDF 를 우선 사용하고, PDF 가 없으면 HWPX 를 사용한다.
- 사이트 검색은 검색어 일부만 일치해도 결과에 포함되어 무관한 자료가 많다 → 2단계 필터 적용

수집 흐름
1) 목록  : SEARCH_KEYWORDS 로 검색 (기간 = 제공 뉴스와 동일) → newsId·제목·날짜·부처
2) 1차 필터: 소관 부처(TARGET_DEPTS)만 남김
3) 상세  : 첨부 PDF(없으면 HWPX) 다운로드 (data/raw/press/ 에 캐시)
4) 2차 필터: 첨부 본문에 '반도체'가 BODY_MIN_COUNT 회 이상인 자료만 저장 (doc_type='press')

PDF 텍스트 추출을 수집 단계에서 하는 이유: 2차 필터에 본문이 필요하기 때문.
추출된 텍스트의 정제(보일러플레이트 제거 등)는 indexer/parse.py 가 담당한다.

호출 위치: collector/__main__.py
"""
import io
import json
import re
import time
import zipfile
from xml.etree import ElementTree
from datetime import datetime

import pymupdf
import requests
from bs4 import BeautifulSoup

from collector.storage import ROOT, log_step, save_documents

BASE_URL = "https://www.korea.kr"
LIST_PATH = "/briefing/pressReleaseList.do"
VIEW_PATH = "/briefing/pressReleaseView.do"
CACHE_DIR = ROOT / "data" / "raw" / "press"
REQUEST_DELAY = 1.0  # 초
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mini-project3-research-crawler)"}

SEARCH_KEYWORDS = ["AI 반도체", "인공지능 반도체", "HBM"]
BEGIN_DATE = "2021-10-16"  # 제공 뉴스와 동일한 기간
END_DATE = "2026-10-16"

# 반도체 정책 소관 부처 (정부조직 개편으로 이름이 바뀐 경우 둘 다 포함)
TARGET_DEPTS = {
    "산업통상자원부", "산업통상부",
    "과학기술정보통신부",
    "기획재정부", "재정경제부",
}
BODY_KEYWORD = "반도체"

# 표준에서 약간 벗어난 PDF 를 열 때 MuPDF 가 출력하는 경고("syntax error: invalid key in dict" 등)를 숨긴다.
# 추출 자체에는 영향이 없음을 확인함 (2026-11-03, 캐시 PDF 300건 검사 결과 열기 실패 0건)
pymupdf.TOOLS.mupdf_display_errors(False)
BODY_MIN_COUNT = 3


# ---------------------------------------------------------------------------
# HTML 파싱 (네트워크 없이 테스트 가능한 순수 함수)
# ---------------------------------------------------------------------------
def _norm_date(text: str) -> str:
    """'2026-10-15' 또는 '2026.10.15' → '2026-10-15'."""
    return text.strip().replace(".", "-")


def parse_list(html: str) -> list[dict]:
    """목록 페이지에서 보도자료 항목(newsId, 제목, 날짜, 부처)을 추출한다."""
    soup = BeautifulSoup(html, "lxml")
    items = []
    for li in soup.select(".list_type li"):
        a = li.find("a", href=True)
        m = re.search(r"newsId=(\d+)", a["href"]) if a else None
        if not m:
            continue
        src = [s.get_text(strip=True) for s in li.select(".source > span")]
        items.append({
            "news_id": m.group(1),
            # 검색어 강조(<span class="highlight">)로 쪼개진 텍스트를 원래 제목 그대로 이어 붙임
            "title": re.sub(r"\s+", " ", li.find("strong").get_text()).strip(),
            "date": _norm_date(src[0]) if src else "",
            "dept": src[1] if len(src) > 1 else "",
        })
    return items


def parse_attachments(html: str) -> list[dict]:
    """상세 페이지에서 본문 추출용 첨부(fileId, 파일명, 형식, 다운로드 URL)를 고른다.

    - PDF 가 있으면 PDF 만 사용 (산업부: 같은 내용을 PDF·HWPX 로 함께 올림)
    - PDF 가 없으면 HWPX 사용 (과기정통부: HWPX·ODT 만 올림)
    - zip(사진·참고자료 묶음), ODT 등은 사용하지 않음
    """
    soup = BeautifulSoup(html, "lxml")
    found, seen = {"pdf": [], "hwpx": []}, set()
    for a in soup.select(".filedown a[href*='/common/download.do']"):
        name = a.get_text(" ", strip=True)
        m = re.search(r"fileId=(\d+)", a["href"])
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if not m or ext not in found or m.group(1) in seen:
            continue  # '내려받기' 버튼 등 파일명이 없는 링크, 중복 링크, 사용하지 않는 형식 제외
        seen.add(m.group(1))
        found[ext].append({"file_id": m.group(1), "name": name, "ext": ext,
                           "url": BASE_URL + a["href"].replace("&amp;", "&")})
    return found["pdf"] or found["hwpx"]


def is_pdf(data: bytes) -> bool:
    """내용이 실제 PDF 인지 확인한다.

    원본 첨부가 서버에서 삭제된 경우 PDF 대신 오류 안내 HTML 페이지가 내려온다 (2021년 자료 일부).
    """
    return data[:5] == b"%PDF-"


def pdf_to_text(pdf_bytes: bytes) -> str:
    """PDF 바이트에서 페이지 순서대로 텍스트를 추출한다."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def is_hwpx(data: bytes) -> bool:
    """내용이 HWPX(zip 형식) 인지 확인한다."""
    return data[:2] == b"PK"


def hwpx_to_text(hwpx_bytes: bytes) -> str:
    """HWPX 에서 본문 텍스트를 문단 단위로 추출한다.

    HWPX = 한글 문서 표준 형식(OWPML). zip 안의 Contents/section0.xml, section1.xml ... 에
    본문이 있고, 문단은 <hp:p>, 글자는 <hp:t> 태그에 들어 있다.
    표 안의 문단처럼 문단이 중첩될 수 있으므로, 트리를 순서대로 훑으며
    <t> 의 글자를 모으고 </p> 를 만날 때마다 줄을 바꾼다.
    """
    zf = zipfile.ZipFile(io.BytesIO(hwpx_bytes))
    sections = sorted(
        (n for n in zf.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)),
        key=lambda n: int(re.search(r"\d+", n).group()),
    )
    lines, buf = [], []
    for name in sections:
        for event, el in ElementTree.iterparse(io.BytesIO(zf.read(name)), events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]  # 네임스페이스 제거: '{...}t' → 't'
            if tag == "t" and el.text:
                buf.append(el.text)
            elif tag == "p":
                if buf:
                    lines.append("".join(buf).strip())
                    buf = []
    return "\n".join(l for l in lines if l).strip()


def attachment_to_text(ext: str, data: bytes) -> str:
    """첨부 형식에 맞는 추출 함수를 호출한다. 내용이 형식과 다르면 빈 문자열."""
    if ext == "pdf" and is_pdf(data):
        return pdf_to_text(data)
    if ext == "hwpx" and is_hwpx(data):
        return hwpx_to_text(data)
    return ""  # 원본이 삭제되어 오류 안내 HTML 이 내려온 경우 등


# ---------------------------------------------------------------------------
# 네트워크
# ---------------------------------------------------------------------------
def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET 요청 (요청 후 딜레이 적용)."""
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp


def fetch_list(keyword: str) -> list[dict]:
    """검색어 하나의 전체 결과를 페이지 끝까지 수집한다."""
    results, page = [], 1
    while True:
        html = _get(BASE_URL + LIST_PATH, {
            "pageIndex": page, "srchWord": keyword,
            "startDate": BEGIN_DATE, "endDate": END_DATE,
            "repCodeType": "", "repCode": "", "period": "",
        }).text
        items = parse_list(html)
        if not items:
            break
        results.extend(items)
        print(f"    '{keyword}' {page}페이지: 누적 {len(results)}건")
        page += 1
    return results


def fetch_attachments(news_id: str) -> list[dict]:
    """보도자료의 본문용 첨부를 내려받아 [{name, ext, bytes}] 로 반환한다 (캐시 사용).

    캐시: data/raw/press/{newsId}.json (첨부 목록) + {newsId}_{fileId}.{ext}
    HWPX 지원 이전에 '첨부 없음'으로 캐시된 항목은 상세 페이지를 다시 조회한다.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / f"{news_id}.json"
    files = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    if not files:
        files = parse_attachments(_get(BASE_URL + VIEW_PATH, {"newsId": news_id}).text)
        meta_path.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
    result = []
    for f in files:
        ext = f.get("ext", "pdf")  # 초기 캐시(PDF 전용)에는 ext 항목이 없음
        path = CACHE_DIR / f"{news_id}_{f['file_id']}.{ext}"
        if not path.exists():
            path.write_bytes(_get(f["url"]).content)
        result.append({"name": f["name"], "ext": ext, "bytes": path.read_bytes()})
    return result


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def to_row(item: dict, body: str, file_names: list[str], collected_at: str) -> dict:
    """보도자료 1건을 storage.save_documents 형식으로 변환한다."""
    return {
        "doc_id": f"press_{item['date'].replace('-', '')}_{item['news_id']}",
        "doc_type": "press",
        "source": item["dept"],
        "title": item["title"],
        "url": f"{BASE_URL}{VIEW_PATH}?newsId={item['news_id']}",
        "published_at": item["date"] + "T00:00:00",
        "collected_at": collected_at,
        "keyword": "; ".join(item["keywords"]),
        "body": body,
        "extra": json.dumps({"news_id": item["news_id"], "files": file_names}, ensure_ascii=False),
    }


def run() -> dict:
    """보도자료를 검색·필터링·다운로드하여 docs.db 에 저장하고 통계를 반환한다."""
    collected_at = datetime.now().isoformat(timespec="seconds")
    stats = {"listed": {}, "errors": []}

    # 1) 목록 수집 (검색어 간 중복은 newsId 로 병합, 어떤 검색어로 찾았는지 기록)
    merged: dict[str, dict] = {}
    for kw in SEARCH_KEYWORDS:
        items = fetch_list(kw)
        stats["listed"][kw] = len(items)
        for it in items:
            merged.setdefault(it["news_id"], {**it, "keywords": []})["keywords"].append(kw)
    stats["unique"] = len(merged)

    # 2) 1차 필터: 소관 부처
    targets = [it for it in merged.values() if it["dept"] in TARGET_DEPTS]
    stats["after_dept"] = len(targets)

    # 3)~4) 첨부 다운로드 → 텍스트 추출 → 2차 필터
    rows, no_file, low_relevance = [], 0, 0
    for i, it in enumerate(targets, 1):
        try:
            files = fetch_attachments(it["news_id"])
            texts = [(f["name"], attachment_to_text(f["ext"], f["bytes"])) for f in files]
            texts = [(n, t) for n, t in texts if t]
            if not texts:  # 본문용 첨부가 없거나, 내려받은 파일이 형식과 다른 경우
                no_file += 1
                continue
            body = "\n\n".join(t for _, t in texts)
            if body.count(BODY_KEYWORD) < BODY_MIN_COUNT:
                low_relevance += 1
                continue
            rows.append(to_row(it, body, [n for n, _ in texts], collected_at))
        except Exception as e:  # 한 건 실패가 전체 수집을 멈추지 않도록 기록 후 계속
            stats["errors"].append(f"{it['news_id']}: {type(e).__name__}: {e}"[:200])
        if i % 20 == 0:
            print(f"    상세 {i}/{len(targets)} 처리, 저장 대상 {len(rows)}건")

    stats.update(no_file=no_file, low_relevance=low_relevance)
    stats["saved"] = save_documents(rows, replace_type="press")
    log_step("crawl_korea_kr", stats)
    return stats
