"""DART OpenAPI 수집 (주제 B 1차 자료: 상장사 사업보고서).

기업이 직접 공시한 사업보고서의 'II. 사업의 내용' 섹션을 수집한다.
뉴스(제3자 보도)와 달리 기업의 공식 입장·투자 계획을 확인하는 1차 근거로 쓰인다.

수집 흐름
1) corpCode.xml : 종목코드 → DART 고유번호(corp_code) 변환
2) list.json    : 기업별 정기공시(사업보고서) 목록 조회
3) document.xml : 공시 원문(zip 안의 XML) 다운로드 → 'II. 사업의 내용' 추출
4) storage.save_documents 로 store/docs.db 에 저장 (doc_type='dart')

준수 사항
- 인증키는 .env 의 DART_API_KEY 에서 읽는다 (코드·로그에 키를 출력하지 않음)
- 요청 간 REQUEST_DELAY 초 대기, 병렬 요청 없음
- 원문 zip 은 data/raw/dart/ 에 캐시 → 재실행 시 다시 내려받지 않음

호출 위치: collector/__main__.py
"""
import io
import json
import os
import re
import time
import warnings
import zipfile
from datetime import datetime

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

from collector.storage import ROOT, log_step, save_documents

API_BASE = "https://opendart.fss.or.kr/api"
VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"  # 사용자가 원문을 여는 링크
CACHE_DIR = ROOT / "data" / "raw" / "dart"
REQUEST_DELAY = 1.0  # 초

# 수집 대상 기업 (종목코드: 표시명). AI 반도체 밸류체인별로 선정 (뉴스 언급 수 근거, 2026-11-03)
# 종목코드가 틀리면 fetch_corp_codes 에서 오류로 알려준다.
TARGET_COMPANIES = {
    # 메모리
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    # 장비 (HBM TC본더)
    "042700": "한미반도체",
    # 설계 IP (NPU)
    "394280": "오픈엣지테크놀로지",
    # 설계 서비스 (디자인하우스)
    "399720": "가온칩스",
    # 기판 (FC-BGA)
    "009150": "삼성전기",
    # PCB (AI 가속기용 고다층 기판)
    "007660": "이수페타시스",
}

# 제공 뉴스와 같은 기간 (사업보고서는 매년 3월 제출 → 2021~2025 회계연도분)
BEGIN_DATE = "20211016"
END_DATE = "20261016"

TARGET_SECTION = "사업의 내용"

# DART 원문 XML 은 표준을 벗어난 태그·인코딩이 섞여 있어 엄격한 XML 파서로는 깨지는 경우가 있다.
# 그래서 관대한 HTML 파서(lxml)를 의도적으로 사용하며, 그에 따른 경고는 숨긴다.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# API 호출
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """프로젝트 루트 .env 에서 DART_API_KEY 를 읽는다."""
    load_dotenv(ROOT / ".env")
    key = os.getenv("DART_API_KEY", "").strip()
    if not key:
        raise RuntimeError(".env 에 DART_API_KEY 가 없습니다. 'DART_API_KEY=발급키' 형식으로 저장하세요.")
    return key


def _get(endpoint: str, params: dict) -> requests.Response:
    """DART API GET 요청 (요청 후 딜레이 적용)."""
    resp = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp


def _check_zip(content: bytes, what: str) -> zipfile.ZipFile:
    """응답이 zip 인지 확인한다.

    DART 는 오류 시 zip 대신 상태 코드가 담긴 XML/JSON 을 돌려주므로,
    zip 이 아니면 그 내용을 오류 메시지로 보여준다.
    """
    if not content.startswith(b"PK"):
        raise RuntimeError(f"{what} 응답이 zip 이 아닙니다: {content[:300].decode('utf-8', 'ignore')}")
    return zipfile.ZipFile(io.BytesIO(content))


def fetch_corp_codes(api_key: str, stock_codes: list[str]) -> dict[str, str]:
    """종목코드 목록 → {종목코드: corp_code} 를 반환한다."""
    resp = _get("corpCode.xml", {"crtfc_key": api_key})
    zf = _check_zip(resp.content, "corpCode.xml")
    xml = zf.read(zf.namelist()[0]).decode("utf-8")
    soup = BeautifulSoup(xml, "lxml-xml")
    result = {}
    for item in soup.find_all("list"):
        stock = (item.stock_code.text or "").strip()
        if stock in stock_codes:
            result[stock] = item.corp_code.text.strip()
    missing = set(stock_codes) - set(result)
    if missing:
        raise RuntimeError(f"corp_code 를 찾지 못한 종목코드: {missing}")
    return result


def fetch_annual_reports(api_key: str, corp_code: str) -> list[dict]:
    """기업의 사업보고서 공시 목록을 조회한다 (정정공시 포함 원본 목록)."""
    params = {
        "crtfc_key": api_key, "corp_code": corp_code,
        "bgn_de": BEGIN_DATE, "end_de": END_DATE,
        "pblntf_ty": "A", "pblntf_detail_ty": "A001",  # 정기공시 > 사업보고서
        "page_count": 100,
    }
    data = _get("list.json", params).json()
    if data.get("status") == "013":  # 조회된 데이터 없음
        return []
    if data.get("status") != "000":
        raise RuntimeError(f"list.json 오류: {data.get('status')} {data.get('message')}")
    return data.get("list", [])


def fetch_document_zip(api_key: str, rcept_no: str) -> bytes:
    """공시 원문 zip 을 반환한다. 캐시가 있으면 캐시를 사용한다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{rcept_no}.zip"
    if cache.exists():
        return cache.read_bytes()
    content = _get("document.xml", {"crtfc_key": api_key, "rcept_no": rcept_no}).content
    _check_zip(content, f"document.xml({rcept_no})")
    cache.write_bytes(content)
    return content


# ---------------------------------------------------------------------------
# 목록 정리·원문 파싱 (네트워크 없이 테스트 가능한 순수 함수)
# ---------------------------------------------------------------------------
def pick_latest_per_period(reports: list[dict]) -> list[dict]:
    """같은 회계기간 보고서 중 가장 최근 접수본(정정본 우선)만 남긴다.

    report_nm 예: '사업보고서 (2024.12)', '[기재정정]사업보고서 (2024.12)'
    정정 전 원본과 정정본이 모두 남으면 같은 내용이 중복 검색되므로 최신 1건만 유지한다.
    """
    latest = {}
    for r in reports:
        m = re.search(r"사업보고서\s*\((\d{4}\.\d{2})\)", r.get("report_nm", ""))
        if not m:
            continue  # 첨부 추가 등 본문이 아닌 공시 제외
        period = m.group(1)
        if period not in latest or r["rcept_no"] > latest[period]["rcept_no"]:
            latest[period] = {**r, "period": period}
    return sorted(latest.values(), key=lambda x: x["period"])


def _decode(raw: bytes) -> str:
    """DART XML 인코딩(UTF-8 또는 CP949)을 판별해 문자열로 변환한다."""
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def extract_section(zip_bytes: bytes, rcept_no: str, section: str = TARGET_SECTION) -> str:
    """공시 원문 zip 에서 대상 섹션(SECTION-1) 텍스트를 추출한다.

    - zip 에는 본문({rcept_no}.xml)과 첨부 XML 이 함께 들어 있으므로 본문을 우선 사용
    - 하위 제목(TITLE)은 '## 제목' 형태로 남긴다 → indexer/chunk.py 에서 헤딩 기준 분할에 사용
    """
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    # 본문 파일명은 정확히 '{접수번호}.xml'. 첨부는 '{접수번호}_00760.xml' 처럼 접미사가 붙고,
    # zip 내부 순서가 일정하지 않으므로 startswith 가 아닌 정확한 이름으로 찾는다.
    # (정확히 일치하는 파일이 없으면 가장 큰 파일을 본문으로 간주)
    main = f"{rcept_no}.xml"
    if main not in names:
        main = max(names, key=lambda n: zf.getinfo(n).file_size)
    soup = BeautifulSoup(_decode(zf.read(main)), "lxml")  # html 파서: 태그명이 소문자로 정규화됨

    for sec in soup.find_all("section-1"):
        title = sec.find("title")
        if title and section in title.get_text():
            for t in sec.find_all("title"):
                t.replace_with(f"\n## {t.get_text(strip=True)}\n")
            for row in sec.find_all("tr"):  # 표의 행은 한 줄로 이어 붙여 셀 구분을 유지
                row.replace_with(" | ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th", "te", "tu"])) + "\n")
            text = sec.get_text("\n")
            text = re.sub(r"[ \t\xa0]+", " ", text)
            return re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return ""


def to_row(report: dict, stock_code: str, body: str, collected_at: str) -> dict:
    """공시 1건을 storage.save_documents 형식으로 변환한다."""
    rcept_dt = report["rcept_dt"]  # YYYYMMDD
    return {
        "doc_id": f"dart_{rcept_dt}_{report['rcept_no']}",
        "doc_type": "dart",
        "source": report["corp_name"],
        "title": f"{report['corp_name']} {report['report_nm'].strip()} - II. 사업의 내용",
        "url": VIEWER_URL.format(rcept_no=report["rcept_no"]),
        "published_at": datetime.strptime(rcept_dt, "%Y%m%d").isoformat(),
        "collected_at": collected_at,
        "keyword": "사업보고서",
        "body": body,
        "extra": json.dumps({
            "rcept_no": report["rcept_no"], "report_nm": report["report_nm"],
            "period": report["period"], "stock_code": stock_code,
            "section": TARGET_SECTION,
        }, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run() -> dict:
    """대상 기업의 사업보고서를 수집해 docs.db 에 저장하고 통계를 반환한다."""
    api_key = get_api_key()
    corp_codes = fetch_corp_codes(api_key, list(TARGET_COMPANIES))
    collected_at = datetime.now().isoformat(timespec="seconds")

    rows, stats = [], {"companies": {}, "empty_section": []}
    for stock, corp_code in corp_codes.items():
        reports = pick_latest_per_period(fetch_annual_reports(api_key, corp_code))
        stats["companies"][TARGET_COMPANIES[stock]] = [r["period"] for r in reports]
        for r in reports:
            print(f"  - {r['corp_name']} {r['report_nm'].strip()} ({r['rcept_no']})")
            body = extract_section(fetch_document_zip(api_key, r["rcept_no"]), r["rcept_no"])
            if not body:
                stats["empty_section"].append(r["rcept_no"])  # 섹션 구조가 달라 추출 실패한 공시
                continue
            rows.append(to_row(r, stock, body, collected_at))

    stats["saved"] = save_documents(rows, replace_type="dart")
    stats["body_chars"] = {r["doc_id"]: len(r["body"]) for r in rows}
    log_step("crawl_dart", stats)
    return stats
