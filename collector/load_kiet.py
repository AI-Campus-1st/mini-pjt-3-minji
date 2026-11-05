"""산업연구원(KIET) 보고서 적재 (산업 분석 1차 자료).

수집 방식: 수동 다운로드 + 로더
- KIET 의 보고서 파일 경로(/common/file/)는 robots.txt 에서 Disallow 로 지정되어 있어
  자동 수집하지 않는다 (2026-11-03 확인, 리포트의 '접근 제약 대응' 근거).
- 목록 페이지(/research/...)는 허용 → 후보 목록만 확인하고, PDF 는 사용자가 직접 내려받아
  data/raw/kiet/ 에 저장한다.

이 모듈은 data/raw/kiet/*.pdf 를 KIET_REPORTS 의 메타데이터(제목·발간일·원문 링크)와 연결해
store/docs.db 에 저장한다 (doc_type='report').

본문 형식: 페이지 사이를 폼피드 문자(\\f)로 구분해 저장한다.
→ indexer 에서 청크의 쪽 번호를 복원해 출처에 'p.12' 처럼 표시할 수 있게 하기 위함.

호출 위치: collector/__main__.py
"""
import glob
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import pymupdf

from collector.storage import ROOT, log_step, save_documents

KIET_DIR = ROOT / "data" / "raw" / "kiet"
SOURCE_NAME = "산업연구원(KIET)"
BASE_URL = "https://www.kiet.re.kr"

pymupdf.TOOLS.mupdf_display_errors(False)  # 비표준 PDF 경고 숨김 (추출에는 영향 없음)

# 수동 다운로드한 보고서 목록 (KIET 목록 페이지에서 확인한 메타데이터)
# match: 파일명(NFC)에 포함된 고유 문자열 — 다운로드 파일명을 그대로 두어도 연결되도록
KIET_REPORTS = [
    {"match": "첨단산업의_한·중_경쟁력", "id": "issue_844", "date": "2026-02-24", "series": "i-KIET 산업경제이슈",
     "title": "첨단산업의 한·중 경쟁력 분석과 정책 방향", "path": "/research/issueView?issue_no=844"},
    {"match": "한일_산업_협력", "id": "report_1166", "date": "2025-12-31", "series": "연구보고서",
     "title": "경제안보 시대 한일 산업협력 방안 연구: 반도체와 이차전지를 중심으로", "path": "/research/reportView?report_no=1166"},
    {"match": "대중_반도체_수출통제", "id": "paper_806", "date": "2024-03-26", "series": "이슈페이퍼",
     "title": "미국의 대중 반도체 수출통제에 따른 중국의 공급망 영향과 시사점", "path": "/research/paperView?paper_no=806"},
    {"match": "대만_반도체산업", "id": "paper_801", "date": "2024-02-26", "series": "이슈페이퍼",
     "title": "대만 반도체산업 분석 및 정책적 시사점", "path": "/research/paperView?paper_no=801"},
    {"match": "시스템반도체_산업의_글로벌", "id": "podata_357", "date": "2023-12-05", "series": "정책자료",
     "title": "시스템반도체 산업의 글로벌 지형 및 정책 방안 연구", "path": "/research/podataView?podata_no=357"},
    {"match": "반도체와_과학법", "id": "issue_779", "date": "2022-08-04", "series": "i-KIET 산업경제이슈",
     "title": "미국 '반도체와 과학법'의 정책적 시사점", "path": "/research/issueView?issue_no=779"},
    {"match": "글로벌_반도체_공급망_재편", "id": "issue_775", "date": "2022-04-29", "series": "i-KIET 산업경제이슈",
     "title": "글로벌 반도체 공급망 재편 움직임과 정책적 시사점", "path": "/research/issueView?issue_no=775"},
]


def pdf_pages_text(path: str) -> tuple[str, int]:
    """PDF 전체 텍스트(페이지 구분자 \\f)와 페이지 수를 반환한다."""
    with pymupdf.open(path) as doc:
        # 일부 PDF 는 페이지 텍스트 안에 \f 가 원래 들어 있어 쪽 구분이 어긋난다 → 먼저 제거
        pages = [page.get_text().replace("\f", "\n") for page in doc]
        return "\f".join(pages), doc.page_count


def match_report(file_name: str) -> dict | None:
    """파일명에 해당하는 보고서 메타데이터를 찾는다 (macOS NFD 파일명 대비 NFC 정규화)."""
    name = unicodedata.normalize("NFC", file_name)
    hits = [r for r in KIET_REPORTS if r["match"] in name]
    return hits[0] if len(hits) == 1 else None


def run() -> dict:
    """data/raw/kiet/*.pdf 를 읽어 docs.db 에 저장하고 통계를 반환한다."""
    collected_at = datetime.now().isoformat(timespec="seconds")
    rows, stats = [], {"unmatched_files": [], "missing_reports": []}
    found_ids = set()

    for path in sorted(glob.glob(str(KIET_DIR / "*.pdf"))):
        file_name = Path(path).name
        meta = match_report(file_name)
        if not meta:
            stats["unmatched_files"].append(unicodedata.normalize("NFC", file_name))
            continue
        body, pages = pdf_pages_text(path)
        found_ids.add(meta["id"])
        rows.append({
            "doc_id": f"report_{meta['date'].replace('-', '')}_{meta['id']}",
            "doc_type": "report",
            "source": SOURCE_NAME,
            "title": meta["title"],
            "url": BASE_URL + meta["path"],
            "published_at": meta["date"] + "T00:00:00",
            "collected_at": collected_at,
            "keyword": meta["series"],
            "body": body,
            "extra": json.dumps({"file_name": unicodedata.normalize("NFC", file_name),
                                 "pages": pages, "collection": "manual_download"}, ensure_ascii=False),
        })

    stats["missing_reports"] = [r["title"] for r in KIET_REPORTS if r["id"] not in found_ids]
    stats["saved"] = save_documents(rows, replace_type="report")
    stats["pages"] = {r["doc_id"]: json.loads(r["extra"])["pages"] for r in rows}
    log_step("load_kiet", stats)
    return stats
