"""제공 데이터(AI 반도체 뉴스 CSV) 적재 및 선별.

입력: data/raw/02_AI반도체_묶음_*.csv (find_raw_file 로 탐색)
      컬럼 = 제목, 본문전문, 원문URL, 발행일, 매체명, 문서종류, 수집키워드
출력: store/docs.db 의 documents 테이블 (doc_type='news')

선별 기준 (EDA 근거: notebooks/eda.ipynb)
- 기간: 제공 범위 5년 전체 사용 (2021-10 ~ 2026-10)
- 본문 결측 제외
- 관련성 필터: 제목에 반도체 관련어가 있거나, 본문에 'AI 반도체/인공지능 반도체'가 2회 이상 등장
  → 키워드가 한 번 스치듯 언급된 증시 시황·정치 기사 등을 제외하기 위함
- 중복 처리는 여기서 하지 않는다 (indexer/dedup.py 담당, 처리 전후 건수를 그쪽에서 기록)

호출 위치: collector/__main__.py
"""
import glob
import unicodedata
from datetime import datetime

import pandas as pd

from collector.storage import ROOT, log_step, save_documents

RAW_DIR = ROOT / "data" / "raw"
RAW_NAME_PREFIX = "02_AI반도체_묶음_"


def find_raw_file() -> str:
    """제공 CSV 경로를 찾는다.

    macOS 는 한글 파일명을 자모 분리형(NFD)으로 저장하므로, 코드의 문자열(NFC)과
    그대로 비교하면 일치하지 않는다. 양쪽을 NFC 로 정규화한 뒤 비교한다.
    """
    for f in glob.glob(str(RAW_DIR / "*.csv")):
        name = unicodedata.normalize("NFC", f.split("/")[-1])
        if name.startswith(RAW_NAME_PREFIX):
            return f
    raise FileNotFoundError(f"제공 데이터 CSV 가 없습니다: {RAW_DIR}/{RAW_NAME_PREFIX}*.csv")

# 관련성 필터 규칙 (값을 바꾸면 EDA 노트북과 리포트 근거도 함께 갱신할 것)
# - 국내 AI 반도체 스타트업명 추가: 제목에 '반도체'가 없어도 업계 핵심 기사인 경우가 많음
#   (엔비디아·TSMC 는 증시 시황 기사가 대량 유입되어 제외 — 샘플 검토 결과)
TITLE_PATTERN = r"반도체|HBM|NPU|칩|AI 가속기|퓨리오사|리벨리온|딥엑스|모빌린트"
BODY_KEYWORD_PATTERN = r"AI ?반도체|인공지능 ?반도체"
BODY_MIN_COUNT = 2


def relevance_mask(df: pd.DataFrame) -> pd.Series:
    """관련성 필터를 통과하는 행이면 True 인 시리즈를 반환한다."""
    title_hit = df["제목"].str.contains(TITLE_PATTERN, regex=True)
    body_cnt = df["본문전문"].str.count(BODY_KEYWORD_PATTERN)
    return title_hit | (body_cnt >= BODY_MIN_COUNT)


def to_rows(df: pd.DataFrame, collected_at: str) -> list[dict]:
    """DataFrame 을 storage.save_documents 형식의 dict 목록으로 변환한다.

    doc_id 형식: news_YYYYMMDD_원본행번호(6자리)
    - 원본 CSV 행 번호를 쓰므로 같은 CSV 로 재실행하면 동일한 ID 가 나온다
      (평가셋 gold_doc_ids 가 재실행 후에도 유효하도록)
    """
    rows = []
    for idx, r in df.iterrows():
        date = pd.to_datetime(r["발행일"])
        rows.append({
            "doc_id": f"news_{date:%Y%m%d}_{idx:06d}",
            "doc_type": "news",
            "source": r["매체명"],
            "title": r["제목"],
            "url": r["원문URL"],
            "published_at": date.isoformat(),
            "collected_at": collected_at,
            "keyword": r["수집키워드"],
            "body": r["본문전문"],
            "extra": None,
        })
    return rows


def run() -> dict:
    """제공 CSV 를 읽어 선별 후 docs.db 에 저장하고 단계별 건수를 반환한다."""
    path = find_raw_file()
    df = pd.read_csv(path, encoding="utf-8-sig")
    stats = {"source_file": unicodedata.normalize("NFC", path.split("/")[-1]), "raw": len(df)}

    # 1) 본문 결측 제외
    df = df[df["본문전문"].notna() & (df["본문전문"].str.strip() != "")]
    stats["after_drop_empty"] = len(df)

    # 2) 관련성 필터
    df = df[relevance_mask(df)]
    stats["after_relevance"] = len(df)
    stats["by_year"] = pd.to_datetime(df["발행일"]).dt.year.value_counts().sort_index().to_dict()
    stats["by_year"] = {str(k): int(v) for k, v in stats["by_year"].items()}

    # 3) 저장 (news 문서를 교체 저장 → 재실행해도 누적되지 않음)
    # 수집 일시: 제공 데이터는 실제 수집 시점이 아닌 적재 시점을 기록
    collected_at = datetime.now().isoformat(timespec="seconds")
    stats["saved"] = save_documents(to_rows(df, collected_at), replace_type="news")

    log_step("load_provided", stats)
    return stats
