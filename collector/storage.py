"""원문 저장 모듈 (SQLite: store/docs.db).

수집 계층이 모은 문서의 원문과 메타데이터를 한 곳에 저장한다.
벡터 DB에는 청크만 들어가므로, 원문 확인이 필요할 때는 이 DB를 조회한다.

사용 위치
- collector/load_provided.py, collector/crawl_*.py : 문서 저장 (save_documents)
- indexer/                                          : 원문 읽기 (load_documents)
- app/                                              : 원문 확인 영역 (조회 전용)
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 기준 경로 (어느 위치에서 실행해도 동일한 파일을 가리키도록 고정)
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "store" / "docs.db"

# 문서 테이블 스키마
# - doc_id      : 문서 고유 ID (평가셋 gold_doc_ids 에서 이 값을 참조)
# - doc_type    : 문서 종류 (news / dart / press / report 등) → 출처 신뢰도 구분에 사용
# - source      : 발행처 (매체명, 기관명, 기업명)
# - published_at: 발행일 (ISO 문자열) → 시점 질문, 기간 필터에 사용
# - collected_at: 수집 일시 (가이드 크롤링 준수 사항)
# - extra       : 문서 종류별 부가 정보 (JSON 문자열)
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    doc_type     TEXT NOT NULL,
    source       TEXT,
    title        TEXT NOT NULL,
    url          TEXT,
    published_at TEXT,
    collected_at TEXT NOT NULL,
    keyword      TEXT,
    body         TEXT NOT NULL,
    extra        TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(published_at);

-- 수집 실행 기록 (처리 전후 건수 등 리포트 근거)
CREATE TABLE IF NOT EXISTS collect_log (
    run_at  TEXT NOT NULL,
    step    TEXT NOT NULL,
    stats   TEXT NOT NULL
);
"""

COLUMNS = ["doc_id", "doc_type", "source", "title", "url",
           "published_at", "collected_at", "keyword", "body", "extra"]


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB 연결을 열고 테이블이 없으면 생성한다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def save_documents(rows: list[dict], replace_type: str | None = None) -> int:
    """문서 목록을 저장한다.

    Args:
        rows: COLUMNS 키를 가진 dict 목록
        replace_type: 지정하면 해당 doc_type 문서를 먼저 삭제한 뒤 저장한다.
            (같은 수집기를 다시 실행해도 결과가 중복 누적되지 않도록 하기 위함)
    Returns:
        저장된 문서 수
    """
    conn = get_conn()
    with conn:
        if replace_type:
            conn.execute("DELETE FROM documents WHERE doc_type = ?", (replace_type,))
        conn.executemany(
            f"INSERT OR REPLACE INTO documents ({','.join(COLUMNS)}) "
            f"VALUES ({','.join('?' * len(COLUMNS))})",
            [tuple(r.get(c) for c in COLUMNS) for r in rows],
        )
    conn.close()
    return len(rows)


def log_step(step: str, stats: dict) -> None:
    """수집 단계별 통계를 collect_log 테이블에 기록한다."""
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO collect_log VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), step,
             json.dumps(stats, ensure_ascii=False)),
        )
    conn.close()


def count_by_type() -> dict:
    """문서 종류별 저장 건수를 반환한다 (적재 검증용)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type").fetchall()
    conn.close()
    return dict(rows)
