"""원문 조회 — 챗봇의 '원문 확인' 영역에 보여줄 문서 전체 본문을 docs.db 에서 읽는다.

가이드 4-5: 벡터 DB(Chroma)에는 청크와 메타데이터만 있고, 원문 전체는 별도 DB에 보관한다.
사용자가 원문 확인을 요청할 때만 여기서 조회한다. (조회 전용, 쓰기 없음)

호출 위치: app/chat.py (출처 목록의 '원문 보기')
"""
import sqlite3

from settings import DOCS_DB


def get_document(doc_id: str) -> dict | None:
    """doc_id 의 메타데이터와 정제된 본문을 반환한다. 없으면 None.

    Returns:
        {"doc_id", "doc_type", "source", "title", "url", "published_at", "body"}
        body 는 정제본(parsed_documents)을 우선 쓰고, 없으면 수집 원문(documents.body).
    """
    # 읽기 전용 모드로 열어 챗봇이 DB를 바꾸지 못하게 한다
    conn = sqlite3.connect(f"file:{DOCS_DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            """SELECT d.doc_id, d.doc_type, d.source, d.title, d.url, d.published_at,
                      COALESCE(p.body, d.body)
               FROM documents d LEFT JOIN parsed_documents p USING (doc_id)
               WHERE d.doc_id = ?""", (doc_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    keys = ["doc_id", "doc_type", "source", "title", "url", "published_at", "body"]
    return dict(zip(keys, row))
