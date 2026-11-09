"""키워드 토큰화 — 인덱서(indexer/keyword.py, 색인)와 검색기(rag/retriever.py, 질문)가 함께 쓴다.

settings.py 처럼 계층 밖 공용 모듈로 둔 이유: 색인과 검색이 반드시 같은 토큰화를 써야 하지만,
챗봇(rag·app)이 인덱서 코드를 import 하면 '인덱서를 빼도 챗봇이 구동된다'는 계층 분리 조건(가이드 4-2, 7장)이 깨진다.

규칙: 한글 연속 구간은 글자 2개씩(바이그램, 1글자면 그대로), 영문·숫자는 단어 그대로(소문자),
소수점 숫자는 FTS5 가 '.' 을 구분자로 보므로 84.6 → 84d6 으로 한 토큰 유지.
    예) "한미반도체의 TC본더 84.6%" → 한미 미반 반도 도체 체의 tc 본더 84d6
"""
import re

HANGUL_RUN = re.compile(r"[가-힣]+")
TOKEN = re.compile(r"[가-힣]+|[0-9]+(?:\.[0-9]+)?|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower()):
        if HANGUL_RUN.fullmatch(t):
            out.extend([t] if len(t) == 1 else [t[i:i + 2] for i in range(len(t) - 1)])
        else:
            out.append(t.replace(".", "d"))
    return out
