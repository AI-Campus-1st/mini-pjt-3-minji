"""LangChain RAG 체인. 챗봇(app/chat.py)과 평가 하네스(Day 5 eval)가 동일하게 호출한다.

실행 흐름 (질문 1회)
    1) rewrite  : 이전 대화가 있으면 후속 질문을 검색용 독립 질문으로 바꾼다 (LLM 1회)
    2) retrieve : Retriever.search_docs 로 문서 단위 상위 k개 → 번호 붙은 Source 목록
    3) gate     : 검색 결과가 없거나 1위 유사도가 MIN_SCORE 미만이면 LLM 을 부르지 않고 거절
    4) generate : SYSTEM_PROMPT + 최근 대화 + 질문 → 답변 (스트리밍 또는 한 번에)

사용 예
    rag = RAGChain()
    prep = rag.prepare("한미반도체의 주요 제품은?", history=[], doc_types=["dart"])
    for token in rag.stream(prep): print(token, end="")
    rag.answer("...")   # 평가용: 스트리밍 없이 dict 반환
"""
import re
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.prompts import (DOC_TYPE_LABEL, PRIMARY_TYPES, REFUSAL, REWRITE_PROMPT, SYSTEM_PROMPT,
                         format_context, is_refusal)
from rag.retriever import Retriever
from settings import CHUNK_CONFIG, LLM_MODEL

# 거절 하한. Day 3 측정에서 답변 가능 문항의 1위 유사도 최솟값이 0.540~0.549 였다.
# 답변 불가 문항도 최대 0.739 까지 올라가므로 유사도만으로는 거절할 수 없다 (report/eval_baseline.md 4절).
# → 명백히 무관한 질문만 거르는 하한으로 쓰고, 나머지 거절은 프롬프트(규칙 2·3)에 맡긴다.
MIN_SCORE = 0.45
K_DOCS = 6          # 답변 근거로 넘길 문서 수 (같은 문서의 청크는 1개로 합침)
FETCH_K = 50        # 문서 단위로 묶기 전에 가져올 청크 수 (eval/run_eval.py 와 동일)
HISTORY_TURNS = 3   # 생성·재작성에 넣을 최근 대화 턴 수 (토큰·비용 제한)


@dataclass
class Source:
    """답변 근거 문서 1건. n 은 답변 속 [n] 표기와 화면 출처 목록의 번호."""
    n: int
    doc_id: str
    doc_type: str
    source: str
    title: str
    url: str
    published_at: str
    section: str | None
    page: int | None
    text: str
    score: float

    @property
    def type_label(self) -> str:
        return DOC_TYPE_LABEL.get(self.doc_type, self.doc_type)

    @property
    def is_primary(self) -> bool:
        return self.doc_type in PRIMARY_TYPES

    @property
    def date(self) -> str:
        return (self.published_at or "")[:10]


@dataclass
class Prepared:
    """생성 직전까지의 결과. 화면은 이것으로 '검색 결과'를 먼저 보여주고 답변을 스트리밍한다."""
    question: str
    search_query: str                 # 재작성된 검색 질문 (재작성 안 했으면 question 과 같음)
    sources: list[Source]
    history: list = field(default_factory=list)
    status: str = "ok"                # ok | empty(검색 결과 없음) | low_score(유사도 미달 거절)

    @property
    def top_score(self) -> float | None:
        return self.sources[0].score if self.sources else None


def to_messages(history: list[dict], turns: int = HISTORY_TURNS) -> list:
    """화면의 대화 기록 [{"role": "user"|"assistant", "content": str}, ...] → LangChain 메시지 (최근 turns 턴)."""
    msgs = []
    for m in history[-turns * 2:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs


def cited_numbers(answer: str) -> list[int]:
    """답변에서 [n] 표기를 찾아 등장 순서대로 중복 없이 반환. 화면에서 '인용된 출처'를 강조하는 데 쓴다."""
    seen = []
    for n in re.findall(r"\[(\d+)\]", answer):
        if int(n) not in seen:
            seen.append(int(n))
    return seen


class RAGChain:
    """검색 → 거절 판단 → 생성. 

    Args:
        config: 청킹 설정 (기본 settings.CHUNK_CONFIG = "B")
        retriever: 테스트에서 가짜 임베딩을 쓰는 Retriever 를 주입할 때 사용
        llm: LangChain 채팅 모델. 기본 ChatOpenAI(LLM_MODEL). 테스트에서 가짜 모델 주입
        rewrite: 멀티턴 후속 질문 재작성 사용 여부
    """

    def __init__(self, config: str = CHUNK_CONFIG, retriever: Retriever | None = None, llm=None,
                 k_docs: int = K_DOCS, min_score: float = MIN_SCORE, rewrite: bool = True):
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)   # 0: 같은 질문에 같은 답 (평가 재현성)
        self.retriever = retriever or Retriever(config)
        self.k_docs, self.min_score, self.use_rewrite = k_docs, min_score, rewrite
        parser = StrOutputParser()
        # LCEL(LangChain Expression Language): 프롬프트 | 모델 | 파서 를 하나의 실행 단위로 연결
        self.answer_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]).partial(refusal=REFUSAL)
            | llm | parser
        )
        self.rewrite_chain = ChatPromptTemplate.from_messages([("human", REWRITE_PROMPT)]) | llm | parser

    # 1) 질의 재작성 ---------------------------------------------------------
    def rewrite(self, question: str, history: list[dict]) -> str:
        """이전 대화가 없으면 그대로, 있으면 검색용 독립 질문으로 바꾼다."""
        if not self.use_rewrite or not history:
            return question
        text = "\n".join(f"{'사용자' if m['role'] == 'user' else '어시스턴트'}: {m['content'][:500]}"
                         for m in history[-HISTORY_TURNS * 2:])
        out = self.rewrite_chain.invoke({"history": text, "question": question}).strip()
        return out or question

    # 2) 검색 + 3) 거절 판단 --------------------------------------------------
    def prepare(self, question: str, history: list[dict] | None = None,
                doc_types=None, year_from=None, year_to=None) -> Prepared:
        """검색까지 수행하고, 생성할지 거절할지 상태를 정한다."""
        history = history or []
        query = self.rewrite(question, history)
        hits = self.retriever.search_docs(query, k_docs=self.k_docs, fetch_k=FETCH_K,
                                          doc_types=doc_types, year_from=year_from, year_to=year_to)
        sources = [Source(n=i, doc_id=h.doc_id, doc_type=h.doc_type, source=h.source, title=h.title,
                          url=h.url, published_at=h.published_at, section=h.section, page=h.page,
                          text=h.text, score=h.score) for i, h in enumerate(hits, 1)]
        status = "ok"
        if not sources:
            status = "empty"
        elif sources[0].score < self.min_score:
            status = "low_score"
        return Prepared(question=question, search_query=query, sources=sources, history=history, status=status)

    # 4) 생성 -----------------------------------------------------------------
    def stream(self, prep: Prepared) -> Iterator[str]:
        """답변 토큰을 순서대로 내보낸다. 거절 상태면 LLM 호출 없이 고정 문구."""
        if prep.status != "ok":
            yield REFUSAL
            return
        yield from self.answer_chain.stream({
            "context": format_context(prep.sources),
            "history": to_messages(prep.history),
            "question": prep.question,
        })

    def answer(self, question: str, history: list[dict] | None = None, **filters) -> dict:
        """스트리밍 없이 한 번에 답한다 (평가 하네스용)."""
        prep = self.prepare(question, history, **filters)
        text = "".join(self.stream(prep))
        return {
            "question": question,
            "search_query": prep.search_query,
            "status": prep.status,
            "refused": is_refusal(text),
            "answer": text,
            "cited": cited_numbers(text),
            "sources": [{"n": s.n, "doc_id": s.doc_id, "doc_type": s.doc_type, "title": s.title,
                         "score": s.score} for s in prep.sources],
        }
