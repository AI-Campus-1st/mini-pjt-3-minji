"""대화 UI — 스트리밍, 멀티턴, 검색 진행 표시, 출처 목록, 원문 확인, 필터 (가이드 4-6, 4-7).

화면 흐름 (질문 1회)
    사용자 입력 → [검색 중 상태 표시] RAGChain.prepare (재작성 + 검색 + 거절 판단)
    → 결과 없음/유사도 미달이면 안내·거절 문구 → 아니면 RAGChain.stream 으로 답변 스트리밍
    → 답변 아래에 번호 출처 목록 (인용된 출처 먼저, 근거 부분·원문 전체 펼치기)

대화 기록은 st.session_state.messages 에 저장되어, 다음 질문 때 rag 체인에 history 로 전달된다(멀티턴).
"""
from dataclasses import asdict

import streamlit as st

from rag.chain import Prepared, RAGChain, Source, cited_numbers
from rag.documents import get_document
from rag.prompts import DOC_TYPE_LABEL, REFUSAL, is_refusal

YEAR_MIN, YEAR_MAX = 2021, 2026   # 수집 문서의 발행 연도 범위
EMPTY_MSG = "선택한 조건에서 검색된 문서가 없습니다. 왼쪽에서 기간이나 출처 종류를 넓혀 다시 질문해 주세요."
LOW_MSG = "질문과 충분히 가까운 문서를 찾지 못해 답변하지 않았습니다. 질문을 더 구체적으로 바꿔 보세요."
EXAMPLES = [
    "SK하이닉스가 HBM 시장에서 급성장한 이유는?",
    "정부의 국산 AI 반도체(NPU) 지원 정책은 어떻게 전개됐나?",
    "한미반도체의 주요 제품은 무엇인가?",
]


@st.cache_resource(show_spinner="검색기를 불러오는 중…")
def get_chain() -> RAGChain:
    """RAG 체인은 벡터 DB 연결을 들고 있으므로 앱 실행 동안 1번만 만든다."""
    return RAGChain()


FILTER_DEFAULTS = {"f_types": list(DOC_TYPE_LABEL), "f_years": (YEAR_MIN, YEAR_MAX), "show_score": False}


@st.fragment
def filter_panel() -> None:
    """사이드바 검색 조건. 프래그먼트(fragment)로 감싼 이유:

    일반 위젯은 조작하면 화면 전체가 처음부터 다시 실행되어, 답변 생성 도중이면 답변이 끊긴다.
    프래그먼트 안의 위젯은 이 함수만 다시 실행하고, 답변이 진행 중이면 끝난 뒤로 미뤄진다.
    값은 key 로 session_state 에 저장되고, current_filters() 가 다음 질문 때 읽는다.
    """
    st.header("검색 조건")
    st.multiselect("출처 종류", options=list(DOC_TYPE_LABEL), format_func=DOC_TYPE_LABEL.get, key="f_types")
    st.slider("발행 연도", YEAR_MIN, YEAR_MAX, key="f_years")
    st.toggle("유사도 점수 표시", key="show_score", help="검색된 문서와 질문의 코사인 유사도 (디버깅용)")
    # 점수 표시는 이미 그려진 출처 목록을 다시 그려야 하므로 전체 화면을 다시 실행한다
    # (프래그먼트 실행은 답변이 끝난 뒤에 일어나므로 답변을 끊지 않는다)
    if st.session_state.get("_score_prev", st.session_state.show_score) != st.session_state.show_score:
        st.session_state._score_prev = st.session_state.show_score
        st.rerun()
    st.session_state._score_prev = st.session_state.show_score
    st.caption("검색 조건은 다음 질문부터 적용됩니다.")
    st.divider()
    if st.button("대화 새로 시작", use_container_width=True):
        st.session_state.messages = []
        st.rerun()   # 프래그먼트 안의 st.rerun() 은 기본값이 화면 전체 재실행
    st.caption("답변은 원문 확인을 돕기 위한 요약입니다. 인용 전 반드시 원문을 확인하세요.")


def current_filters() -> dict:
    """session_state 의 검색 조건 → RAGChain.prepare 에 넘길 인자."""
    types = st.session_state.f_types
    y0, y1 = st.session_state.f_years
    # 전체 범위를 고르면 필터를 걸지 않는다 (불필요한 조건으로 검색이 느려지지 않게)
    return {
        "doc_types": types if types and len(types) < len(DOC_TYPE_LABEL) else None,
        "year_from": y0 if y0 > YEAR_MIN else None,
        "year_to": y1 if y1 < YEAR_MAX else None,
        "_no_type": not types,
    }


def render_sources(sources: list[dict], answer: str) -> None:
    """번호 출처 목록. 답변에서 인용된 번호를 먼저, 나머지는 '참고만 한 문서'로 접어 둔다."""
    if not sources:
        return
    # 번호 순으로 정렬해 답변 속 [n] 과 찾기 쉽게 한다
    cited = sorted(n for n in cited_numbers(answer) if 1 <= n <= len(sources))
    by_n = {s["n"]: s for s in sources}
    used = [by_n[n] for n in cited]
    rest = [s for s in sources if s["n"] not in cited]

    st.markdown("**출처**")
    for s in used:
        _source_item(s)
    if rest:
        label = (f"검색된 문서 {len(rest)}건 (답변 근거 아님)" if is_refusal(answer) and not used
                 else f"검색됐지만 인용되지 않은 문서 {len(rest)}건")
        with st.expander(label):
            for s in rest:
                _source_item(s)


def _source_item(s: dict) -> None:
    """출처 1건: [n] 출처 종류 · 발행처 · 보도일 · 제목(링크) + 근거 부분/원문 펼치기."""
    src = Source(**s)
    tag = " · 1차 자료" if src.is_primary else ""   # 뉴스는 라벨에 이미 '언론 보도'가 있으므로 생략
    title = f"[{src.title}]({src.url})" if src.url else src.title
    loc = " · ".join(x for x in [src.section, f"{src.page}쪽" if src.page else None] if x)
    score = f" · 유사도 {src.score:.3f}" if st.session_state.get("show_score") else ""
    st.markdown(f"**[{src.n}]** `{src.type_label}`{tag} · {src.source} · 보도일 {src.date}{score}  \n{title}"
                + (f"  \n<small>{loc}</small>" if loc else ""), unsafe_allow_html=True)
    with st.expander("근거 부분 / 원문 보기"):
        st.caption("검색된 부분")
        st.text(src.text)
        doc = get_document(src.doc_id)   # 원문은 docs.db 에서 따로 조회 (가이드 4-5)
        if doc and doc["body"]:
            st.caption(f"원문 전체 ({len(doc['body']):,}자)")
            st.text_area("원문", doc["body"], height=300, key=f"body_{id(s)}_{src.doc_id}",
                         label_visibility="collapsed", disabled=True)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("search_query") and m["search_query"] != m.get("question"):
                st.caption(f"검색어로 바꾼 질문: {m['search_query']}")
            render_sources(m.get("sources", []), m["content"])


def answer(question: str, filters: dict, resume: bool = False) -> None:
    """질문 1회 처리: 검색 상태 표시 → 거절/안내 또는 스트리밍 답변 → 기록 저장.

    Args:
        resume: True 면 이미 기록에 들어간 '답 없는 질문'을 이어서 답한다.
                (답변 도중 위젯 조작·코드 변경으로 화면이 다시 실행되면 답변 단계가 끊기는데,
                 이때 질문만 남고 답이 사라지지 않게 하기 위함)
    """
    rag = get_chain()
    msgs = st.session_state.messages[:-1] if resume else st.session_state.messages
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    if not resume:
        st.session_state.messages.append({"role": "user", "content": question})
        render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        if filters.pop("_no_type"):
            text, prep = EMPTY_MSG, None
            st.info(text)
        else:
            with st.status("관련 문서를 검색하는 중…", expanded=False) as status:
                prep: Prepared = rag.prepare(question, history, **filters)
                if prep.search_query != question:
                    st.write(f"검색어로 바꾼 질문: {prep.search_query}")
                status.update(label=f"문서 {len(prep.sources)}건 검색 완료", state="complete")

            if prep.status == "empty":
                text = EMPTY_MSG
                st.info(text)
            elif prep.status == "low_score":
                text = f"{REFUSAL}\n\n{LOW_MSG}"
                st.markdown(text)
            else:
                try:
                    text = st.write_stream(rag.stream(prep))   # 토큰이 오는 대로 화면에 출력
                except Exception as e:                          # API 키 누락·네트워크 오류 등
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                    return
                if prep.search_query != question:
                    st.caption(f"검색어로 바꾼 질문: {prep.search_query}")
                render_sources([asdict(s) for s in prep.sources], text)

    st.session_state.messages.append({
        "role": "assistant", "content": text, "question": question,
        "search_query": prep.search_query if prep else None,
        # 거절·빈 결과일 때는 출처를 보여주지 않는다 (근거가 아니므로)
        "sources": [asdict(s) for s in prep.sources] if prep and prep.status == "ok" else [],
    })


def render() -> None:
    st.title("AI 반도체 리서치 챗봇")
    st.caption("기업 공시 · 정부 보도자료 · 산업 보고서 · 뉴스를 근거로 답하고, 원문으로 연결합니다.")
    st.session_state.setdefault("messages", [])
    for k, v in FILTER_DEFAULTS.items():   # 위젯 key 의 초깃값 (첫 실행 1회)
        st.session_state.setdefault(k, v)
    with st.sidebar:
        filter_panel()
    filters = current_filters()

    for m in st.session_state.messages:
        render_message(m)

    clicked = None
    examples = st.empty()   # 첫 질문이 들어오면 예시 버튼 영역을 비우기 위한 자리
    if not st.session_state.messages:
        with examples.container():
            st.markdown("**예시 질문**")
            cols = st.columns(len(EXAMPLES))
            for col, q in zip(cols, EXAMPLES):
                if col.button(q, use_container_width=True):
                    clicked = q

    question = st.chat_input("질문을 입력하세요") or clicked
    # 마지막 기록이 사용자 질문이면 이전 실행에서 답변이 끊긴 것 → 이어서 답한다
    pending = bool(st.session_state.messages) and st.session_state.messages[-1]["role"] == "user"
    if question:
        examples.empty()
        if pending:   # 끊긴 질문은 버리고 새 질문에 답한다 (질문-답 짝이 어긋나지 않게)
            st.session_state.messages.pop()
        answer(question, filters)
    elif pending:
        answer(st.session_state.messages[-1]["content"], filters, resume=True)
