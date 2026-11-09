"""평가 결과 대시보드 (Plotly) — 챗봇과 별도 페이지 (가이드 4-8, Day 5).

읽는 파일 (모두 조회 전용, 쓰기 없음)
- eval/results/retrieval_{tag}_{A|B}.json : 검색 평가 (eval/run_eval.py)
- eval/results/answers_{tag}.jsonl        : 답변 실행 결과 (eval/run_answers.py)
- eval/grades_{tag}.json                  : 답변 채점 (정확/부분/오답, 출처, 실패 원인 계층)
- store/docs.db                           : 지식베이스 구성 (문서 종류·연도, 청크 수)

색 규칙 (dataviz 기준 팔레트, 검증 스크립트 통과)
- 문서 종류 = 범주형 고정 순서: 뉴스 파랑, 공시 주황, 보도자료 청록, 보고서 노랑
- 답변 판정 = 상태색 (정확 초록 / 부분 노랑 / 오답 빨강) + 항상 글자 라벨 함께
- 노랑·청록은 밝은 배경 대비가 3:1 미만이라 모든 차트에 숫자 라벨과 표 보기를 함께 둔다.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag.chain import MIN_SCORE
from settings import DOCS_DB, ROOT

RESULTS = ROOT / "eval" / "results"
GRADES = ROOT / "eval"

# 답변 실험 (report/evaluation.md 2부 순서). 라벨은 '무엇을 바꿨는지'
EXPERIMENTS = [
    ("baseline", "0 baseline"),
    ("chunks3", "1 문서당 청크 3개"),
    ("balance1", "2 + 1차 자료 보장"),
    ("prompt2", "3 + 프롬프트 v2 (미채택)"),
    ("prompt3", "4 + 프롬프트 v3"),
    ("kwchunk1", "5 + 문서 안 키워드 청크 (최종)"),
]
FINAL_TAG = "kwchunk1"
GRADE_ORDER = ["정확", "부분", "오답"]
GRADE_COLOR = {"정확": "#0ca30c", "부분": "#fab219", "오답": "#d03b3b"}          # 상태색
DOC_TYPES = ["news", "dart", "press", "report"]
DOC_LABEL = {"news": "뉴스", "dart": "기업 공시", "press": "정부 보도자료", "report": "산업 보고서"}
DOC_COLOR = {"뉴스": "#2a78d6", "기업 공시": "#eb6834", "정부 보도자료": "#1baf7a", "산업 보고서": "#eda100"}
CONFIG_COLOR = {"A (고정 1,000자)": "#2a78d6", "B (구조 기반)": "#eb6834"}
TYPE_LABEL = {"fact": "사실", "compare": "비교", "time": "시점", "synthesis": "종합", "unanswerable": "답변 불가"}


# ---------------------------------------------------------------------------
# 데이터 읽기 (파일이 없으면 빈 값 → 해당 차트만 건너뜀)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_answers(tag: str) -> list[dict]:
    p = RESULTS / f"answers_{tag}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


@st.cache_data(ttl=60)
def load_grades(tag: str) -> list[dict]:
    p = GRADES / f"grades_{tag}.json"
    return json.load(open(p, encoding="utf-8"))["grades"] if p.exists() else []


@st.cache_data(ttl=60)
def load_retrieval(tag: str) -> dict:
    out = {}
    for cfg in ("A", "B"):
        p = RESULTS / f"retrieval_{tag}_{cfg}.json"
        if p.exists():
            out[cfg] = json.load(open(p, encoding="utf-8"))
    return out


@st.cache_data(ttl=60)
def retrieval_experiments() -> pd.DataFrame:
    """검색 단계 실험 비교 (eval/exp_balance.py, eval/exp_hybrid.py 결과). 모델에 넘기는 문서 6개 기준."""
    rows = []
    sources = [("exp_balance.json", "base", "의미 검색 (baseline)"),
               ("exp_balance.json", "min1_m0.10", "+ 1차 자료 최소 1개 (채택)"),
               ("exp_balance.json", "fixed3", "+ 1차 자료 고정 3개"),
               ("exp_hybrid.json", "rrf", "전체 하이브리드 RRF (미채택)"),
               ("exp_hybrid.json", "kw1", "+ 문서 안 키워드 청크 (최종)")]
    for fname, key, label in sources:
        p = RESULTS / fname
        if not p.exists():
            continue
        r = json.load(open(p, encoding="utf-8")).get(key)
        if r:
            rows += [{"설정": label, "지표": m, "값": r[k]}
                     for k, m in [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("hit@6", "Hit@6"), ("mrr@6", "MRR@6")]]
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def load_kb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """지식베이스 구성: (문서 종류·연도별 대표 문서 수, 문서 종류별 청크 수 — 설정 B)."""
    conn = sqlite3.connect(f"file:{DOCS_DB}?mode=ro", uri=True)
    try:
        docs = pd.read_sql(
            """SELECT d.doc_type, substr(d.published_at, 1, 4) AS year, count(*) AS n
               FROM documents d JOIN dedup_map m USING (doc_id) WHERE m.keep = 1 GROUP BY 1, 2""", conn)
        chunks = pd.read_sql("SELECT doc_type, count(*) AS n FROM chunks WHERE config = 'B' GROUP BY 1", conn)
    finally:
        conn.close()
    for df in (docs, chunks):
        df["종류"] = df["doc_type"].map(DOC_LABEL)
    return docs, chunks


def experiment_table() -> pd.DataFrame:
    """실험별 요약 지표 한 줄씩."""
    rows = []
    for tag, label in EXPERIMENTS:
        ans, grades = load_answers(tag), load_grades(tag)
        if not ans or not grades:
            continue
        g = [x for x in grades if x["grade"] != "거절 성공"]
        a = [x for x in ans if x["type"] != "unanswerable"]
        u = [x for x in ans if x["type"] == "unanswerable"]
        rows.append({
            "실험": label,
            **{k: sum(x["grade"] == k for x in g) for k in GRADE_ORDER},
            "과잉 거절": sum(x["refused"] for x in a),
            "거절 정확도": f"{sum(x['refused'] for x in u)}/{len(u)}",
            "출처 오류": sum(x["source"] == "오류" for x in g),
            "정답 문서 검색률": round(sum(x["gold_retrieved"] for x in a) / len(a), 3),
            "평균 응답(초)": round(sum(x["latency_s"] for x in ans) / len(ans), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 320) -> go.Figure:
    """공통 모양: 얇은 막대, 흐린 격자, 범례는 위쪽 한 줄."""
    fig.update_layout(height=height, margin=dict(l=8, r=24, t=40, b=40), bargap=0.35,
                      uniformtext=dict(minsize=10, mode="show"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", automargin=True)
    fig.update_traces(textangle=0, selector=dict(type="bar"))   # 좁은 막대에서도 숫자를 눕히지 않음
    return fig


def chart_experiments(df: pd.DataFrame) -> go.Figure:
    """실험별 정확/부분/오답 — 가로 누적 막대 (34문항)."""
    long = df.melt(id_vars="실험", value_vars=GRADE_ORDER, var_name="판정", value_name="문항 수")
    fig = px.bar(long, y="실험", x="문항 수", color="판정", orientation="h", text="문항 수",
                 color_discrete_map=GRADE_COLOR, category_orders={"판정": GRADE_ORDER})
    fig.update_traces(textposition="inside", insidetextanchor="middle", marker_line_width=2,
                      marker_line_color="rgba(255,255,255,0.9)",
                      hovertemplate="%{y}<br>%{fullData.name}: %{x}문항<extra></extra>")
    fig.update_yaxes(autorange="reversed", title=None)   # 위에서 아래로 실험 순서 (0 → 4)
    fig.update_xaxes(title="답변 가능 34문항", range=[0, 34])
    return style(fig, 300)


def chart_by_type(answers: list[dict], grades: list[dict]) -> go.Figure:
    """최종 설정의 질문 유형별 판정 비율."""
    gmap = {g["id"]: g["grade"] for g in grades}
    df = pd.DataFrame([{"유형": TYPE_LABEL[a["type"]], "판정": gmap[a["id"]]}
                       for a in answers if a["type"] != "unanswerable" and a["id"] in gmap])
    cnt = df.groupby(["유형", "판정"]).size().reset_index(name="문항 수")
    fig = px.bar(cnt, x="유형", y="문항 수", color="판정", text="문항 수", color_discrete_map=GRADE_COLOR,
                 category_orders={"판정": GRADE_ORDER, "유형": ["사실", "비교", "시점", "종합"]})
    fig.update_traces(textposition="inside", marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)",
                      hovertemplate="%{x} · %{fullData.name}: %{y}문항<extra></extra>")
    fig.update_xaxes(title=None)
    return style(fig)


def chart_layers(grades: list[dict]) -> go.Figure:
    """정확이 아닌 문항의 실패 원인 계층 (최종 설정)."""
    names = {"검색": "검색: 1차 자료·최신 정보 미검색", "검색-청크": "검색: 답 청크 미전달", "생성": "생성: 답변 단계"}
    df = pd.DataFrame([names.get(g["layer"], g["layer"]) for g in grades
                       if g["grade"] in ("부분", "오답") and g["layer"]], columns=["원인"])
    cnt = df.value_counts().reset_index(name="문항 수").sort_values("문항 수")
    fig = px.bar(cnt, x="문항 수", y="원인", orientation="h", text="문항 수")
    fig.update_traces(marker_color="#2a78d6", textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x}문항<extra></extra>")
    fig.update_yaxes(title=None)
    return style(fig, 220)


def chart_retrieval(ret: dict) -> go.Figure:
    """청킹 설정 A vs B 검색 지표."""
    rows = []
    names = {"A": "A (고정 1,000자)", "B": "B (구조 기반)"}
    for cfg, r in ret.items():
        for k, label in [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("hit@5", "Hit@5"), ("mrr@10", "MRR@10")]:
            rows.append({"설정": names[cfg], "지표": label, "값": r[k]})
    fig = px.bar(pd.DataFrame(rows), x="지표", y="값", color="설정", barmode="group", text="값",
                 color_discrete_map=CONFIG_COLOR)
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False,
                      hovertemplate="%{fullData.name} · %{x}: %{y:.3f}<extra></extra>")
    fig.update_yaxes(range=[0, 1.08], title=None)
    fig.update_xaxes(title=None)
    return style(fig)


def chart_scores(r: dict) -> go.Figure:
    """1위 유사도 분포: 답변 가능 vs 답변 불가 + 거절 하한선."""
    df = pd.DataFrame([{"구분": "답변 불가" if q["type"] == "unanswerable" else "답변 가능",
                        "1위 유사도": q["top1_score"], "문항": q["id"]} for q in r["per_question"]])
    fig = px.strip(df, x="1위 유사도", y="구분", color="구분", hover_data=["문항"],
                   color_discrete_map={"답변 가능": "#2a78d6", "답변 불가": "#eb6834"})
    fig.update_traces(marker_size=10, marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    fig.add_vline(x=MIN_SCORE, line_dash="dash", line_color="gray",
                  annotation_text=f"거절 하한 {MIN_SCORE}", annotation_position="top")
    fig.update_yaxes(title=None)
    fig.update_layout(showlegend=False)
    return style(fig, 240)


def chart_kb(docs: pd.DataFrame) -> go.Figure:
    """연도별 대표 문서 수 — 종류마다 규모가 수백 배 달라 작은 그림 여러 개(축 따로)."""
    fig = px.bar(docs.sort_values("year"), x="year", y="n", color="종류", facet_col="종류", facet_col_wrap=4,
                 text="n", color_discrete_map=DOC_COLOR,
                 category_orders={"종류": [DOC_LABEL[t] for t in DOC_TYPES]})
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}년 · %{y:,}건<extra></extra>")
    fig.update_yaxes(matches=None, showticklabels=False, title=None)
    fig.update_xaxes(title=None, type="category")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    return style(fig, 260)


SERIES5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # 범주형 고정 순서 앞 5칸 (검증 팔레트)


def chart_retrieval_experiments(df: pd.DataFrame) -> go.Figure:
    """검색 실험별 Hit@1·3·6·MRR (가이드 4-8 '실험 설정별 막대 비교')."""
    order = list(dict.fromkeys(df["설정"]))
    fig = px.bar(df, x="지표", y="값", color="설정", barmode="group", text="값",
                 color_discrete_sequence=SERIES5, category_orders={"설정": order})
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False, textfont_size=10,
                      hovertemplate="%{fullData.name} · %{x}: %{y:.3f}<extra></extra>")
    fig.update_yaxes(range=[0, 1.1], title=None)
    fig.update_xaxes(title=None)
    fig = style(fig, 360)
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    return fig


def chart_scores_by_grade(answers: list[dict], grades: list[dict]) -> go.Figure:
    """정답·오답 사례별 1위 유사도 (가이드 4-8 '정답 사례와 오답 사례의 점수 분포', 최종 설정)."""
    gmap = {g["id"]: g["grade"] for g in grades}
    df = pd.DataFrame([{"판정": gmap.get(a["id"], "?"), "1위 유사도": max(x["score"] for x in a["sources"]),
                        "문항": a["id"]} for a in answers if a["sources"]])
    order = ["정확", "부분", "오답", "거절 성공"]
    colors = {**GRADE_COLOR, "거절 성공": "#2a78d6"}
    fig = px.strip(df, x="1위 유사도", y="판정", color="판정", hover_data=["문항"],
                   category_orders={"판정": order}, color_discrete_map=colors)
    fig.update_traces(marker_size=10, marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    fig.add_vline(x=MIN_SCORE, line_dash="dash", line_color="gray",
                  annotation_text=f"거절 하한 {MIN_SCORE}", annotation_position="top")
    fig.update_yaxes(title=None)
    fig.update_layout(showlegend=False)
    return style(fig, 260)


def chart_latency(tags: list[tuple[str, str]]) -> go.Figure:
    """실험별 응답 시간 분포 (가이드 4-8 '응답시간 분포' — 심화 확장 전후 비교)."""
    rows = [{"실험": label, "응답 시간(초)": a["latency_s"], "문항": a["id"]}
            for tag, label in tags for a in load_answers(tag)]
    fig = px.box(pd.DataFrame(rows), x="응답 시간(초)", y="실험", points="all", hover_data=["문항"])
    fig.update_traces(marker_color="#2a78d6", line_color="#2a78d6", marker_size=6)
    fig.update_yaxes(autorange="reversed", title=None)
    return style(fig, 320)


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
def render() -> None:
    st.title("평가 결과")
    st.caption("평가셋 40문항(답변 가능 34 · 답변 불가 6) 기준. 상세: report/evaluation.md 1부, report/evaluation.md 2부")

    exp = experiment_table()
    final_ans, final_grades = load_answers(FINAL_TAG), load_grades(FINAL_TAG)

    # 1) 핵심 수치 — 최종 설정
    if not exp.empty:
        last, first = exp.iloc[-1], exp.iloc[0]
        c = st.columns(4)
        c[0].metric("정확 (34문항)", f"{last['정확']}문항", f"{last['정확'] - first['정확']:+d} vs baseline")
        c[1].metric("정답 문서 검색률", f"{last['정답 문서 검색률']:.3f}",
                    f"{last['정답 문서 검색률'] - first['정답 문서 검색률']:+.3f}")
        c[2].metric("거절 정확도 (답변 불가)", last["거절 정확도"])
        c[3].metric("평균 응답 시간", f"{last['평균 응답(초)']}초",
                    f"{last['평균 응답(초)'] - first['평균 응답(초)']:+.2f}초", delta_color="inverse")

    # 2) 실험별 답변 품질
    st.subheader("답변 품질 — 개선 실험별")
    st.caption("한 번에 한 가지만 바꿔 측정했다. 같은 입력에서도 OpenAI 응답은 실행마다 조금 달라져 1~2문항 차이는 변동 범위일 수 있다.")
    if exp.empty:
        st.info("답변 평가 결과가 없습니다. `python eval/run_answers.py --tag baseline` 을 먼저 실행하세요.")
    else:
        st.plotly_chart(chart_experiments(exp), use_container_width=True)
        with st.expander("표로 보기"):
            st.dataframe(exp, hide_index=True, use_container_width=True)
        st.markdown("**응답 시간 분포 — 실험별**")
        st.plotly_chart(chart_latency([(t, l) for t, l in EXPERIMENTS if load_answers(t)]), use_container_width=True)
        st.caption("하이브리드(문서 안 키워드 청크) 적용 후 평균 3.35초 → 5.78초: 키워드 검색 0.5초 + 문맥이 길어져 생성 시간 증가.")

    # 3) 최종 설정 세부
    if final_ans and final_grades:
        left, right = st.columns(2)
        with left:
            st.subheader("질문 유형별 (최종)")
            st.plotly_chart(chart_by_type(final_ans, final_grades), use_container_width=True)
        with right:
            st.subheader("남은 실패의 원인 (최종)")
            st.plotly_chart(chart_layers(final_grades), use_container_width=True)
            with st.expander("문항별 판정 보기"):
                st.dataframe(pd.DataFrame(final_grades), hide_index=True, use_container_width=True)
        st.markdown("**정답·오답 사례별 1위 문서 유사도 (최종)**")
        st.plotly_chart(chart_scores_by_grade(final_ans, final_grades), use_container_width=True)
        st.caption("정확·오답·거절 사례의 유사도가 겹친다 → 유사도 임계값만으로는 오답을 걸러낼 수 없어, 하한은 명백히 무관한 경우만 거르고 나머지는 프롬프트가 판단한다.")

    # 4) 검색 성능
    rx = retrieval_experiments()
    if not rx.empty:
        st.subheader("검색 성능 — 개선 실험별 (설정 B, 문서 6개 기준)")
        st.plotly_chart(chart_retrieval_experiments(rx), use_container_width=True)
        st.caption("1차 자료 보장으로 Hit@6 0.824 → 0.971. 전체 하이브리드(RRF)는 뉴스 잡음으로 0.765 로 떨어져 미채택. "
                   "문서 안 키워드 청크는 문서 선택을 바꾸지 않아 지표는 같고, 답 청크 전달이 늘었다(답변 품질로 확인).")
        with st.expander("표로 보기"):
            st.dataframe(rx.pivot_table(index="설정", columns="지표", values="값", sort=False), use_container_width=True)
    st.subheader("검색 성능 — 청킹 설정 A vs B")
    tags = {"정답 라벨 보완 후 (gold_v2)": "gold_v2", "1차 측정 (baseline)": "baseline"}
    pick = st.radio("측정", list(tags), horizontal=True, label_visibility="collapsed")
    ret = load_retrieval(tags[pick])
    if ret:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(chart_retrieval(ret), use_container_width=True)
        with right:
            if "B" in ret:
                st.markdown("**1위 문서 유사도 분포 (설정 B)**")
                st.plotly_chart(chart_scores(ret["B"]), use_container_width=True)
                st.caption("답변 불가 문항도 0.74까지 올라가 유사도만으로는 거절할 수 없다 → 명백히 무관한 경우만 하한으로 거르고 나머지는 프롬프트가 판단.")
        with st.expander("유형별 Hit@3 표"):
            st.dataframe(pd.DataFrame({TYPE_LABEL[t]: {f"설정 {c}": r["hit@3_by_type"][t] for c, r in ret.items()}
                                       for t in ret[next(iter(ret))]["hit@3_by_type"]}), use_container_width=True)
    else:
        st.info("검색 평가 결과가 없습니다. `python eval/run_eval.py --tag baseline` 을 먼저 실행하세요.")

    # 5) 지식베이스 구성
    st.subheader("지식베이스 구성 (중복 제거 후 대표 문서)")
    try:
        docs, chunks = load_kb()
    except Exception as e:   # DB 가 없거나 잠겨 있을 때도 페이지는 열리게
        st.info(f"지식베이스 정보를 읽지 못했습니다: {e}")
        return
    tot = docs.groupby("종류")["n"].sum()
    share = (chunks.set_index("종류")["n"] / chunks["n"].sum() * 100).round(1)
    c = st.columns(4)
    for col, t in zip(c, DOC_TYPES):
        lab = DOC_LABEL[t]
        col.metric(lab, f"{int(tot.get(lab, 0)):,}건", f"청크 비중 {share.get(lab, 0)}%", delta_color="off")
    st.plotly_chart(chart_kb(docs), use_container_width=True)
    st.caption("뉴스가 청크의 93%를 차지해 1차 자료가 검색에서 밀린다 → 실험 2(1차 자료 최소 보장)의 근거.")
    with st.expander("표로 보기"):
        st.dataframe(docs.pivot_table(index="종류", columns="year", values="n", fill_value=0), use_container_width=True)
