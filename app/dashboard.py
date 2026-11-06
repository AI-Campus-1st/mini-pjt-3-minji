"""평가 결과 시각화 (Plotly). Day 5 에서 구현한다."""
import streamlit as st


def render() -> None:
    st.title("평가 결과")
    st.info("Day 5 에서 설정별 Hit@k·MRR, 유형별 정확도, 유사도 분포 차트를 추가합니다.")
