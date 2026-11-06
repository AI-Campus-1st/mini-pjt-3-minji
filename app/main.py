"""④ Streamlit 진입점. 실행: streamlit run app/main.py

챗봇 페이지(chat.py)와 평가 결과 페이지(dashboard.py)로 구성한다. 조회 전용
(임베딩 생성·크롤링·인덱스 쓰기를 하지 않는다. 가이드 4-2).
"""
import sys
from pathlib import Path

# `streamlit run app/main.py` 는 app/ 폴더를 기준으로 실행되므로,
# 프로젝트 루트를 import 경로에 넣어야 settings, rag 패키지를 찾을 수 있다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import chat, dashboard  # noqa: E402

st.set_page_config(page_title="AI 반도체 리서치 챗봇", page_icon="🔎", layout="wide")

nav = st.navigation([
    st.Page(chat.render, title="챗봇", icon="💬", url_path="chat", default=True),
    st.Page(dashboard.render, title="평가 결과", icon="📊", url_path="dashboard"),
])
nav.run()
