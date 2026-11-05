# AI 반도체 산업 동향·기업 전략 분석 RAG 챗봇

미니프로젝트 3 — 공공 문서·웹검색 결합 RAG 기반 질의응답 시스템 (주제 B. 산업·기술 동향 / AI 반도체)

## 대상 사용자
반도체·IT 기업 전략기획/산업분석 주니어 담당자가 뉴스·공시·보고서를 교차 확인하는 리서치 지원 도구

## 계층 구조
| 계층 | 디렉터리 | 실행 방식 |
|---|---|---|
| ① 수집기 | `collector/` | 배치 |
| ② 인덱서 | `indexer/` | 배치 |
| ③ 저장소 | `store/` | 상시 존재 |
| ④ 챗봇 | `app/` (체인은 `rag/` 공유) | 상시 실행 |
| 평가 하네스 | `eval/` (`rag/` 동일 체인 호출) | 배치 |

## 실행 순서
```bash
pip install -r requirements.txt
cp .env.example .env        # 키 입력 (커밋 금지)
python -m collector         # ① 제공 데이터 적재 + 직접 수집
python -m indexer           # ② 파싱·중복처리·청킹·임베딩 → store/
streamlit run app/main.py   # ④ 챗봇 (인덱서 재실행 없이 단독 구동)
python eval/run_eval.py     # 평가
```

## 데이터
- 제공 데이터: `data/raw/` (AI 반도체 뉴스, 저장소 미포함)
- 직접 수집: DART 사업보고서, 정부 보도자료(korea.kr), 산업보고서(KIET)
