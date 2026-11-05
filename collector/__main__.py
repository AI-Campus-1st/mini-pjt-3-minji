"""① 수집 계층 진입점.

실행 (프로젝트 루트에서)
    python -m collector              # 전체 실행
    python -m collector --only dart  # 특정 단계만 실행 (provided / dart / press / kiet)

제공 데이터 적재(load_provided)와 직접 수집(crawl_*)을 순서대로 실행하고
결과 원문을 storage 모듈을 통해 store/docs.db 에 저장한다.
차트·UI 코드를 포함하지 않는다.
"""
import argparse

from collector import crawl_dart, crawl_korea_kr, load_kiet, load_provided
from collector.storage import count_by_type

# (단계 이름, 설명, 실행 함수) — 새 수집기는 여기에 추가
STEPS = [
    ("provided", "제공 데이터 적재", load_provided.run),
    ("dart", "DART 사업보고서 수집", crawl_dart.run),
    ("press", "정책브리핑 보도자료 수집", crawl_korea_kr.run),
    ("kiet", "KIET 보고서 적재 (수동 다운로드분)", load_kiet.run),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="수집 계층 실행")
    parser.add_argument("--only", choices=[s[0] for s in STEPS], help="한 단계만 실행")
    args = parser.parse_args()

    steps = [s for s in STEPS if args.only in (None, s[0])]
    for i, (name, desc, fn) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {desc}")
        stats = fn()
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print("문서 종류별 저장 건수:", count_by_type())


if __name__ == "__main__":
    main()
