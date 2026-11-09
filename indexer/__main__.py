"""② 인덱싱 계층 진입점 (배치).

실행 (프로젝트 루트에서)
    python -m indexer              # 전체 실행
    python -m indexer --only parse # 특정 단계만 실행
    python -m indexer --only dedup --dedup-threshold 0.6   # 중복 기준 변경 실험
    python -m indexer --only embed --config A              # 설정 A 만 임베딩 (맥, OpenAI 키 필요)
    python -m indexer --only keyword --config B            # 하이브리드 검색용 키워드 색인

흐름: parse(정제) → dedup(중복 처리) → chunk(청킹) → embed(임베딩·Chroma 적재) → keyword(BM25 색인)
각 단계는 store/docs.db 의 중간 테이블을 읽고 쓴다. 원문(documents)은 수정하지 않는다.
"""
import argparse

from indexer import chunk, dedup, embed, keyword, parse

# (단계 이름, 설명, 실행 함수)
STEPS = [
    ("parse", "텍스트 정제", parse.run),
    ("dedup", "중복 처리", dedup.run),
    ("chunk", "청킹 (설정 A, B)", chunk.run),
    ("embed", "임베딩 + Chroma 적재", embed.run),
    ("keyword", "키워드 색인 (BM25, 하이브리드 검색용)", keyword.run),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="인덱싱 계층 실행")
    parser.add_argument("--only", choices=[s[0] for s in STEPS], help="한 단계만 실행")
    parser.add_argument("--dedup-threshold", type=float, default=dedup.DEFAULT_THRESHOLD,
                        help=f"재배포본 판정 유사도 (기본 {dedup.DEFAULT_THRESHOLD}, 실험 시 0.6)")
    parser.add_argument("--config", choices=["A", "B", "all"], default="all", help="임베딩할 청킹 설정 (embed 단계)")
    parser.add_argument("--limit", type=int, help="임베딩할 청크 수 제한 (동작 확인용)")
    args = parser.parse_args()
    steps = [s for s in STEPS if args.only in (None, s[0])]
    for i, (name, desc, fn) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {desc}")
        if name == "dedup":
            results = [fn(threshold=args.dedup_threshold)]
        elif name == "embed":
            configs = ["A", "B"] if args.config == "all" else [args.config]
            results = [fn(config=cfg, limit=args.limit) for cfg in configs]
        elif name == "keyword":
            # 챗봇이 쓰는 설정 B 만 기본으로 만든다 (all 이면 A, B 모두)
            configs = ["A", "B"] if args.config == "all" and args.only == "keyword" else [args.config if args.config != "all" else "B"]
            results = [fn(config=cfg) for cfg in configs]
        else:
            results = [fn()]
        for stats in results:
            for k, v in stats.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
