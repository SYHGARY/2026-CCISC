from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.annotation import write_annotation_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-annotation packet for LogicGuard."
    )
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--output",
        default="data/annotation_seed.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="docs/ANNOTATION_PROTOCOL.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_annotation_packet(
        output_path=PROJECT_ROOT / args.output,
        manifest_path=PROJECT_ROOT / args.manifest,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Annotation seed saved: {PROJECT_ROOT / args.output}")
    print(f"Protocol manifest saved: {PROJECT_ROOT / args.manifest}")


if __name__ == "__main__":
    main()
