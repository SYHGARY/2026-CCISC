from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.converters import convert_security_events, trace_to_dict


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert security event JSONL into LogicTrace JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/converted_security_traces.jsonl")
    parser.add_argument(
        "--user-goal",
        default="",
        help="Explicit goal applied to converted traces. No goal is inferred from safety labels.",
    )
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    events = read_jsonl(input_path)
    traces = convert_security_events(events, user_goal=args.user_goal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(trace_to_dict(trace), ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )

    print(f"Converted events: {len(events)}")
    print(f"Generated traces: {len(traces)}")
    print(f"Generated steps: {sum(len(trace.steps) for trace in traces)}")
    print(f"Output saved: {output_path}")


if __name__ == "__main__":
    main()
