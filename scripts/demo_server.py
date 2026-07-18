from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "demo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.demo_api import analyze_payload, repair_payload
from llm_logic_guard.attacks import SCENARIOS, AttackLab


ATTACK_LAB = AttackLab(PROJECT_ROOT / "outputs" / "attack_lab")


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/samples":
            self._send_json({"samples": _load_samples()})
            return
        if path == "/api/v1/scenarios":
            self._send_json({
                "scenarios": [
                    {
                        "id": scenario_id,
                        "attack_type": item["attack_type"],
                        "goal": item["goal"],
                        "candidate_action": item["candidate_action"],
                        "is_attack": scenario_id in {
                            "prompt_injection",
                            "memory_poisoning",
                            "environment_pollution",
                        },
                    }
                    for scenario_id, item in SCENARIOS.items()
                ]
            })
            return
        if path == "/api/v1/metrics":
            report_path = PROJECT_ROOT / "outputs" / "competition_experiment.json"
            payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            self._send_json(payload)
            return

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        target = static_files.get(path)
        if target is None:
            self.send_error(404)
            return
        filename, content_type = target
        content = (DEMO_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/analyze":
                result = analyze_payload(payload)
            elif path == "/api/repair":
                result = repair_payload(payload)
            elif path == "/api/v1/attacks/run":
                scenario_id = str(payload.get("scenario_id") or "prompt_injection")
                if scenario_id not in SCENARIOS:
                    raise ValueError("unknown scenario")
                result = ATTACK_LAB.run(
                    scenario_id,
                    defended=bool(payload.get("defended", True)),
                ).to_dict()
            else:
                self.send_error(404)
                return
            self._send_json(result)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")
        return payload

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _load_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    sources = [
        ("Core examples", PROJECT_ROOT / "data" / "sample_traces.jsonl"),
        ("Converted security log", PROJECT_ROOT / "data" / "converted_security_demo.jsonl"),
    ]
    for group, path in sources:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trace = json.loads(line)
                samples.append({
                    "label": _sample_label(trace),
                    "group": group,
                    "trace": trace,
                })
    return samples


def _sample_label(trace: dict[str, Any]) -> str:
    trace_id = str(trace.get("trace_id") or "Untitled trace")
    return trace_id.replace("case_", "").replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Logic Consistency Guard demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
