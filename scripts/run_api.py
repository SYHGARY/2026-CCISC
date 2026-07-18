from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed. Run `python -m pip install -e .` "
            "or use `python scripts/demo_server.py` for the standard-library fallback."
        ) from exc
    uvicorn.run("llm_logic_guard.api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
