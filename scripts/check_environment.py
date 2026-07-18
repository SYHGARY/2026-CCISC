from __future__ import annotations

import importlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "PyYAML": "yaml",
    "httpx": "httpx",
    "z3-solver": "z3",
    "langgraph": "langgraph",
    "langgraph-checkpoint-sqlite": "langgraph.checkpoint.sqlite",
}
OPTIONAL_PACKAGES = {
    "transformers": "transformers",
    "torch": "torch",
    "sentencepiece": "sentencepiece",
}


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "missing"


def check_imports(packages: dict[str, str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for distribution, module_name in packages.items():
        try:
            importlib.import_module(module_name)
            results[distribution] = {
                "available": True,
                "version": package_version(distribution),
            }
        except Exception as exc:
            results[distribution] = {
                "available": False,
                "version": package_version(distribution),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def check_backends() -> dict[str, Any]:
    details: dict[str, Any] = {}

    try:
        import torch

        details["torch"] = {
            "tensor_test": int((torch.tensor([1, 2]) + 1).sum().item()) == 5,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    except Exception as exc:
        details["torch"] = {"tensor_test": False, "error": str(exc)}

    try:
        import z3

        value = z3.Int("environment_check_value")
        solver = z3.Solver()
        solver.add(value > 0, value < 2)
        details["z3"] = {
            "solver_test": solver.check() == z3.sat,
            "version": z3.get_version_string(),
        }
    except Exception as exc:
        details["z3"] = {"solver_test": False, "error": str(exc)}

    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(dict)
        graph.add_node("pass_through", lambda state: state)
        graph.add_edge(START, "pass_through")
        graph.add_edge("pass_through", END)
        graph.compile()
        details["langgraph"] = {"compile_test": True}
    except Exception as exc:
        details["langgraph"] = {"compile_test": False, "error": str(exc)}

    try:
        from llm_logic_guard.api import app

        details["api"] = {
            "import_test": True,
            "route_count": len(app.routes),
        }
    except Exception as exc:
        details["api"] = {"import_test": False, "error": str(exc)}

    return details


def main() -> None:
    imports = check_imports(REQUIRED_PACKAGES)
    optional_imports = check_imports(OPTIONAL_PACKAGES)
    backends = check_backends()
    python_recommended = sys.version_info[:2] == (3, 12)
    python_supported = sys.version_info >= (3, 10)
    imports_ok = all(item["available"] for item in imports.values())
    backend_ok = all(
        backends.get(name, {}).get(key)
        for name, key in {
            "z3": "solver_test",
            "langgraph": "compile_test",
            "api": "import_test",
        }.items()
    )
    report = {
        "passed": python_supported and imports_ok and backend_ok,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "recommended_major_minor": "3.12",
            "supported_minimum": "3.10",
            "recommended": python_recommended,
            "valid": python_supported,
        },
        "platform": platform.platform(),
        "packages": imports,
        "optional_packages": optional_imports,
        "backends": backends,
        "notes": {
            "torch": "CPU mode is valid; CUDA is optional.",
            "transformers": "Transformer NLI is optional. Set RUN_NLI_TESTS=1 and install/cache the model to run model-backed tests.",
            "api_keys": "No API key is required for deterministic checks.",
        },
    }
    output_path = PROJECT_ROOT / "outputs" / "environment_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nEnvironment report: {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
