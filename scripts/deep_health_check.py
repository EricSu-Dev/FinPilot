"""Run FinPilot deep health checks from a server shell.

Usage:
    python scripts/deep_health_check.py
    python scripts/deep_health_check.py --external
    python scripts/deep_health_check.py --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_PYTHON = (
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else PROJECT_ROOT / ".venv" / "bin" / "python"
)
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

from app.health import run_deep_health_check  # noqa: E402


def _print_text_report(result: dict) -> None:
    print(f"FinPilot deep health: {result['status']}")
    print(f"external checks: {'enabled' if result.get('external') else 'skipped'}")
    print()

    for item in result["checks"]:
        duration = item.get("duration_ms")
        duration_text = f" ({duration}ms)" if duration is not None else ""
        print(f"[{item['status'].upper()}] {item['name']}{duration_text}")
        print(f"  {item['message']}")
        data = item.get("data") or {}
        for key, value in data.items():
            print(f"  - {key}: {value}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FinPilot deep health checks.")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Also test TCP connectivity to external providers. This does not call paid APIs.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    args = parser.parse_args()

    result = run_deep_health_check(external=args.external)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _print_text_report(result)

    return 0 if result["status"] in {"healthy", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
