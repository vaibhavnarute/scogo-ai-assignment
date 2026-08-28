"""Reset one deterministic evaluation fixture to its committed baseline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FIXTURE_NAMES = ("F1", "F2", "F3", "F4", "F5")
EVALS_ROOT = Path(__file__).resolve().parent


def reset_fixture(name: str, destination_root: Path | None = None) -> Path:
    if name not in FIXTURE_NAMES:
        raise ValueError(f"unknown fixture {name!r}; choose from {', '.join(FIXTURE_NAMES)}")
    source = EVALS_ROOT / "fixtures" / name
    destination_root = destination_root or EVALS_ROOT / "workspaces"
    destination = destination_root.resolve() / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=FIXTURE_NAMES)
    parser.add_argument("--destination-root", type=Path)
    args = parser.parse_args()
    print(reset_fixture(args.fixture, args.destination_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

