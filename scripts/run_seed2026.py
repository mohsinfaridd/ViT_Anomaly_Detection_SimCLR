#!/usr/bin/env python3
"""Convenience wrapper: Protocol A for seed 2026."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "seed2026.yaml"


def main() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_protocol_a.py"), "--config", str(CONFIG)]
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
