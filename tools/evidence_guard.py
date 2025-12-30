#!/usr/bin/env python3
"""
Evidence hygiene guard.

Fail-closed if Markdown evidence artifacts are tracked in-repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


def _git_ls_files(repo_root: Path, pathspec: str) -> List[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", pathspec],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(stderr.strip() or "git ls-files failed")

    raw = result.stdout.split(b"\x00")
    return [
        entry.decode("utf-8", errors="replace")
        for entry in raw
        if entry
    ]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        tracked = _git_ls_files(repo_root, "evidence")
    except RuntimeError as exc:
        print(f"EVIDENCE HYGIENE ERROR: {exc}", file=sys.stderr)
        return 2

    blocked = [
        path
        for path in tracked
        if path.lower().startswith("evidence/")
        and path.lower().endswith(".md")
    ]

    if blocked:
        print("EVIDENCE HYGIENE VIOLATION: Markdown evidence files are tracked.")
        for path in blocked:
            print(f" - {path}")
        if any(path.lower().endswith("recovered_masterpiece.md") for path in blocked):
            print("RECOVERED_MASTERPIECE.md must never be tracked in the repo.")
        print("Move raw evidence outside the repo or convert to hash-only JSON.")
        print("Update EVIDENCE_INDEX.md with the new artifact reference.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
