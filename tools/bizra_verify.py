#!/usr/bin/env python3
"""
bizra_verify.py - BUILD-VERIFY-METRICS kernel (zero external deps).

Usage:
  python tools/bizra_verify.py --out evidence --artifact-name bizra_scaffold --artifact-version local
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
}


def run(
    cmd: List[str], cwd: Path | None = None, timeout: int = 1800
) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err + "\nTIMEOUT"
    return proc.returncode, out, err


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_tree_hash(root: Path) -> str:
    """
    Deterministic hash of file paths + content hashes.
    Skips common build/cache directories.
    """
    files: List[Path] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_file():
            files.append(path)
    files.sort(key=lambda item: str(item.relative_to(root)).lower())
    digest = hashlib.sha256()
    for file_path in files:
        rel = str(file_path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def detect_stack(root: Path) -> Dict[str, bool]:
    return {
        "python": (root / "pyproject.toml").exists()
        or (root / "requirements.txt").exists(),
        "node": (root / "package.json").exists(),
        "rust": (root / "Cargo.toml").exists(),
    }


def try_git_commit(root: Path) -> str:
    code, out, _ = run(["git", "rev-parse", "HEAD"], cwd=root)
    return out.strip() if code == 0 else "unknown"


def ihsan_score(vec: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        if key == "threshold":
            continue
        total += vec.get(key, 0.0) * weight
    return max(0.0, min(1.0, total))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="evidence", help="Output directory for receipts/metrics"
    )
    parser.add_argument("--artifact-name", default="bizra", help="Artifact name")
    parser.add_argument("--artifact-version", default="dev", help="Artifact version")
    parser.add_argument(
        "--ihsan-config",
        default="config/ihsan_vector.json",
        help="Ihsan weights config",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail on any skipped critical check"
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    out_dir = Path(args.out).resolve()
    (out_dir / "receipts").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checks").mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    stack = detect_stack(root)
    commit = try_git_commit(root)

    tree_hash = repo_tree_hash(root)
    checks.append(
        {
            "name": "repo_tree_hash",
            "status": "pass",
            "details": {"repo_tree_sha256": tree_hash},
        }
    )

    # Tests
    if stack["python"]:
        tests_dir = root / "tests"
        if tests_dir.exists():
            code, out, err = run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                cwd=root,
                timeout=1800,
            )
            checks.append(
                {
                    "name": "python_unittest",
                    "status": "pass" if code == 0 else "fail",
                    "details": {
                        "code": code,
                        "stdout": out[-4000:],
                        "stderr": err[-4000:],
                    },
                }
            )
        else:
            checks.append(
                {
                    "name": "python_unittest",
                    "status": "skip",
                    "details": {"reason": "no tests/ directory detected"},
                }
            )

    if stack["node"]:
        code, out, err = run(["npm", "test", "--silent"], cwd=root, timeout=1800)
        checks.append(
            {
                "name": "node_tests",
                "status": "pass" if code == 0 else "fail",
                "details": {"code": code, "stdout": out[-4000:], "stderr": err[-4000:]},
            }
        )

    if stack["rust"]:
        code, out, err = run(["cargo", "test", "--locked"], cwd=root, timeout=3600)
        checks.append(
            {
                "name": "cargo_test",
                "status": "pass" if code == 0 else "fail",
                "details": {"code": code, "stdout": out[-4000:], "stderr": err[-4000:]},
            }
        )

    # Audits (optional)
    if stack["rust"]:
        code, out, err = run(["cargo", "audit", "--json"], cwd=root, timeout=1800)
        if out.strip():
            (out_dir / "checks" / "cargo_audit.json").write_text(out, encoding="utf-8")
            checks.append(
                {
                    "name": "cargo_audit",
                    "status": "pass" if code == 0 else "fail",
                    "details": {
                        "code": code,
                        "saved": "checks/cargo_audit.json",
                        "stderr": err[-2000:],
                    },
                }
            )
        else:
            checks.append(
                {
                    "name": "cargo_audit",
                    "status": "skip",
                    "details": {
                        "reason": "cargo-audit not installed or failed",
                        "code": code,
                        "stderr": err[-2000:],
                    },
                }
            )

    if stack["node"]:
        code, out, err = run(["npm", "audit", "--json"], cwd=root, timeout=1800)
        if out.strip():
            (out_dir / "checks" / "npm_audit.json").write_text(out, encoding="utf-8")
            checks.append(
                {
                    "name": "npm_audit",
                    "status": "pass" if code == 0 else "fail",
                    "details": {
                        "code": code,
                        "saved": "checks/npm_audit.json",
                        "stderr": err[-2000:],
                    },
                }
            )
        else:
            checks.append(
                {
                    "name": "npm_audit",
                    "status": "skip",
                    "details": {
                        "reason": "npm audit produced no output",
                        "code": code,
                        "stderr": err[-2000:],
                    },
                }
            )

    if stack["python"]:
        code, out, err = run(["pip-audit", "-f", "json"], cwd=root, timeout=1800)
        if code == 0 and out.strip():
            (out_dir / "checks" / "pip_audit.json").write_text(out, encoding="utf-8")
            checks.append(
                {
                    "name": "pip_audit",
                    "status": "pass",
                    "details": {"saved": "checks/pip_audit.json"},
                }
            )
        else:
            checks.append(
                {
                    "name": "pip_audit",
                    "status": "skip",
                    "details": {
                        "reason": "pip-audit not installed or failed",
                        "code": code,
                        "stderr": err[-2000:],
                    },
                }
            )

    policy_dir = root / "policies"
    if policy_dir.exists():
        policy_hash = repo_tree_hash(policy_dir)
        policy_decision = "allow"
    else:
        policy_hash = "unconfigured"
        policy_decision = "deny"
    policy = {
        "decision": policy_decision,
        "engine": "placeholder",
        "ruleset_hash": policy_hash,
    }

    ihsan_cfg_path = root / args.ihsan_config
    if ihsan_cfg_path.exists():
        cfg = json.loads(ihsan_cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {"threshold": 0.9, "weights": {}}

    total = len([check for check in checks if check["name"] != "repo_tree_hash"])
    passed = len(
        [
            check
            for check in checks
            if check["status"] == "pass" and check["name"] != "repo_tree_hash"
        ]
    )
    strict_skip = args.strict and any(
        check["status"] == "skip" and check["name"] != "repo_tree_hash"
        for check in checks
    )
    any_fail = any(check["status"] == "fail" for check in checks) or strict_skip

    receipt_completeness = 1.0
    correctness = 0.5 + 0.5 * (passed / max(1, total))

    safety = 0.5
    audit_checks = {"cargo_audit", "npm_audit", "pip_audit"}
    if any(
        check["name"] in audit_checks and check["status"] != "skip" for check in checks
    ):
        safety = 0.7
    if any(
        check["name"] in audit_checks and check["status"] == "fail" for check in checks
    ):
        safety = 0.3

    vec = {
        "correctness": float(max(0.0, min(1.0, correctness))),
        "safety": float(max(0.0, min(1.0, safety))),
        "user_benefit": 0.6,
        "efficiency": 0.6,
        "auditability": float(receipt_completeness),
        "anti_centralization": 0.5,
        "robustness": 0.6,
        "adl_fairness": 0.5,
    }
    weights = cfg.get("weights", {})
    threshold = float(cfg.get("threshold", 0.9))
    score = ihsan_score(vec, weights)
    vec_out = {**vec, "threshold": threshold, "score": score}

    metrics = {
        "snr": 2.0,
        "ihsan_vector": vec_out,
        "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
        "error_rate": 0.0 if not any_fail else 1.0,
        "policy_denies": 0 if policy_decision == "allow" else 1,
        "receipt_completeness": receipt_completeness,
        "coverage": 0.0,
    }

    try:
        memory_gb = float(os.environ.get("MEMORY_GB", "0") or 0)
    except ValueError:
        memory_gb = 0.0
    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "node": os.environ.get("NODE_VERSION", "unknown"),
        "rust": os.environ.get("RUST_VERSION", "unknown"),
        "cpu": platform.processor() or "unknown",
        "memory_gb": memory_gb,
    }

    receipt_id = hashlib.sha256(
        f"{args.artifact_name}|{args.artifact_version}|{commit}|{tree_hash}".encode()
    ).hexdigest()
    notes = ["Ihsan and SNR are heuristic until wired to real benchmarks and policies."]
    if strict_skip:
        notes.append("Strict mode failed due to skipped checks.")
    if policy_decision != "allow":
        notes.append("Policy directory missing; failing closed on policy enforcement.")
    receipt = {
        "receipt_id": receipt_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "artifact": {
            "name": args.artifact_name,
            "version": args.artifact_version,
            "commit": commit,
            "path": str(root),
        },
        "checks": checks,
        "policy": policy,
        "metrics": metrics,
        "hashes": {"repo_tree_sha256": tree_hash},
        "environment": env,
        "result": {
            "overall": (
                "pass"
                if (not any_fail and score >= threshold and policy_decision == "allow")
                else "fail"
            ),
            "confidence": 0.75 if not any_fail else 0.4,
            "notes": notes,
            "risks": [
                "Policy engine is placeholder",
                "Ihsan scoring is heuristic",
                "Latency and coverage are not measured yet",
            ],
        },
    }

    (out_dir / "receipts" / f"{receipt_id}.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics" / "latest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"[BIZRA] Receipt: {out_dir / 'receipts' / (receipt_id + '.json')}")
    print(
        f"[BIZRA] Overall: {receipt['result']['overall']} | Ihsan: {score:.3f} (>= {threshold})"
    )
    return 0 if receipt["result"]["overall"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
