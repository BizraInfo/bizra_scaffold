"""Lifecycle emulation.

The big "complete lifecycle" text in your artifact outlines a multi-phase story:
- audit baseline
- security hardening
- scaling to multiple nodes
- enterprise readiness

We can't actually spin up a million nodes in this sandbox, but we *can*:
  1) represent each phase as a set of concrete, verifiable change proposals
  2) run them through the same 8-stage Node-0 pipeline
  3) produce receipts + an end-of-run report (SNR, acceptance rate, etc.)

The output is meant to be inspected and repeated. Nothing is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .node_zero import NodeZero, ThoughtResult


@dataclass
class PhaseResult:
    name: str
    thoughts: List[ThoughtResult]

    def summary(self) -> dict:
        commits = sum(1 for t in self.thoughts if t.committed)
        rejects = len(self.thoughts) - commits
        return {
            "phase": self.name,
            "total": len(self.thoughts),
            "committed": commits,
            "rejected": rejects,
        }


def run_lifecycle(root: Path, env: str = "dev", fixed_timestamp_ns: int | None = None) -> Dict:
    node = NodeZero(root)

    phases: List[Tuple[str, List[Dict]]] = [
        (
            "PHASE 0: AUDIT BASELINE",
            [
                {
                    "intent": "audit_baseline",
                    "proposed_action": "snapshot_metrics",
                    "risk_level": "0.05",
                    "benefit": "0.80",
                    "cost": "0.30",
                    "fairness": "0.85",
                    "predicted_ok": True,
                    "observed_ok": True,
                }
            ],
        ),
        (
            "PHASE 1: FOUNDATION HARDENING",
            [
                {
                    "intent": "security_patch",
                    "proposed_action": "enable_memory_protection",
                    "risk_level": "0.20",
                    "benefit": "0.90",
                    "cost": "0.40",
                    "fairness": "0.90",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
                {
                    "intent": "testing_increase",
                    "proposed_action": "add_chaos_tests",
                    "risk_level": "0.15",
                    "benefit": "0.85",
                    "cost": "0.50",
                    "fairness": "0.90",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
            ],
        ),
        (
            "PHASE 2: SCALABILITY",
            [
                {
                    "intent": "sharding_design",
                    "proposed_action": "enable_shard_router",
                    "risk_level": "0.35",
                    "benefit": "0.88",
                    "cost": "0.55",
                    "fairness": "0.92",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
                {
                    "intent": "consensus_upgrade",
                    "proposed_action": "increase_validators",
                    "risk_level": "0.30",
                    "benefit": "0.86",
                    "cost": "0.60",
                    "fairness": "0.93",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
            ],
        ),
        (
            "PHASE 3: ENTERPRISE READINESS",
            [
                {
                    "intent": "compliance_report",
                    "proposed_action": "generate_audit_pack",
                    "risk_level": "0.10",
                    "benefit": "0.82",
                    "cost": "0.35",
                    "fairness": "0.95",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
                # Include one intentionally borderline change to show gating.
                {
                    "intent": "aggressive_perf_tuning",
                    "proposed_action": "disable_safety_checks",
                    "risk_level": "0.70",
                    "benefit": "0.60",
                    "cost": "0.10",
                    "fairness": "0.40",
                    "predicted_ok": True,
                    "observed_ok": True,
                },
            ],
        ),
    ]

    phase_results: List[PhaseResult] = []
    all_thoughts: List[ThoughtResult] = []

    ts = fixed_timestamp_ns
    for phase_name, payloads in phases:
        thoughts: List[ThoughtResult] = []
        for p in payloads:
            # Keep timestamps increasing when fixed_timestamp_ns is provided.
            if ts is not None:
                ts += 1_000_000  # +1ms
            tr = node.process(
                p,
                env=env,
                artifact_class="execution",
                session_id="lifecycle",
                agent_id="node-0",
                fixed_timestamp_ns=ts,
                deterministic_nonce=True,
            )
            thoughts.append(tr)
            all_thoughts.append(tr)
        phase_results.append(PhaseResult(name=phase_name, thoughts=thoughts))

    summary = {
        "policy_hash": node.policy_hash,
        "env": env,
        "phases": [pr.summary() for pr in phase_results],
        "snr": node.snr.to_dict(),
        "receipts_dir": str((root / "docs" / "evidence" / "receipts").resolve()),
    }

    return summary
