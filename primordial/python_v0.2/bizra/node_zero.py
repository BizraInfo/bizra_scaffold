"""Node-0 (Genesis) thought lifecycle implementation.

This is a runnable, verifiable *simulation* of the mature 8-stage pipeline:
  1) SENSE
  2) REASON
  3) SCORE (8D Ihsan)
  4) FATE (formal-ish gate)
  5) SAT consensus (6 validators + veto)
  6) ACT / ROLLBACK
  7) LEDGER (signed receipt + hash chain)
  8) SNR update

The goal is to be deterministic where it matters (scoring, hashing, receipts),
while remaining small enough to iterate quickly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from .canonical import canonical_bytes
from .consensus import Validator, parse_weight_centi, run_consensus
from .crypto import Ed25519Keypair, sha256_hex
from .evidence import Envelope, ReplayGuard, now_ns, payload_hash
from .fate import fate_check
from .fixed import Fixed64
from .ihsan import IhsanEngine
from .ledger import Ledger, ReceiptPaths
from .snr import SNRMonitor


@dataclass
class ThoughtResult:
    committed: bool
    receipt_path: Path
    receipt_id: str
    reason: str
    ihsan: str
    snr: float


class NodeZero:
    def __init__(self, root: Path):
        self.root = root
        self.constitution_path = root / "constitution.yaml"
        if not self.constitution_path.exists():
            raise FileNotFoundError(f"Missing constitution.yaml at {self.constitution_path}")

        self.policy_hash = sha256_hex(self.constitution_path.read_bytes())

        # Load config.
        cfg = yaml.safe_load(self.constitution_path.read_text(encoding="utf-8"))

        # Ihsan engine.
        self.ihsan_engine = IhsanEngine.load_from_constitution(self.constitution_path)

        # Consensus config.
        cons = cfg.get("consensus", {})
        self.validators = [
            Validator(
                role=str(v["role"]),
                weight_centi=parse_weight_centi(v.get("weight", 1.0)),
                veto=bool(v.get("veto", False)),
            )
            for v in cons.get("validators", [])
        ]
        self.pass_ratio_required = Fixed64.from_decimal_str(str(cons.get("pass_ratio", "0.70")))

        # Replay guard.
        ttl = int(cfg.get("replay_guard", {}).get("nonce_ttl_ns", 600_000_000_000))
        self.replay_guard = ReplayGuard(root / "state" / "replay_guard.json", ttl)

        # Keys.
        self.keypair = Ed25519Keypair.load_or_generate(root / "state" / "node0_ed25519.key")

        # Ledger.
        self.ledger = Ledger(
            paths=ReceiptPaths(receipts_dir=root / "docs" / "evidence" / "receipts", state_dir=root / "state"),
            keypair=self.keypair,
        )

        # SNR.
        eps = int(cfg.get("snr", {}).get("eps", 1))
        self.snr = SNRMonitor(root / "state" / "snr.json", eps=eps)

    def _next_counter(self, session_id: str) -> int:
        last = self.replay_guard.state.session_counters.get(session_id, 0)
        return int(last) + 1

    def _deterministic_nonce(self, session_id: str, counter: int, payload_hash_hex: str) -> str:
        return sha256_hex(f"{session_id}:{counter}:{payload_hash_hex}".encode("utf-8"))[:32]

    def process(
        self,
        payload_obj: Dict,
        *,
        env: str = "dev",
        artifact_class: str = "execution",
        session_id: str = "session-0",
        agent_id: str = "node-0",
        fixed_timestamp_ns: Optional[int] = None,
        deterministic_nonce: bool = True,
    ) -> ThoughtResult:
        """Run the 8-stage pipeline and produce a signed receipt."""

        # 1) SENSE
        timestamp_ns = int(fixed_timestamp_ns) if fixed_timestamp_ns is not None else now_ns()
        p_hash = payload_hash(payload_obj)
        thought_id = sha256_hex((p_hash + str(timestamp_ns)).encode("utf-8"))

        counter = self._next_counter(session_id)
        nonce = self._deterministic_nonce(session_id, counter, p_hash) if deterministic_nonce else None
        if nonce is None:
            from .evidence import random_nonce

            nonce = random_nonce()

        env_threshold = self.ihsan_engine.threshold_for(env, artifact_class=artifact_class)

        envlp = Envelope(
            policy_hash=self.policy_hash,
            session_id=session_id,
            agent_id=agent_id,
            nonce=nonce,
            counter=counter,
            timestamp_ns=timestamp_ns,
            payload_hash=p_hash,
        )

        ok, replay_reason = self.replay_guard.check_and_mark(session_id, nonce, counter, timestamp_ns)
        if not ok:
            # Immediate rejection receipt.
            receipt_path = self.ledger.write_receipt(
                "rejection",
                timestamp_ns,
                body={
                    "thought_id": thought_id,
                    "envelope": envlp.to_dict(),
                    "ihsan_score": None,
                    "fate_result": "UNSAT",
                    "fate_reason": replay_reason,
                    "sat_consensus": None,
                    "action": None,
                    "decision": "rejected",
                    "reason": replay_reason,
                },
            )
            self.snr.add_noise(1)
            rid = receipt_path.stem
            return ThoughtResult(
                committed=False,
                receipt_path=receipt_path,
                receipt_id=rid,
                reason=replay_reason,
                ihsan="0.000000",
                snr=self.snr.snr(),
            )

        # 2) REASON
        # For the sim, the "reasoning" is a deterministic proposal.
        proposal = self._reason(payload_obj, thought_id)

        # 3) SCORE (8D Ihsan)
        dim_scores = self._score(payload_obj, proposal)
        ih = self.ihsan_engine.evaluate(dim_scores)

        # Gate 0: if ihsan below threshold -> human veto route.
        if ih.composite < env_threshold:
            receipt_path = self.ledger.write_receipt(
                "rejection",
                timestamp_ns,
                body={
                    "thought_id": thought_id,
                    "envelope": envlp.to_dict(),
                    "ihsan_score": ih.to_dict(),
                    "fate_result": "UNSAT",
                    "fate_reason": f"ihsan_below_threshold({ih.composite.to_decimal_str(6)}<{env_threshold.to_decimal_str(6)})",
                    "sat_consensus": None,
                    "action": proposal,
                    "decision": "rejected",
                    "reason": "human_veto_gate",
                },
            )
            self.snr.add_noise(1)
            rid = receipt_path.stem
            return ThoughtResult(
                committed=False,
                receipt_path=receipt_path,
                receipt_id=rid,
                reason="human_veto_gate",
                ihsan=ih.composite.to_decimal_str(6),
                snr=self.snr.snr(),
            )

        # 4) FATE (formal-ish)
        fate = fate_check(ih.composite, env_threshold, ih.dimensions)
        if not fate.sat:
            receipt_path = self.ledger.write_receipt(
                "rejection",
                timestamp_ns,
                body={
                    "thought_id": thought_id,
                    "envelope": envlp.to_dict(),
                    "ihsan_score": ih.to_dict(),
                    "fate_result": fate.as_str(),
                    "fate_reason": fate.reason,
                    "sat_consensus": None,
                    "action": proposal,
                    "decision": "rejected",
                    "reason": "fate_gate",
                },
            )
            self.snr.add_noise(1)
            rid = receipt_path.stem
            return ThoughtResult(
                committed=False,
                receipt_path=receipt_path,
                receipt_id=rid,
                reason="fate_gate",
                ihsan=ih.composite.to_decimal_str(6),
                snr=self.snr.snr(),
            )

        # 5) SAT consensus
        cons = run_consensus(self.validators, ih.dimensions, self.pass_ratio_required)
        if not cons.passed:
            receipt_path = self.ledger.write_receipt(
                "rejection",
                timestamp_ns,
                body={
                    "thought_id": thought_id,
                    "envelope": envlp.to_dict(),
                    "ihsan_score": ih.to_dict(),
                    "fate_result": fate.as_str(),
                    "fate_reason": fate.reason,
                    "sat_consensus": cons.to_dict(),
                    "action": proposal,
                    "decision": "rejected",
                    "reason": "sat_consensus",
                },
            )
            self.snr.add_noise(1)
            rid = receipt_path.stem
            return ThoughtResult(
                committed=False,
                receipt_path=receipt_path,
                receipt_id=rid,
                reason="sat_consensus",
                ihsan=ih.composite.to_decimal_str(6),
                snr=self.snr.snr(),
            )

        # 6) ACT
        action_result = self._act(payload_obj, proposal)

        # 7) LEDGER
        receipt_path = self.ledger.write_receipt(
            "execution",
            timestamp_ns,
            body={
                "thought_id": thought_id,
                "envelope": envlp.to_dict(),
                "ihsan_score": ih.to_dict(),
                "fate_result": fate.as_str(),
                "fate_reason": fate.reason,
                "sat_consensus": cons.to_dict(),
                "action": proposal,
                "decision": "committed",
                "result": action_result,
            },
        )

        # 8) SNR update
        self.snr.add_signal(1)

        rid = receipt_path.stem
        return ThoughtResult(
            committed=True,
            receipt_path=receipt_path,
            receipt_id=rid,
            reason="committed",
            ihsan=ih.composite.to_decimal_str(6),
            snr=self.snr.snr(),
        )

    # -----------------
    # Stubbed internals
    # -----------------

    def _reason(self, payload_obj: Dict, thought_id: str) -> Dict:
        """Deterministic reasoning stub."""
        # Proposed action is derived from intent.
        intent = str(payload_obj.get("intent", "noop"))
        risk = str(payload_obj.get("risk_level", "0"))
        return {
            "action": payload_obj.get("proposed_action", "noop"),
            "intent": intent,
            "risk_level": risk,
            "trace": {
                "thought_id": thought_id,
                "inputs": sorted(payload_obj.keys()),
                "rule": "deterministic_stub_v1",
            },
        }

    def _score(self, payload_obj: Dict, proposal: Dict) -> Dict[str, Fixed64]:
        """Score each Ihsan dimension in a deterministic, inspectable way."""
        # Correctness: predicted_ok==observed_ok -> 1 else 0
        predicted_ok = bool(payload_obj.get("predicted_ok", True))
        observed_ok = bool(payload_obj.get("observed_ok", True))
        adl = Fixed64.one() if predicted_ok == observed_ok else Fixed64.zero()

        # Safety: 1 - risk_level
        rl = str(payload_obj.get("risk_level", "0.0"))
        try:
            risk = Fixed64.from_decimal_str(rl).clamp01()
        except Exception:
            risk = Fixed64.zero()
        amanah = (Fixed64.one() - risk).clamp01()

        # Benefit
        b = str(payload_obj.get("benefit", "0.5"))
        try:
            ihsan = Fixed64.from_decimal_str(b).clamp01()
        except Exception:
            ihsan = Fixed64.from_decimal_str("0.5")

        # Efficiency: 1 - cost
        c = str(payload_obj.get("cost", "0.5"))
        try:
            cost = Fixed64.from_decimal_str(c).clamp01()
        except Exception:
            cost = Fixed64.from_decimal_str("0.5")
        hikmah = (Fixed64.one() - cost).clamp01()

        # Auditability: based on presence of trace
        bayan = Fixed64.one() if proposal.get("trace") else Fixed64.from_decimal_str("0.5")

        # Anti-centralization: if distributed_hint true -> 1 else 0.7
        tawhid = Fixed64.one() if bool(payload_obj.get("distributed_hint", True)) else Fixed64.from_decimal_str("0.7")

        # Robustness: if rollback_available true -> 1 else 0.6
        sabr = Fixed64.one() if bool(payload_obj.get("rollback_available", True)) else Fixed64.from_decimal_str("0.6")

        # Fairness: provided
        f = str(payload_obj.get("fairness", "0.8"))
        try:
            mizan = Fixed64.from_decimal_str(f).clamp01()
        except Exception:
            mizan = Fixed64.from_decimal_str("0.8")

        return {
            "adl": adl,
            "amanah": amanah,
            "ihsan": ihsan,
            "hikmah": hikmah,
            "bayan": bayan,
            "tawhid": tawhid,
            "sabr": sabr,
            "mizan": mizan,
        }

    def _act(self, payload_obj: Dict, proposal: Dict) -> Dict:
        """Action stub: returns a deterministic result object."""
        return {
            "status": "ok",
            "applied_action": proposal.get("action"),
            "note": "simulation_only",
        }
