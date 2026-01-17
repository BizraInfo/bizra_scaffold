"""Evidence envelope + replay protection.

This mirrors the "Evidence Envelope" design:
  - policy_hash binding
  - session scope
  - agent id
  - nonce + monotonic counter
  - timestamp_ns (integer)
  - payload_hash

A ReplayGuard enforces:
  - nonce uniqueness with TTL
  - per-session counter monotonicity

In the production design you'd store this in a shared KV store / ledger.
For this simulation we keep a small persistent JSON state file.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from .canonical import canonical_bytes
from .crypto import sha256_hex


def now_ns() -> int:
    # Python's time.time_ns() is deterministic per platform and returns int.
    return time.time_ns()


def random_nonce(nbytes: int = 16) -> str:
    return os.urandom(nbytes).hex()


@dataclass
class Envelope:
    policy_hash: str
    session_id: str
    agent_id: str
    nonce: str
    counter: int
    timestamp_ns: int
    payload_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayGuardState:
    nonce_ttl_ns: int
    seen_nonces: Dict[str, int]
    session_counters: Dict[str, int]

    @staticmethod
    def empty(nonce_ttl_ns: int) -> "ReplayGuardState":
        return ReplayGuardState(
            nonce_ttl_ns=nonce_ttl_ns,
            seen_nonces={},
            session_counters={},
        )


class ReplayGuard:
    def __init__(self, state_path: Path, nonce_ttl_ns: int):
        self.state_path = state_path
        self.nonce_ttl_ns = int(nonce_ttl_ns)
        self.state = self._load_or_init()

    def _load_or_init(self) -> ReplayGuardState:
        if self.state_path.exists():
            import json

            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ReplayGuardState(
                nonce_ttl_ns=int(raw["nonce_ttl_ns"]),
                seen_nonces={k: int(v) for k, v in raw.get("seen_nonces", {}).items()},
                session_counters={
                    k: int(v) for k, v in raw.get("session_counters", {}).items()
                },
            )
        return ReplayGuardState.empty(self.nonce_ttl_ns)

    def _persist(self) -> None:
        import json

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nonce_ttl_ns": int(self.state.nonce_ttl_ns),
            "seen_nonces": self.state.seen_nonces,
            "session_counters": self.state.session_counters,
        }
        self.state_path.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")

    def _gc(self, now: int) -> None:
        ttl = int(self.state.nonce_ttl_ns)
        expired = [n for n, t in self.state.seen_nonces.items() if (now - t) > ttl]
        for n in expired:
            self.state.seen_nonces.pop(n, None)

    def check_and_mark(self, session_id: str, nonce: str, counter: int, timestamp_ns: int) -> Tuple[bool, str]:
        """Return (ok, reason)."""
        now = int(timestamp_ns)
        self._gc(now)

        if nonce in self.state.seen_nonces:
            return False, "replay_nonce"

        last = self.state.session_counters.get(session_id, -1)
        if counter <= last:
            return False, f"non_monotonic_counter(last={last}, got={counter})"

        self.state.seen_nonces[nonce] = now
        self.state.session_counters[session_id] = int(counter)
        self._persist()
        return True, "ok"


def payload_hash(payload_obj: object) -> str:
    return sha256_hex(canonical_bytes(payload_obj))
