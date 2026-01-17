"""Linear hash chain for receipts (blockchain-like linking)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .crypto import sha256_hex


@dataclass
class HashChainState:
    prev_hash: str

    @staticmethod
    def genesis() -> "HashChainState":
        return HashChainState(prev_hash="00" * 32)


class HashChain:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_or_init()

    def _load_or_init(self) -> HashChainState:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return HashChainState(prev_hash=str(raw["prev_hash"]))
        return HashChainState.genesis()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"prev_hash": self.state.prev_hash}, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def step(self, bytes_to_chain: bytes) -> tuple[str, str]:
        prev = self.state.prev_hash
        cur = sha256_hex(bytes.fromhex(prev) + bytes_to_chain)
        self.state.prev_hash = cur
        self._persist()
        return prev, cur
