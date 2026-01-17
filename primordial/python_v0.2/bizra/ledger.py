"""Receipt ledger (Third Fact receipts).

Receipts are written as canonical JSON with an Ed25519 signature.
They link into a linear hash chain for tamper-evidence.

Directory layout mirrors the docs:
  docs/evidence/receipts/

Receipt IDs are time-stamped and have a monotonic counter to avoid collisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .canonical import canonical_bytes
from .crypto import Ed25519Keypair, SignatureBundle, sha256_hex
from .hash_chain import HashChain


@dataclass
class ReceiptPaths:
    receipts_dir: Path
    state_dir: Path

    def ensure(self) -> None:
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)


class ReceiptCounter:
    def __init__(self, path: Path):
        self.path = path
        self.count = self._load()

    def _load(self) -> int:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return int(raw.get("count", 0))
        return 0

    def next(self) -> int:
        self.count += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"count": self.count}, sort_keys=True, indent=2), encoding="utf-8")
        return self.count


class Ledger:
    def __init__(self, paths: ReceiptPaths, keypair: Ed25519Keypair):
        self.paths = paths
        self.keypair = keypair
        self.paths.ensure()
        self.counter = ReceiptCounter(self.paths.state_dir / "receipt_counter.json")
        self.hash_chain = HashChain(self.paths.state_dir / "hash_chain.json")

    def _receipt_id(self, prefix: str, timestamp_ns: int) -> str:
        # timestamp_ns -> YYYYMMDDHHMMSS (UTC-ish not strictly, but deterministic enough for naming)
        # We avoid timezone conversion complexity for the sim.
        import datetime

        ts_s = timestamp_ns // 1_000_000_000
        dt = datetime.datetime.utcfromtimestamp(ts_s)
        stamp = dt.strftime("%Y%m%d%H%M%S")
        seq = self.counter.next()
        return f"{prefix}-{stamp}-{seq:06d}"

    def write_receipt(self, receipt_type: str, timestamp_ns: int, body: Dict) -> Path:
        """Write a receipt and return its path.

        `body` should contain all fields except receipt_id and signature.
        """
        prefix = "EXEC" if receipt_type == "execution" else "REJ"
        rid = self._receipt_id(prefix, timestamp_ns)

        unsigned = {
            "receipt_id": rid,
            "type": receipt_type,
            **body,
            "timestamp_ns": int(timestamp_ns),
        }

        unsigned_bytes = canonical_bytes(unsigned)
        sig = SignatureBundle.sign(self.keypair, unsigned_bytes)

        prev_hash, cur_hash = self.hash_chain.step(unsigned_bytes)

        signed = {
            **unsigned,
            "hash_chain": {"prev": prev_hash, "current": cur_hash},
            "signature": {
                "scheme": sig.scheme,
                "public_key_b64": sig.public_key_b64,
                "signature_b64": sig.signature_b64,
            },
        }

        out_path = self.paths.receipts_dir / f"{rid}.json"
        out_path.write_text(json.dumps(signed, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    @staticmethod
    def verify_receipt(path: Path) -> Dict:
        """Verify signature and hash-chain step integrity for a single receipt.

        This doesn't validate chain continuity across multiple receipts;
        that's done by replaying sequentially.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        sig = raw.get("signature")
        if not sig:
            raise ValueError("missing signature")

        # Rebuild unsigned payload.
        unsigned = {k: v for k, v in raw.items() if k not in ("signature", "hash_chain")}
        unsigned_bytes = canonical_bytes(unsigned)
        SignatureBundle(
            scheme=sig["scheme"],
            public_key_b64=sig["public_key_b64"],
            signature_b64=sig["signature_b64"],
        ).verify(unsigned_bytes)

        # Validate chain hash.
        chain = raw.get("hash_chain")
        if not chain:
            raise ValueError("missing hash_chain")
        prev = chain["prev"]
        cur = chain["current"]
        expected = sha256_hex(bytes.fromhex(prev) + unsigned_bytes)
        if expected != cur:
            raise ValueError("hash_chain mismatch")

        return raw
