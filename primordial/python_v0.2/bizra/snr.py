"""SNR monitor.

SNR = (signal + eps) / (noise + eps)
where signal means committed actions, noise means rejected/rolled-back actions.

This is a deliberately simple KPI to keep the prototype grounded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SNRState:
    signal: int
    noise: int
    eps: int

    @staticmethod
    def empty(eps: int) -> "SNRState":
        return SNRState(signal=0, noise=0, eps=eps)


class SNRMonitor:
    def __init__(self, path: Path, eps: int = 1):
        self.path = path
        self.state = self._load_or_init(eps)

    def _load_or_init(self, eps: int) -> SNRState:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return SNRState(signal=int(raw.get("signal", 0)), noise=int(raw.get("noise", 0)), eps=int(raw.get("eps", eps)))
        return SNRState.empty(eps)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state.__dict__, sort_keys=True, indent=2), encoding="utf-8")

    def add_signal(self, n: int = 1) -> None:
        self.state.signal += int(n)
        self._persist()

    def add_noise(self, n: int = 1) -> None:
        self.state.noise += int(n)
        self._persist()

    def snr(self) -> float:
        # This is a reporting value; determinism isn't critical here.
        return (self.state.signal + self.state.eps) / (self.state.noise + self.state.eps)

    def to_dict(self) -> dict:
        return {
            "signal": self.state.signal,
            "noise": self.state.noise,
            "eps": self.state.eps,
            "snr": round(self.snr(), 6),
        }
