# BIZRA Genesis Simulation (v0.2)

This is a **deterministic, verifiable prototype** of the "mature" BIZRA Genesis pipeline you described:

- 8D Ihsan scoring (Fixed64 Q32.32)
- environment-aware thresholds (dev/ci/prod)
- evidence envelope + anti-replay (nonce + counter + TTL)
- FATE gate (formal-ish constraints)
- SAT consensus simulation (6 validators + veto)
- signed Third Fact receipts (Ed25519) + linear hash chain
- SNR monitor

It is intentionally implemented in **pure Python** to run in this sandbox where Rust isn't available.
The code is structured so you can port the core logic to Rust later with minimal ambiguity.

## Quick start

From the project root:

```bash
python cli.py run-once --env dev
python cli.py lifecycle --env dev
```

Receipts are written to:

```
docs/evidence/receipts/
```

Verify a receipt:

```bash
python cli.py verify-receipt docs/evidence/receipts/EXEC-....json
```

## Determinism notes

- All scoring and thresholding uses Q32.32 fixed-point (no floats).
- JSON hashing uses stable canonical JSON output.
- Consensus weights are integers (centiweight units).

## Security disclaimer

This is a *simulation* and should not be used as-is for production cryptography or distributed consensus.
The point is to make the **execution semantics** explicit and inspectable.
