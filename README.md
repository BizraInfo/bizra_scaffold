# BIZRA Vanguard Genesis

![Sovereign](https://img.shields.io/badge/sovereign-verified-0b3d2e)
![Ihsan Gate](https://img.shields.io/badge/ihsan%20gate-0.95-0b3d2e)

Golden Seed. Vanguard Node. Genesis-grade bootloader and API.

## What This Is

- Fail-closed bootloader with Ihsan gate enforcement
- Canonical API surface at `core/engine/api.py`
- Physics telemetry via `/status`
- Genesis manifesto in `BLOCK0.md`

## How to Run

Boot the node:

```bash
python -m core.boot
```

Lite mode (no heavy subsystems):

```bash
BIZRA_LITE=1 python -m core.boot
```

Run the API:

```bash
uvicorn core.engine.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` identity
- `GET /status` physics + capabilities + modes
- `GET /health` liveness

## Observability (Physics Monitor)

`/status` returns real-time physics:

- `boot_latency_ms`
- `memory_usage_mb`
- `cpu_usage_percent`
- `ihsan_score`

## Verification (BLOCK0)

Expected SHA256:

```
07e796c0c66bddb8fd5250c75db54d51508a02889c41cbca79a1b7ea36a4e726
```

Verify locally:

```bash
sha256sum BLOCK0.md
```

PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 BLOCK0.md
```

## CI Gate

The Vanguard gate runs a boot dry-run and validates the API telemetry gate.
See `.github/workflows/vanguard-gate.yml`.
