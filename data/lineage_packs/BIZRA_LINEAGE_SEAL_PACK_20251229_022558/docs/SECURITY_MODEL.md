# BIZRA Security Model (Kernel)

## Minimum threat model
- Tool misuse or prompt injection leading to unauthorized actions.
- Data exfiltration via logs, traces, or responses.
- Supply-chain compromise (dependencies or artifacts).
- Insider abuse.
- Metrics gaming (PoI or impact inflation).

## Security invariants
1. No tool call without policy decision.
2. No secrets in logs.
3. Every output has a receipt and hashes.
4. Reproducible builds (lockfiles, pinned toolchains).

## Immediate TODO
- Add `/policies` and wire OPA/Rego.
- Add secret scanning.
- Add SBOM generation and signing.
