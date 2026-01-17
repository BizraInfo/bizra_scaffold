# BIZRA Genesis Seed (v0.1)

This is a **seed** repository scaffold that turns the BIZRA vision into a runnable, testable, auditable nucleus.

## What this seed proves (today)

- A **Covenant/Constitution** exists as a root document.
- A minimal **Sense → Reason → Verify → Ledger** loop is runnable (software-only).
- Every action produces an **evidence chain** (hash-linked log) and an **Ihsan score**.

## Run (locally)

```bash
cargo run -p node-zero
```

## Next steps

1. Replace the in-memory signatures with hardware-backed keys (TPM / Secure Boot / OTP).
2. Replace the stubbed `proof` with a real zk proof pipeline (ezkl or a zkVM).
3. Attach a real sensor (camera / GPIO / API) to drive the Sense stream.

