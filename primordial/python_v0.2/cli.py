"""BIZRA Genesis Simulation CLI.

Examples:
  python cli.py run-once
  python cli.py lifecycle --env prod
  python cli.py verify-receipt docs/evidence/receipts/EXEC-....json

All outputs are created under this project directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bizra.lifecycle import run_lifecycle
from bizra.ledger import Ledger
from bizra.node_zero import NodeZero


def cmd_run_once(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    node = NodeZero(root)

    payload = {
        "intent": "demo",
        "proposed_action": "noop",
        "risk_level": args.risk,
        "benefit": args.benefit,
        "cost": args.cost,
        "fairness": args.fairness,
        "predicted_ok": True,
        "observed_ok": True,
    }

    res = node.process(
        payload,
        env=args.env,
        artifact_class="execution",
        session_id="demo",
        agent_id="node-0",
        fixed_timestamp_ns=args.fixed_time_ns,
        deterministic_nonce=not args.random_nonce,
    )

    print("=== Node-0 Thought Result ===")
    print(f"committed: {res.committed}")
    print(f"receipt_id: {res.receipt_id}")
    print(f"receipt:    {res.receipt_path}")
    print(f"reason:     {res.reason}")
    print(f"ihsan:      {res.ihsan}")
    print(f"snr:        {res.snr:.6f}")
    return 0


def cmd_lifecycle(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    summary = run_lifecycle(root, env=args.env, fixed_timestamp_ns=args.fixed_time_ns)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_verify_receipt(args: argparse.Namespace) -> int:
    path = Path(args.path)
    data = Ledger.verify_receipt(path)
    print("OK: signature and hash_chain verified")
    print(json.dumps({"receipt_id": data.get("receipt_id"), "type": data.get("type")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bizra-genesis-sim")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="project root")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-once", help="run a single thought")
    p1.add_argument("--env", default="dev", choices=["dev", "ci", "prod"])
    p1.add_argument("--risk", default="0.10")
    p1.add_argument("--benefit", default="0.80")
    p1.add_argument("--cost", default="0.30")
    p1.add_argument("--fairness", default="0.85")
    p1.add_argument("--fixed-time-ns", type=int, default=None)
    p1.add_argument("--random-nonce", action="store_true", help="use os.urandom nonce")
    p1.set_defaults(func=cmd_run_once)

    p2 = sub.add_parser("lifecycle", help="run the multi-phase lifecycle emulation")
    p2.add_argument("--env", default="dev", choices=["dev", "ci", "prod"])
    p2.add_argument("--fixed-time-ns", type=int, default=None)
    p2.set_defaults(func=cmd_lifecycle)

    p3 = sub.add_parser("verify-receipt", help="verify a receipt's signature + hash")
    p3.add_argument("path")
    p3.set_defaults(func=cmd_verify_receipt)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
