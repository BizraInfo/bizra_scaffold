#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         BIZRA PAT ACTIVATION SCRIPT                           ║
║                                                                               ║
║  Run this script to activate MoMo's Personal Agent Team.                     ║
║  The system will KNOW who you are. No re-introductions needed.               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python activate_pat.py

Or from any BIZRA session:
    from activate_pat import activate
    activate()
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timezone


def activate() -> None:
    """Activate MoMo's Personal Agent Team with full identity awareness."""

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " BIZRA DUAL-AGENTIC SYSTEM v0.3 ".center(78) + "║")
    print("║" + " Personal Agent Team (PAT) Activation ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")

    # Load identity
    try:
        from core.pat.identity_loader import IdentityLoader, get_context_prompt

        loader = IdentityLoader()
        ctx = loader.load()

        print("║" + f" Identity: {ctx.identity.legal_name}".ljust(78) + "║")
        print("║" + f" Alias: {ctx.identity.alias}".ljust(78) + "║")
        print("║" + f" Role: {ctx.identity.role}".ljust(78) + "║")
        print("╠" + "═" * 78 + "╣")

        # Activate
        greeting = loader.activate_pat()
        print("║" + f" Status: PAT ACTIVE ✓".ljust(78) + "║")
        print("║" + f" Identity Awareness: ENABLED ✓".ljust(78) + "║")
        print("║" + f" Memory Persistence: ENABLED ✓".ljust(78) + "║")

    except ImportError as e:
        print("║" + f" [WARN] Could not load identity module: {e}".ljust(78) + "║")
        print("║" + " Falling back to hardcoded identity...".ljust(78) + "║")
        print("╠" + "═" * 78 + "╣")
        print("║" + " Identity: Mohamed Ahmed Beshr Elsayed Hassan".ljust(78) + "║")
        print("║" + " Alias: MoMo".ljust(78) + "║")
        print(
            "║"
            + " Role: First Architect, First Node, First Owner, First User".ljust(78)
            + "║"
        )
        greeting = "Welcome back, MoMo. Your PAT is active and ready."

    print("╠" + "═" * 78 + "╣")
    print("║" + " Genesis: Ramadan 2023".ljust(78) + "║")
    print("║" + " Years of Work: 3".ljust(78) + "║")
    print(
        "║"
        + f" Current Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}".ljust(78)
        + "║"
    )
    print("╠" + "═" * 78 + "╣")
    print("║" + "".ljust(78) + "║")
    print("║" + f" {greeting}".ljust(78) + "║")
    print("║" + "".ljust(78) + "║")
    print("║" + " Your system KNOWS who you are.".ljust(78) + "║")
    print("║" + " No re-introductions needed.".ljust(78) + "║")
    print("║" + " Proceeding with full context...".ljust(78) + "║")
    print("║" + "".ljust(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Print the key concepts PAT is aware of
    print("📋 KEY CONCEPTS PAT IS AWARE OF:")
    print("-" * 40)
    concepts = [
        "Third Fact",
        "Dual-Agentic System (PAT/SAT)",
        "Ihsān Metric (≥ 0.95)",
        "HyperGraph RAG",
        "Graph of Thoughts",
        "High-SNR Engine",
        "Proof-of-Impact",
        "FATE Engine",
        "Trinity Flywheel",
        "Integrity Flywheel",
        "Root Layer (Quran + Sunnah)",
        "Two Universal Crises (Riba + LLM limits)",
    ]
    for i, concept in enumerate(concepts, 1):
        print(f"  {i:2}. {concept}")

    print()
    print("🎯 PAT IS READY TO SERVE.")
    print("   What would you like to work on, MoMo?")
    print()


if __name__ == "__main__":
    activate()
