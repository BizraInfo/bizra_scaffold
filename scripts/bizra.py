#!/usr/bin/env python3
"""
BIZRA CLI - Unified Command Interface
═══════════════════════════════════════════════════════════════════════════════
The BIZRA CLI provides a single entry point for all system operations.

Commands:
  bizra status    - Display system health dashboard
  bizra scan      - Scan data lake and update manifest
  bizra verify    - Run compliance verification
  bizra watch     - Start file system monitoring
  bizra score     - Score a file or path for SNR quality

Examples:
  python scripts/bizra.py status
  python scripts/bizra.py status --json
  python scripts/bizra.py scan --verbose
  python scripts/bizra.py verify
  python scripts/bizra.py watch --interval 30
  python scripts/bizra.py score path/to/file.py

BIZRA SOT Compliance:
  - Section 3: IM ≥ 0.95 threshold enforcement
  - Section 7: Evidence logging for all operations
  - Section 8: Version control integration

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.control_center import BIZRAControlCenter

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII ART BANNER
# ═══════════════════════════════════════════════════════════════════════════════


BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗███████╗██████╗  █████╗      ██████╗██╗     ██╗                 ║
║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗    ██╔════╝██║     ██║                 ║
║   ██████╔╝██║  ███╔╝ ██████╔╝███████║    ██║     ██║     ██║                 ║
║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║    ██║     ██║     ██║                 ║
║   ██████╔╝██║███████╗██║  ██║██║  ██║    ╚██████╗███████╗██║                 ║
║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝     ╚═════╝╚══════╝╚═╝                 ║
║                                                                              ║
║   AEON OMEGA - Unified Command Interface v1.0.0                              ║
║   Ihsān Threshold: 0.95 | Fail-Closed Architecture                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════


async def cmd_status(args: argparse.Namespace) -> int:
    """Display system health status."""
    center = BIZRAControlCenter(workspace_path=Path("."))
    health = await center.check_health()

    if args.json:
        print(center.get_health_json())
    else:
        if not args.quiet:
            print(BANNER)
        center.print_status(health)

    if args.monitor:
        print(f"\nStarting continuous monitoring (interval: {args.interval}s)...")
        print("Press Ctrl+C to stop\n")

        await center.start_monitoring(interval_seconds=args.interval)

        try:
            while True:
                await asyncio.sleep(args.interval)
                health = await center.check_health()
                if not args.json:
                    center.print_status(health)
                else:
                    print(center.get_health_json())
        except KeyboardInterrupt:
            await center.stop_monitoring()
            print("\nMonitoring stopped.")

    return 0 if health.healthy else 1


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan data lake and update manifest."""
    if not args.quiet:
        print(BANNER)

    script = Path("scripts/run_watcher.py")
    if not script.exists():
        print("Error: run_watcher.py not found")
        return 1

    cmd = [sys.executable, str(script), "--scan"]
    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd)
    return result.returncode


def cmd_verify(args: argparse.Namespace) -> int:
    """Run compliance verification."""
    if not args.quiet:
        print(BANNER)

    script = Path("scripts/verify_compliance.py")
    if not script.exists():
        print("Error: verify_compliance.py not found")
        return 1

    result = subprocess.run([sys.executable, str(script)])
    return result.returncode


def cmd_watch(args: argparse.Namespace) -> int:
    """Start file system monitoring."""
    if not args.quiet:
        print(BANNER)

    script = Path("scripts/run_watcher.py")
    if not script.exists():
        print("Error: run_watcher.py not found")
        return 1

    cmd = [
        sys.executable,
        str(script),
        "--watch",
        "--interval",
        str(args.interval),
    ]
    if args.verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nWatch stopped.")
        return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a file or path for SNR quality."""
    if not args.quiet:
        print(BANNER)

    script = Path("scripts/run_watcher.py")
    if not script.exists():
        print("Error: run_watcher.py not found")
        return 1

    cmd = [
        sys.executable,
        str(script),
        "--score",
        args.path,
    ]
    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd)
    return result.returncode


def cmd_help(args: argparse.Namespace) -> int:
    """Show help information."""
    print(BANNER)
    print(
        """
BIZRA CLI Commands:
═══════════════════════════════════════════════════════════════════════════════

  status          Display system health dashboard
                  --json          Output as JSON
                  --monitor       Continuous monitoring mode
                  --interval N    Monitor interval in seconds (default: 60)

  scan            Scan data lake paths and update manifest
                  --verbose       Show detailed progress

  verify          Run full compliance verification
                  Checks: SOT, Claims, Evidence, Secrets, Data Lake

  watch           Start real-time file system monitoring
                  --interval N    Poll interval in seconds (default: 10)

  score PATH      Calculate SNR quality score for a file or directory
                  Scores: CRITICAL (>0.90), HIGH (>0.80), MEDIUM (≥0.50), LOW

Global Options:
  -v, --verbose   Enable verbose output
  -q, --quiet     Suppress banner
  --help          Show this help message

Examples:
  bizra status                    # Show health dashboard
  bizra status --json             # JSON output
  bizra status --monitor          # Continuous monitoring
  bizra scan                      # Rescan data lake
  bizra verify                    # Run compliance checks
  bizra watch --interval 30       # Watch with 30s interval
  bizra score core/engine.py      # Score single file
  bizra score core/               # Score directory

═══════════════════════════════════════════════════════════════════════════════
BIZRA AEON OMEGA | Ihsān Threshold: 0.95 | Fail-Closed Architecture
═══════════════════════════════════════════════════════════════════════════════
"""
    )
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for BIZRA CLI."""
    parser = argparse.ArgumentParser(
        prog="bizra",
        description="BIZRA AEON OMEGA - Unified Command Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bizra status              Display system health
  bizra scan               Scan data lake
  bizra verify             Run compliance checks
  bizra watch              Start monitoring
  bizra score FILE         Score file quality
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress banner",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status command
    status_parser = subparsers.add_parser("status", help="Display system health")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.add_argument(
        "--monitor", action="store_true", help="Continuous monitoring"
    )
    status_parser.add_argument(
        "--interval", type=int, default=60, help="Monitor interval (seconds)"
    )

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan data lake")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Run compliance verification")

    # watch command
    watch_parser = subparsers.add_parser("watch", help="Start file monitoring")
    watch_parser.add_argument(
        "--interval", type=int, default=10, help="Poll interval (seconds)"
    )

    # score command
    score_parser = subparsers.add_parser("score", help="Score file quality")
    score_parser.add_argument("path", help="File or directory to score")

    # help command
    help_parser = subparsers.add_parser("help", help="Show help")

    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.verbose)

    # Route to command handler
    if args.command == "status":
        return asyncio.run(cmd_status(args))
    elif args.command == "scan":
        return cmd_scan(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "watch":
        return cmd_watch(args)
    elif args.command == "score":
        return cmd_score(args)
    elif args.command == "help" or args.command is None:
        return cmd_help(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
