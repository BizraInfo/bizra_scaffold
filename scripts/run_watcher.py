#!/usr/bin/env python3
"""
BIZRA AEON OMEGA - Data Lake Watcher CLI
═══════════════════════════════════════════════════════════════════════════════
Command-line interface for managing BIZRA data lake and knowledge directories.

Usage:
    python scripts/run_watcher.py --scan           # Initial scan
    python scripts/run_watcher.py --verify         # Verify against manifest
    python scripts/run_watcher.py --watch          # Continuous monitoring
    python scripts/run_watcher.py --status         # Show current status
    python scripts/run_watcher.py --score          # Compute SNR scores

BIZRA SOT Compliance:
    - Section 3 (Invariants): IM ≥ 0.95 enforced
    - Section 7 (Evidence Policy): All changes logged
    - Section 8 (Change Control): Version-tracked manifests
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_lake_watcher import DataLakeWatcher, FileChange, create_default_watcher
from core.data_lake_config import get_config, validate_paths


def print_header(title: str) -> None:
    """Print formatted header."""
    print()
    print("═" * 70)
    print(f" {title}")
    print("═" * 70)
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("═" * 70)
    print()


def change_listener(change: FileChange) -> None:
    """Log file changes as they occur."""
    icons = {
        "CREATED": "➕",
        "MODIFIED": "📝",
        "DELETED": "❌",
        "MOVED": "📦",
        "HASH_MISMATCH": "⚠️",
    }
    icon = icons.get(change.change_type.name, "?")
    print(f"  {icon} [{change.change_type.name}] {change.asset.relative_path}")


async def cmd_scan(watcher: DataLakeWatcher, args: argparse.Namespace) -> int:
    """Perform initial scan of all watched paths."""
    print_header("DATA LAKE SCAN")
    
    # Register change listener for real-time output
    watcher.add_change_listener(change_listener)
    
    print("Scanning watched paths...")
    for wp in watcher.list_watched_paths():
        status = "✅" if wp["exists"] else "⚠️ NOT FOUND"
        print(f"  • {wp['alias']}: {wp['path']} {status}")
    print()
    
    # Perform scan
    changes = await watcher.scan_all()
    
    print()
    print(f"Scan complete. {len(changes)} changes detected.")
    print()
    
    # Score all assets
    if watcher.enable_snr_scoring:
        print("Computing SNR scores...")
        distribution = await watcher.score_all_assets()
        print(f"  Quality Distribution:")
        for quality, count in distribution.items():
            print(f"    {quality}: {count}")
        print()
    
    # Save manifest
    manifest_path = watcher.save_manifest()
    print(f"Manifest saved: {manifest_path}")
    
    # Print summary
    summary = watcher.get_summary()
    print()
    print("Summary:")
    print(f"  Total Assets: {summary['total_assets']}")
    print(f"  Total Size: {summary['total_size_human']}")
    
    return 0


async def cmd_verify(watcher: DataLakeWatcher, args: argparse.Namespace) -> int:
    """Verify current state against saved manifest."""
    print_header("MANIFEST VERIFICATION")
    
    # Load manifest
    if not watcher.load_manifest():
        print("❌ No manifest found. Run --scan first.")
        return 1
    
    print(f"Loaded manifest with {len(watcher.assets)} assets")
    print()
    
    # Verify
    print("Verifying file system against manifest...")
    report = watcher.verify_manifest()
    
    print()
    print("Verification Results:")
    print(f"  Integrity Score: {report['integrity_score']:.1%}")
    print()
    print(f"  ✅ Matched: {len(report['matched'])}")
    print(f"  📝 Modified: {len(report['modified'])}")
    print(f"  ❌ Missing: {len(report['missing'])}")
    print(f"  ➕ New: {len(report['new'])}")
    
    # Show details if issues found
    if report['modified']:
        print()
        print("Modified files:")
        for m in report['modified'][:10]:  # Limit output
            print(f"    {m['path']}")
        if len(report['modified']) > 10:
            print(f"    ... and {len(report['modified']) - 10} more")
    
    if report['missing']:
        print()
        print("Missing files:")
        for m in report['missing'][:10]:
            print(f"    {m}")
        if len(report['missing']) > 10:
            print(f"    ... and {len(report['missing']) - 10} more")
    
    if report['new']:
        print()
        print("New files (not in manifest):")
        for n in report['new'][:10]:
            print(f"    {n}")
        if len(report['new']) > 10:
            print(f"    ... and {len(report['new']) - 10} more")
    
    # Determine exit code
    if report['integrity_score'] >= 0.95:
        print()
        print("✅ VERIFICATION PASSED")
        return 0
    else:
        print()
        print("⚠️ VERIFICATION ISSUES DETECTED")
        return 1


async def cmd_status(watcher: DataLakeWatcher, args: argparse.Namespace) -> int:
    """Show current watcher status."""
    print_header("DATA LAKE STATUS")
    
    # Check path existence
    print("Watched Paths:")
    path_status = validate_paths()
    for alias, exists in path_status.items():
        status = "✅ EXISTS" if exists else "⚠️ NOT FOUND"
        print(f"  • {alias}: {status}")
    print()
    
    # Load manifest if exists
    if watcher.load_manifest():
        summary = watcher.get_summary()
        
        print("Manifest Status:")
        print(f"  Total Assets: {summary['total_assets']}")
        print(f"  Total Size: {summary['total_size_human']}")
        print()
        
        print("Quality Distribution:")
        for quality, count in summary['quality_distribution'].items():
            if count > 0:
                pct = (count / max(1, summary['total_assets'])) * 100
                print(f"    {quality}: {count} ({pct:.1f}%)")
        print()
        
        print("File Type Distribution (top 10):")
        types = sorted(
            summary['file_type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for ftype, count in types:
            print(f"    {ftype or 'unknown'}: {count}")
        print()
        
        if summary['stats']['last_scan_time']:
            print(f"Last Scan: {summary['stats']['last_scan_time']}")
            print(f"Scan Duration: {summary['stats']['last_scan_duration_ms']:.2f}ms")
    else:
        print("No manifest found. Run --scan to create one.")
    
    return 0


async def cmd_watch(watcher: DataLakeWatcher, args: argparse.Namespace) -> int:
    """Start continuous watching."""
    print_header("CONTINUOUS WATCH MODE")
    
    interval = args.interval
    print(f"Starting watch loop (interval: {interval}s)")
    print("Press Ctrl+C to stop")
    print()
    
    # Register change listener
    watcher.add_change_listener(change_listener)
    
    # Load existing manifest
    watcher.load_manifest()
    
    # Start watching
    await watcher.start_watching(interval_seconds=interval)
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print()
        print("Stopping watcher...")
        await watcher.stop_watching()
        
        # Save final manifest
        watcher.save_manifest()
        print("Manifest saved.")
    
    return 0


async def cmd_score(watcher: DataLakeWatcher, args: argparse.Namespace) -> int:
    """Compute SNR scores for all assets."""
    print_header("SNR QUALITY SCORING")
    
    # Load manifest
    if not watcher.load_manifest():
        print("No manifest found. Running scan first...")
        await watcher.scan_all()
    
    # Score all assets
    print(f"Scoring {len(watcher.assets)} assets...")
    distribution = await watcher.score_all_assets()
    
    print()
    print("Quality Distribution:")
    total = sum(distribution.values())
    for quality, count in sorted(distribution.items()):
        pct = (count / max(1, total)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {quality:10s}: {count:5d} ({pct:5.1f}%) {bar}")
    print()
    
    # Show top HIGH quality assets
    high_assets = [
        a for a in watcher.assets.values()
        if a.quality.name in ("CRITICAL", "HIGH")
    ]
    high_assets.sort(key=lambda x: x.snr_score, reverse=True)
    
    if high_assets:
        print("Top HIGH Quality Assets:")
        for asset in high_assets[:15]:
            print(f"  {asset.snr_score:.3f} | {asset.relative_path}")
    print()
    
    # Show LOW quality candidates for review
    low_assets = [
        a for a in watcher.assets.values()
        if a.quality.name == "LOW"
    ]
    
    if low_assets:
        print(f"LOW Quality Assets (noise candidates): {len(low_assets)}")
        for asset in low_assets[:10]:
            print(f"  {asset.snr_score:.3f} | {asset.relative_path}")
        if len(low_assets) > 10:
            print(f"  ... and {len(low_assets) - 10} more")
    
    # Save updated manifest
    watcher.save_manifest()
    print()
    print("Manifest updated with SNR scores.")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Data Lake Watcher CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_watcher.py --scan           # Initial scan
    python scripts/run_watcher.py --verify         # Verify integrity
    python scripts/run_watcher.py --status         # Show status
    python scripts/run_watcher.py --watch          # Continuous monitoring
    python scripts/run_watcher.py --score          # Compute SNR scores
        """
    )
    
    # Commands
    parser.add_argument("--scan", action="store_true", help="Scan all watched paths")
    parser.add_argument("--verify", action="store_true", help="Verify against manifest")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--watch", action="store_true", help="Start continuous watching")
    parser.add_argument("--score", action="store_true", help="Compute SNR scores")
    
    # Options
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
        help="Directory for manifest files"
    )
    parser.add_argument(
        "--no-snr",
        action="store_true",
        help="Disable SNR scoring"
    )
    
    args = parser.parse_args()
    
    # Create watcher
    watcher = create_default_watcher(
        manifest_dir=Path(args.manifest_dir)
    )
    watcher.enable_snr_scoring = not args.no_snr
    
    # Determine command
    if args.scan:
        return asyncio.run(cmd_scan(watcher, args))
    elif args.verify:
        return asyncio.run(cmd_verify(watcher, args))
    elif args.status:
        return asyncio.run(cmd_status(watcher, args))
    elif args.watch:
        return asyncio.run(cmd_watch(watcher, args))
    elif args.score:
        return asyncio.run(cmd_score(watcher, args))
    else:
        # Default to status
        return asyncio.run(cmd_status(watcher, args))


if __name__ == "__main__":
    sys.exit(main())
