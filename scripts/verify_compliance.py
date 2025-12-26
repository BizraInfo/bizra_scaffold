#!/usr/bin/env python3
"""
BIZRA AEON OMEGA - Self-Compliance Verification
═══════════════════════════════════════════════════════════════════════════════
Verifies that the codebase follows its own documented rules and policies.

Includes:
- SOT Consistency Check
- Claim Registry Audit
- Evidence Artifacts Validation
- Secret Key Audit
- Data Lake Watcher Status (if paths exist)
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime


def check_data_lake_paths() -> dict:
    """
    Check status of BIZRA data lake paths.
    
    Returns dict with path status information.
    """
    paths = [
        ("data_lake", Path("C:/BIZRA-DATA-LAKE")),
        ("node0_knowledge", Path("C:/BIZRA-NODE0/knowledge")),
    ]
    
    result = {
        "paths": [],
        "all_exist": True,
        "total_files": 0,
    }
    
    for alias, path in paths:
        exists = path.exists()
        file_count = 0
        if exists and path.is_dir():
            try:
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            except (PermissionError, OSError):
                file_count = -1
        
        result["paths"].append({
            "alias": alias,
            "path": str(path),
            "exists": exists,
            "file_count": file_count,
        })
        
        if not exists:
            result["all_exist"] = False
        else:
            result["total_files"] += max(0, file_count)
    
    return result


def main():
    print("=" * 70)
    print(" BIZRA AEON OMEGA - Self-Compliance Verification")
    print("=" * 70)
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    issues = []
    
    # 1. SOT Consistency Check
    print("1. SOT IHSAN THRESHOLD CONSISTENCY")
    with open("BIZRA_SOT.md") as f:
        sot = f.read()
    
    # Check both Section 3 and Section 4 have 0.95
    section3 = "0.95" in sot.split("## 3. Invariants")[1].split("## 4.")[0] if "## 3. Invariants" in sot else False
    section4 = "0.95" in sot.split("## 4. PoI Parameters")[1].split("## 5.")[0] if "## 4. PoI Parameters" in sot else False
    
    print(f"   Section 3 (Invariants): 0.95 = {section3}")
    print(f"   Section 4 (PoI Params): 0.95 = {section4}")
    
    if section3 and section4:
        print("   ✅ ALIGNED")
    else:
        print("   ❌ MISMATCH")
        issues.append("SOT ihsan_threshold inconsistency")
    print()
    
    # 2. Claim Registry Audit
    print("2. CLAIM REGISTRY AUDIT")
    with open("evidence/CLAIM_REGISTRY.yaml", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    
    version = registry.get("version", "unknown")
    last_verified = registry.get("last_verified", "never")
    claims = registry.get("claims", [])
    
    verified_count = sum(1 for c in claims if c.get("status") == "VERIFIED")
    null_evidence = sum(1 for c in claims if c.get("status") == "VERIFIED" and not c.get("evidence_artifact_path"))
    
    print(f"   Version: {version}")
    print(f"   Last Verified: {last_verified}")
    print(f"   VERIFIED Claims: {verified_count}")
    print(f"   VERIFIED with null evidence: {null_evidence}")
    
    if null_evidence == 0:
        print("   ✅ COMPLIANT")
    else:
        print("   ❌ EVIDENCE MISSING")
        issues.append(f"{null_evidence} VERIFIED claims lack evidence")
    print()
    
    # 3. Evidence Artifacts
    print("3. EVIDENCE ARTIFACTS")
    evidence_dir = Path("evidence/architecture")
    artifacts = list(evidence_dir.glob("*.log")) if evidence_dir.exists() else []
    
    print(f"   Architecture Evidence: {len(artifacts)} files")
    for a in artifacts:
        print(f"   - {a.name}")
    
    if len(artifacts) >= 2:
        print("   ✅ PRESENT")
    else:
        print("   ⚠️ INCOMPLETE")
        issues.append("Missing evidence artifacts")
    print()
    
    # 4. Secret Key Audit
    print("4. SECRET KEY AUDIT")
    keys_dir = Path("keys")
    
    if not keys_dir.exists():
        print("   Keys directory not found")
        print("   ✅ CLEAN")
    else:
        secret_files = [
            f for f in keys_dir.iterdir() 
            if "secret" in f.name.lower() 
            and not f.name.endswith(".example")
            and f.name != "README.md"
        ]
        
        print(f"   Secret files found: {len(secret_files)}")
        # Security: Don't log actual secret filenames
        if secret_files:
            print(f"   ❌ {len(secret_files)} secret file(s) detected")
        
        if len(secret_files) == 0:
            print("   ✅ CLEAN")
        else:
            print("   ❌ SECRETS FOUND")
            issues.append("Secret files in keys/ directory")
    print()
    
    # 5. Data Lake Watcher Status
    print("5. DATA LAKE WATCHER STATUS")
    dl_status = check_data_lake_paths()
    
    for p in dl_status["paths"]:
        status_icon = "✅" if p["exists"] else "⚠️"
        file_info = f" ({p['file_count']} files)" if p["file_count"] >= 0 else ""
        print(f"   {status_icon} {p['alias']}: {p['path']}{file_info}")
    
    if dl_status["all_exist"]:
        print(f"   ✅ ALL PATHS ACCESSIBLE ({dl_status['total_files']} total files)")
    else:
        print("   ⚠️ SOME PATHS NOT FOUND (create them to enable watcher)")
    
    # Check manifest
    manifest_path = Path("data/manifests/data_lake_manifest.json")
    if manifest_path.exists():
        print(f"   ✅ Manifest exists: {manifest_path}")
    else:
        print(f"   ⚠️ No manifest yet (run watcher to generate)")
    print()
    
    # Summary
    print("=" * 70)
    if not issues:
        print(" ✅ ALL COMPLIANCE CHECKS PASSED")
    else:
        print(f" ❌ {len(issues)} ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    print("=" * 70)
    
    return 0 if not issues else 1


if __name__ == "__main__":
    exit(main())
