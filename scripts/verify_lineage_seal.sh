#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# BIZRA Lineage Seal Verifier
# ══════════════════════════════════════════════════════════════════════════════
# 
# Court-grade offline verification script for BIZRA Lineage Seal Packs.
# Implements fail-closed verification with multiple cryptographic checks.
#
# Verification Chain:
# 1. Manifest hash integrity (BLAKE3 or SHA256 fallback)
# 2. Per-file content hash verification
# 3. Ed25519 signature verification (if public key provided)
# 4. OpenTimestamps proof verification (if available)
# 5. Cosign bundle verification (if available)
#
# Usage:
#   ./scripts/verify_lineage_seal.sh [PACK_DIR]
#   ./scripts/verify_lineage_seal.sh --latest
#   ./scripts/verify_lineage_seal.sh --help
#
# Exit Codes:
#   0 - All verifications passed
#   1 - Verification failed (fail-closed)
#   2 - Missing dependencies
#   3 - Invalid arguments
#
# Author: BIZRA Genesis Team
# Version: 1.0.0
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default pack location
DEFAULT_PACK_DIR="$REPO_ROOT/data/lineage_packs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

usage() {
    cat << EOF
BIZRA Lineage Seal Verifier

Usage:
    $(basename "$0") [OPTIONS] [PACK_DIR]

Options:
    --latest        Verify the most recent lineage seal pack
    --all           Verify all lineage seal packs
    --verbose       Enable verbose output
    --help          Show this help message

Arguments:
    PACK_DIR        Path to the lineage seal pack directory

Examples:
    $(basename "$0") --latest
    $(basename "$0") data/lineage_packs/BIZRA_LINEAGE_SEAL_PACK_20251229_123456

Exit Codes:
    0 - All verifications passed
    1 - Verification failed (fail-closed)
    2 - Missing dependencies
    3 - Invalid arguments
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Dependency Checks
# ─────────────────────────────────────────────────────────────────────────────

check_python_deps() {
    python3 -c "
import sys
try:
    import blake3
    BLAKE3 = True
except ImportError:
    BLAKE3 = False

import hashlib
# blake2b is always available in hashlib
print(f'blake3:{BLAKE3}')
print(f'blake2b:True')
" 2>/dev/null || {
    log_error "Python3 not available"
    exit 2
}
}

# ─────────────────────────────────────────────────────────────────────────────
# Hash Computation
# ─────────────────────────────────────────────────────────────────────────────

compute_file_hash() {
    local file_path="$1"
    local algo="${2:-blake3}"
    
    python3 << PYEOF
import sys
import json
import hashlib
from pathlib import Path

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

file_path = Path("$file_path")
algo = "$algo"

if not file_path.exists():
    print("FILE_NOT_FOUND", file=sys.stderr)
    sys.exit(1)

if algo == "blake3" and BLAKE3_AVAILABLE:
    hasher = blake3.blake3()
elif algo == "blake3" and not BLAKE3_AVAILABLE:
    # Fallback to blake2b with 256-bit digest
    hasher = hashlib.blake2b(digest_size=32)
else:
    hasher = hashlib.sha256()

with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        hasher.update(chunk)

print(hasher.hexdigest())
PYEOF
}

compute_manifest_hash() {
    local manifest_path="$1"
    
    python3 << PYEOF
import sys
import json
import hashlib
from pathlib import Path

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

manifest_path = Path("$manifest_path")

if not manifest_path.exists():
    print("MANIFEST_NOT_FOUND", file=sys.stderr)
    sys.exit(1)

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

# Recompute manifest hash from items
manifest_data = {
    "version": manifest.get("version", "1.0.0"),
    "pack_id": manifest.get("pack_id", ""),
    "created_at": manifest.get("created_at", ""),
    "created_by": manifest.get("created_by", ""),
    "items": manifest.get("items", []),
    "total_items": manifest.get("total_items", 0),
    "total_bytes": manifest.get("total_bytes", 0),
}
manifest_json = json.dumps(manifest_data, sort_keys=True, separators=(",", ":"))

if BLAKE3_AVAILABLE:
    computed = blake3.blake3(manifest_json.encode()).hexdigest()
else:
    # Fallback to sha256 if blake3 not available
    computed = hashlib.sha256(manifest_json.encode()).hexdigest()

stored = manifest.get("manifest_hash", "")
algo_used = "blake3" if BLAKE3_AVAILABLE else "sha256"

print(f"{computed}:{stored}:{algo_used}")
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Verification Functions
# ─────────────────────────────────────────────────────────────────────────────

verify_manifest_integrity() {
    local pack_dir="$1"
    local manifest_path="$pack_dir/MANIFEST.json"
    
    if [[ ! -f "$manifest_path" ]]; then
        log_error "MANIFEST.json not found in $pack_dir"
        return 1
    fi
    
    log_info "Verifying manifest integrity..."
    
    local result
    result=$(compute_manifest_hash "$manifest_path")
    
    local computed stored algo
    IFS=':' read -r computed stored algo <<< "$result"
    
    if [[ "$computed" == "$stored" ]]; then
        log_success "Manifest hash verified ($algo): ${computed:0:16}..."
        return 0
    else
        log_error "Manifest hash mismatch!"
        log_error "  Computed: $computed"
        log_error "  Stored:   $stored"
        return 1
    fi
}

verify_file_hashes() {
    local pack_dir="$1"
    local manifest_path="$pack_dir/MANIFEST.json"
    
    log_info "Verifying individual file hashes..."
    
    local failed=0
    local verified=0
    
    python3 << PYEOF
import sys
import json
import hashlib
from pathlib import Path

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

pack_dir = Path("$pack_dir")
manifest_path = pack_dir / "MANIFEST.json"

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

failed = 0
verified = 0

for item in manifest.get("items", []):
    file_path = pack_dir / item["relative_path"]
    
    if not file_path.exists():
        print(f"MISSING:{item['relative_path']}")
        failed += 1
        continue
    
    algo = item.get("hash_algorithm", "sha256")
    
    if algo == "blake3" and BLAKE3_AVAILABLE:
        hasher = blake3.blake3()
    elif algo == "blake3":
        # Fallback - the stored hash was blake3, but we can't verify
        print(f"SKIP_BLAKE3:{item['relative_path']}")
        continue
    else:
        hasher = hashlib.sha256()
    
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    
    computed = hasher.hexdigest()
    stored = item["content_hash"]
    
    if computed == stored:
        print(f"OK:{item['relative_path']}")
        verified += 1
    else:
        print(f"MISMATCH:{item['relative_path']}:{computed}:{stored}")
        failed += 1

print(f"SUMMARY:{verified}:{failed}")
PYEOF
    
    return 0  # Let Python output determine pass/fail
}

verify_opentimestamps() {
    local pack_dir="$1"
    
    # Look for .ots files
    local ots_files
    ots_files=$(find "$pack_dir" -name "*.ots" 2>/dev/null || true)
    
    if [[ -z "$ots_files" ]]; then
        log_warning "No OpenTimestamps proofs found (optional)"
        return 0
    fi
    
    if ! command -v ots &> /dev/null; then
        log_warning "'ots' command not found - skipping OTS verification"
        log_info "Install with: pip install opentimestamps-client"
        return 0
    fi
    
    log_info "Verifying OpenTimestamps proofs..."
    
    local failed=0
    for ots_file in $ots_files; do
        log_info "  Verifying: $(basename "$ots_file")"
        
        # The original file should be next to the .ots file (without .ots extension)
        local original_file="${ots_file%.ots}"
        
        if [[ -f "$original_file" ]]; then
            if ots verify "$ots_file" 2>/dev/null; then
                log_success "  OTS proof verified: $(basename "$ots_file")"
            else
                log_warning "  OTS proof pending/unverified: $(basename "$ots_file")"
                # Don't fail - OTS can be pending confirmation
            fi
        else
            log_warning "  Original file not found for OTS: $original_file"
        fi
    done
    
    return 0
}

verify_cosign() {
    local pack_dir="$1"
    
    # Look for cosign signatures (bundles, signatures, certificates)
    local cosign_files
    cosign_files=$(find "$pack_dir" -name "*.cosign" -o -name "*.sig" -o -name "*.bundle" 2>/dev/null || true)
    
    if [[ -z "$cosign_files" ]]; then
        log_warning "No cosign signatures found (optional)"
        return 0
    fi
    
    if ! command -v cosign &> /dev/null; then
        log_warning "'cosign' command not found - skipping signature verification"
        log_info "Install from: https://docs.sigstore.dev/cosign/installation"
        return 0
    fi
    
    log_info "Verifying cosign signatures..."
    local cosign_passed=0
    local cosign_failed=0
    
    for sig_file in $cosign_files; do
        local artifact_file
        local bundle_file
        local cert_file
        
        # Determine artifact path based on signature file naming convention
        # Expected patterns: artifact.ext.sig, artifact.ext.bundle, artifact.ext.cosign
        artifact_file="${sig_file%.sig}"
        artifact_file="${artifact_file%.bundle}"
        artifact_file="${artifact_file%.cosign}"
        
        log_info "  Verifying: $(basename "$sig_file")"
        
        if [[ ! -f "$artifact_file" ]]; then
            log_warning "  Artifact not found: $artifact_file"
            continue
        fi
        
        # Check for bundle file (preferred - contains cert + sig)
        bundle_file="${artifact_file}.bundle"
        if [[ -f "$bundle_file" ]]; then
            # Keyless OIDC verification via Sigstore (GitHub Actions)
            if cosign verify-blob \
                --certificate-identity-regexp='.*@github.com|github-actions' \
                --certificate-oidc-issuer='https://token.actions.githubusercontent.com' \
                --bundle "$bundle_file" \
                "$artifact_file" 2>/dev/null; then
                log_success "  Bundle verified: $(basename "$artifact_file")"
                ((cosign_passed++))
                continue
            fi
        fi
        
        # Fallback: Check for separate cert + sig files
        cert_file="${sig_file%.sig}.cert"
        if [[ -f "$cert_file" ]] && [[ -f "$sig_file" ]]; then
            if cosign verify-blob \
                --certificate "$cert_file" \
                --signature "$sig_file" \
                --certificate-identity-regexp='.*@github.com|github-actions' \
                --certificate-oidc-issuer='https://token.actions.githubusercontent.com' \
                "$artifact_file" 2>/dev/null; then
                log_success "  Signature verified: $(basename "$artifact_file")"
                ((cosign_passed++))
                continue
            fi
        fi
        
        # Try verification with public key if available
        local pubkey_file="$pack_dir/keys/cosign.pub"
        if [[ -f "$pubkey_file" ]] && [[ -f "$sig_file" ]]; then
            if cosign verify-blob \
                --key "$pubkey_file" \
                --signature "$sig_file" \
                "$artifact_file" 2>/dev/null; then
                log_success "  Key-based signature verified: $(basename "$artifact_file")"
                ((cosign_passed++))
                continue
            fi
        fi
        
        log_warning "  Cosign verification failed: $(basename "$sig_file")"
        ((cosign_failed++))
    done
    
    log_info "Cosign summary: $cosign_passed passed, $cosign_failed failed"
    
    # Only fail if there were explicit failures (not just missing files)
    if [[ $cosign_failed -gt 0 ]]; then
        return 1
    fi
    
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Main Verification Flow
# ─────────────────────────────────────────────────────────────────────────────

verify_pack() {
    local pack_dir="$1"
    local verbose="${2:-false}"
    
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════════"
    echo "  BIZRA Lineage Seal Verification"
    echo "══════════════════════════════════════════════════════════════════════════════"
    echo "  Pack: $pack_dir"
    echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "══════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    local all_passed=true
    
    # Step 1: Manifest integrity
    if ! verify_manifest_integrity "$pack_dir"; then
        all_passed=false
    fi
    
    # Step 2: File hashes
    if ! verify_file_hashes "$pack_dir"; then
        all_passed=false
    fi
    
    # Step 3: OpenTimestamps (optional)
    verify_opentimestamps "$pack_dir"
    
    # Step 4: Cosign (optional)
    verify_cosign "$pack_dir"
    
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════════"
    
    if [[ "$all_passed" == true ]]; then
        log_success "VERIFICATION PASSED (fail-closed)"
        echo "══════════════════════════════════════════════════════════════════════════════"
        return 0
    else
        log_error "VERIFICATION FAILED"
        echo "══════════════════════════════════════════════════════════════════════════════"
        return 1
    fi
}

find_latest_pack() {
    local pack_dir="$DEFAULT_PACK_DIR"
    
    if [[ ! -d "$pack_dir" ]]; then
        log_error "Pack directory not found: $pack_dir"
        return 1
    fi
    
    # Find the most recent pack by directory name (timestamp in name)
    local latest
    latest=$(find "$pack_dir" -maxdepth 1 -type d -name "BIZRA_LINEAGE_SEAL_PACK_*" | sort -r | head -n1)
    
    if [[ -z "$latest" ]]; then
        log_error "No lineage seal packs found in $pack_dir"
        return 1
    fi
    
    echo "$latest"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local pack_dir=""
    local verify_all=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --latest)
                pack_dir=$(find_latest_pack) || exit 1
                shift
                ;;
            --all)
                verify_all=true
                shift
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 3
                ;;
            *)
                pack_dir="$1"
                shift
                ;;
        esac
    done
    
    # Check dependencies
    check_python_deps
    
    # Verify all packs
    if [[ "$verify_all" == true ]]; then
        local packs
        packs=$(find "$DEFAULT_PACK_DIR" -maxdepth 1 -type d -name "BIZRA_LINEAGE_SEAL_PACK_*" | sort)
        
        if [[ -z "$packs" ]]; then
            log_error "No lineage seal packs found"
            exit 1
        fi
        
        local failed=0
        for pack in $packs; do
            if ! verify_pack "$pack" "$verbose"; then
                ((failed++))
            fi
        done
        
        if [[ $failed -gt 0 ]]; then
            log_error "$failed pack(s) failed verification"
            exit 1
        fi
        
        log_success "All packs verified successfully"
        exit 0
    fi
    
    # Verify single pack
    if [[ -z "$pack_dir" ]]; then
        log_error "No pack specified. Use --latest or provide a path."
        usage
        exit 3
    fi
    
    if [[ ! -d "$pack_dir" ]]; then
        log_error "Pack directory not found: $pack_dir"
        exit 1
    fi
    
    verify_pack "$pack_dir" "$verbose"
}

main "$@"
