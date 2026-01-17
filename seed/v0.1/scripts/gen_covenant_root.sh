#!/usr/bin/env bash
set -euo pipefail

# Compute a simple root hash for the covenant.
# Replace SHA-256 with SHA3-512 if you require it.
sha256sum COVENANT.md | awk '{print $1}' > .covenant_root

printf "Covenant root written to .covenant_root: %s\n" "$(cat .covenant_root)"
