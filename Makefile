# ══════════════════════════════════════════════════════════════════════════════
# BIZRA Makefile - Build, Verify, Seal
# ══════════════════════════════════════════════════════════════════════════════

SHELL := /bin/bash
.PHONY: verify clean seal seal-ots verify-seal verify-all test lint

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

verify:
	python tools/bizra_verify.py --out evidence --artifact-name bizra --artifact-version local

verify-chain:
	@echo "⛓️ Verifying Genesis Chain..."
	python scripts/verify_genesis_chain.py

verify-seal:
	@echo "🔏 Verifying lineage seal pack..."
	@if [ -x scripts/verify_lineage_seal.sh ]; then \
		./scripts/verify_lineage_seal.sh --latest; \
	else \
		bash scripts/verify_lineage_seal.sh --latest; \
	fi

verify-all: verify verify-chain verify-seal
	@echo "✅ All verifications complete"

# ─────────────────────────────────────────────────────────────────────────────
# SEAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

system-manifest:
	@echo "🖥️ Generating host-centric system manifest..."
	python scripts/genesis_system_attestation.py --owner "Momo"

node-zero:
	@echo "🆔 Creating Node0 identity with manifest binding..."
	python core/genesis/node_zero.py --create --owner "Momo"

seal:
	@echo "🔏 Generating Genesis Seal..."
	python scripts/generate_genesis_seal.py --seal --owner "Momo"

seal-ots:
	@echo "🔏 Generating Genesis Seal with OpenTimestamps anchor..."
	python scripts/generate_genesis_seal.py --seal --ots --owner "Momo"

seal-pack:
	@echo "📦 Creating Lineage Seal Pack..."
	python -c "from core.genesis.lineage_seal_pack import LineageSealPack; \
		pack = LineageSealPack(); \
		pack._create_staging(); \
		pack.add_genesis_files(); \
		manifest = pack.seal(); \
		print(f'Pack created: {pack._staging_dir}')"

# ─────────────────────────────────────────────────────────────────────────────
# GENESIS BLOCK PUBLICATION (Complete Sequence)
# ─────────────────────────────────────────────────────────────────────────────

genesis-publish: system-manifest node-zero seal seal-pack verify-chain
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "          🚀 GENESIS BLOCK PUBLICATION COMPLETE"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make seal-ots' to anchor to Bitcoin (requires 'ots' CLI)"
	@echo "  2. Wait 1-6 hours for Bitcoin confirmation"
	@echo "  3. Run 'make verify-all' for complete verification"
	@echo "  4. Commit and tag: git tag -s genesis-v1.0.0"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -q --tb=short

lint:
	ruff check .
	mypy --config-file mypy.ini --no-error-summary

format:
	black .
	isort .

clean:
	rm -rf evidence
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
