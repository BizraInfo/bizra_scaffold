# BIZRA Unified Scaffold

[![CI/CD Pipeline](https://github.com/BizraInfo/bizra_scaffold/actions/workflows/ci.yml/badge.svg)](https://github.com/BizraInfo/bizra_scaffold/actions/workflows/ci.yml)
[![Security Scan](https://github.com/BizraInfo/bizra_scaffold/actions/workflows/security-secret-scan.yml/badge.svg)](https://github.com/BizraInfo/bizra_scaffold/actions/workflows/security-secret-scan.yml)
[![License](https://img.shields.io/badge/license-See%20LICENSE__NOTE-blue.svg)](LICENSE_NOTE.txt)

**Version:** 0.1  
**Status:** Active Development

## Purpose

BIZRA Unified Scaffold is a **starter kit and reference implementation** for building evidence-based, cognitive AI systems with robust integrity verification. It provides:

- **Unified architecture** for BIZRA components (Proof of Integrity, Dual-Agentic Systems, Attestation)
- **Production-ready security** hardening (JWT auth, secret scanning, vulnerability management)
- **Verification kernel** for build-verify-metrics workflows
- **Multi-language support** (Python 3.10+, Rust stable)
- **Elite practitioner templates** for documentation, CI/CD, and governance

This scaffold helps you unify scattered BIZRA components into a cohesive monorepo with a Single Source of Truth (SOT) and evidence-based claims verification.

## Architecture Overview

### System Components

```
bizra_scaffold/
├── core/                      # Core Python modules
│   ├── engine/               # FastAPI runtime and HTTP API
│   ├── security/             # JWT auth, hardened security utilities
│   ├── verification/         # PoI verification and metrics
│   ├── agents/               # PAT/SAT dual-agentic orchestration
│   ├── layers/               # Memory layers (L2/L3/L4)
│   └── knowledge/            # Knowledge graph and semantic processing
├── crates/                    # Rust implementation crates
│   ├── attestation-engine/   # Genesis Node Zero attestation
│   ├── bizra-pat-sat/        # Rust PAT/SAT implementation
│   └── bizra-verify/         # Verification kernel (Rust)
├── tests/                     # Test suite (pytest + cargo test)
├── tools/                     # Build and verification scripts
├── .github/workflows/         # CI/CD pipelines
└── docs/                      # Documentation
```

### Key Technologies

- **Python 3.10+** - Core runtime (FastAPI, PyTorch, NetworkX)
- **Rust stable** - High-performance attestation and verification
- **Neo4j** - Graph database for knowledge representation
- **FastAPI + Uvicorn** - Async HTTP API with OpenAPI docs
- **Prometheus** - Metrics and observability
- **GitHub Actions** - CI/CD automation

### Architecture Principles

1. **Evidence-Based (Ihsān)** - Every claim must be grounded in verifiable artifacts
2. **Proof of Integrity (PoI)** - Cryptographic attestation for system state
3. **Dual-Agentic** - PAT (Proof Agentic Thinker) and SAT (Semantic Agentic Thinker)
4. **Defense in Depth** - Security hardening at every layer
5. **Signal-to-Noise Ratio (SNR)** - High-impact, minimal-change philosophy

## Contents

| File | Description |
|---|---|
| `report.md` | Evidence-based analysis summarizing the current state of BIZRA repositories, architecture, security, performance, documentation, and recommendations for unification. |
| `SOT_template.md` | A template for your **Single Source of Truth (SOT)** file, where you define canonical names, invariants, PoI parameters, and evidence requirements. Customize this file to reflect your project's agreed rules. |
| `BIZRA_SOT.md` | Draft single source of truth for the unified system, derived from `SOT_template.md`. |
| `EVIDENCE_INDEX.md` | Index of evidence artifacts required to validate claims across documentation and reports. |
| `cognitive_sovereign.py` | Cognitive Sovereign kernel implementation (AEON OMEGA v9.8.0). |
| `Genesis_NodeZero_Attestation_Spec_v1.0.md` | Official specification of the Genesis Node Zero attestation protocol (for reference). |
| `README.md` | This file. |

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Rust stable** (optional, for Rust crates)
- **Git** for version control
- **Neo4j** (optional, for graph database features)

### Local Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/BizraInfo/bizra_scaffold.git
   cd bizra_scaffold
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For GPU support (optional):
   ```bash
   pip install torch==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configure environment**

   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

5. **Run verification kernel**

   ```bash
   python tools/bizra_verify.py --out evidence --artifact-name bizra --artifact-version local
   ```

   Or using Make:
   ```bash
   make verify
   ```

### Running the API Server

```bash
# Set PYTHONPATH
export PYTHONPATH=$PWD

# Run with uvicorn
uvicorn core.engine.api:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at: http://localhost:8000/docs

## How to Use

1. **Create a new GitHub repository** (e.g. `bizra-unified`), and clone it to your machine.

2. **Copy the contents of this `bizra_scaffold` folder** into your new repository.

3. **Set up a monorepo structure** using a workspace manager (Nx or Turborepo). For example:

   ```
   bizra-unified/
     apps/
       api/            # REST or GraphQL API entrypoint (Node.js)
     packages/
       poi-core/       # PoI scoring and verification library (Rust/TypeScript)
       blockgraph/     # Consensus and BlockGraph logic
       agent-os/       # Dual-Agentic orchestration (PAT/SAT)
       ...
     docs/             # Documentation site built with Docusaurus (optional)
     BIZRA_SOT.md      # Your customized Single Source of Truth (copy from the template)
     ...
   ```

4. **Fill out `BIZRA_SOT.md`** using `SOT_template.md` as a starting point. Define canonical terms (token names, layer names), invariants (e.g. "every claim must have evidence"), and PoI parameters (weights, thresholds, decay rates). This file becomes the binding contract for your project.

5. **Migrate code from existing directories** (`bizra-genesis-node`, `BIZRA-Dual-Agentic-system-`, `BIZRA-OS`, etc.) into the appropriate packages within your monorepo. Remove duplicate copies and keep only the latest, verified code. Use version control to track history.

6. **Ensure that every claim or metric in your documentation links back to evidence**, such as benchmark logs, test results, or signed attestation files. Your CI pipeline should reject changes that break the SOT or refer to claims without proof.

7. **Run your development environment locally** using VS Code, GitHub Copilot, and Claude 4.5 Opus. Because everything is now unified, you can run tests (`nx test`), build services (`nx build`), and spin up local dev servers (`nx serve api`) from a single entry point.

By following these steps you'll convert years of fragmented development into a cohesive, future-proof system.

## Build and Test

### Linting and Formatting

```bash
# Format Python code
black .
isort .

# Lint Python code
ruff check .

# Type check
mypy --ignore-missing-imports core/ tests/
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html --cov-report=term

# Run specific test
pytest tests/test_core_modules.py -v
```

### Rust Crates

```bash
cd crates/attestation-engine

# Format
cargo fmt

# Lint
cargo clippy -- -D warnings

# Test
cargo test --verbose

# Build
cargo build --release
```

### Security Scanning

```bash
# Python security scan
bandit -r core/ -ll

# Dependency vulnerability scan
safety check -r requirements.txt

# Secret scanning (requires gitleaks)
gitleaks detect --source . --verbose
```

## Release Process

Releases are managed through semantic versioning:

1. **Update version** in `pyproject.toml` and relevant `Cargo.toml` files
2. **Create release branch**: `git checkout -b release/vX.Y.Z`
3. **Tag the release**: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. **Push tag**: `git push origin vX.Y.Z`
5. **CI/CD** automatically builds and deploys

### Version Strategy

- **Patch (0.1.X)** - Bug fixes, security patches
- **Minor (0.X.0)** - New features, backward compatible
- **Major (X.0.0)** - Breaking changes

## Security

### Reporting Vulnerabilities

Please see [SECURITY.md](SECURITY.md) for our security policy and how to report vulnerabilities responsibly.

### Security Features

- **Secret Scanning** - Automated gitleaks scanning in CI
- **Dependency Scanning** - Bandit and Safety checks
- **JWT Authentication** - Hardened token management with rotation
- **Input Validation** - Pydantic models for API validation
- **Least Privilege** - GitHub Actions with minimal permissions
- **Dependency Pinning** - All versions explicitly pinned

### Security Best Practices

- Never commit secrets (use `.env` files, see `.env.template`)
- Use `requirements-production.txt` for production deployments
- Rotate JWT secrets regularly
- Enable HTTPS/TLS in production
- Review security scan results before merging PRs

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development workflow and branching strategy
- Coding standards (Python PEP 8, Rust guidelines)
- Testing requirements
- Commit message conventions
- Pull request process

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run linters and tests locally
5. Commit with conventional commits: `feat(scope): description`
6. Push and create a pull request
7. Wait for CI checks and code review

## Verification Kernel (BUILD-VERIFY-METRICS)

This scaffold now includes a verification kernel that produces evidence receipts and metrics.

### Quick start (local)

```bash
python tools/bizra_verify.py --out evidence --artifact-name bizra_scaffold --artifact-version local
```

### CI

The workflow in `.github/workflows/verify.yml` runs the kernel and uploads evidence artifacts.

## Documentation

- **[README.md](README.md)** - This file (overview and quick start)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[SECURITY.md](SECURITY.md)** - Security policy and reporting
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Detailed installation instructions
- **[BIZRA_SOT.md](BIZRA_SOT.md)** - Single Source of Truth (canonical definitions)
- **[PROTOCOL.md](PROTOCOL.md)** - Protocol specifications
- **[docs/](docs/)** - Additional documentation

## CI/CD Pipelines

This repository uses GitHub Actions for continuous integration and deployment:

- **[ci.yml](.github/workflows/ci.yml)** - Main CI/CD pipeline (lint, test, build, deploy)
- **[verify.yml](.github/workflows/verify.yml)** - Build-verify-metrics kernel
- **[security-secret-scan.yml](.github/workflows/security-secret-scan.yml)** - Secret and security scanning
- **[evidence-spine.yml](.github/workflows/evidence-spine.yml)** - Evidence collection
- **[verify-metrics-gate.yml](.github/workflows/verify-metrics-gate.yml)** - Quality gates

All workflows follow security best practices:
- Least-privilege permissions
- Dependency caching for performance
- Safe for fork contributions
- No `pull_request_target` vulnerabilities

## Project Status

**Current Phase:** Active Development (v0.1)

### Recent Updates

- ✅ Security hardening (JWT, secret scanning, dependency pinning)
- ✅ CI/CD pipeline with multi-stage deployment
- ✅ Verification kernel and evidence collection
- ✅ Rust attestation engine implementation
- 🚧 Production deployment automation
- 🚧 Comprehensive test coverage expansion
- 📋 SBOM generation and supply chain security

## License

See [LICENSE_NOTE.txt](LICENSE_NOTE.txt) for licensing information.

## Support

- **Issues**: [GitHub Issues](https://github.com/BizraInfo/bizra_scaffold/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BizraInfo/bizra_scaffold/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

## Acknowledgments

Built with the Ihsān principle: evidence-based claims, cryptographic integrity, and elite practitioner standards.

---

**BIZRA AEON OMEGA** - Cognitive Sovereignty through Verifiable Integrity
