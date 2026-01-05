# Contributing to BIZRA Scaffold

Thank you for your interest in contributing to BIZRA Scaffold! This document provides guidelines and standards for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Security](#security)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect:

- **Respectful communication** in all interactions
- **Constructive feedback** that helps improve the project
- **Collaboration** over competition
- **Evidence-based decision making** following the Ihsān principle (no fabricated claims)

### Enforcement

Violations of the code of conduct should be reported to the project maintainers. All reports will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Rust stable** (for Rust crates)
- **Git** for version control
- **Make** for build automation (optional)

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

4. **Install development tools**

   ```bash
   pip install ruff mypy black isort pytest pytest-cov pytest-asyncio hypothesis
   ```

5. **Set up environment variables**

   ```bash
   cp .env.template .env
   # Edit .env with your local configuration
   ```

6. **Verify installation**

   ```bash
   python tools/bizra_verify.py --out evidence --artifact-name bizra --artifact-version local
   ```

### Project Structure

```
bizra_scaffold/
├── core/                 # Core Python modules
│   ├── engine/          # API and runtime engine
│   ├── security/        # Security utilities (JWT, auth)
│   ├── verification/    # Verification and validation
│   └── ...
├── crates/              # Rust crates
│   ├── attestation-engine/
│   ├── bizra-pat-sat/
│   └── bizra-verify/
├── tests/               # Test suite
├── tools/               # Build and verification tools
├── .github/workflows/   # CI/CD pipelines
└── docs/                # Documentation
```

## Development Workflow

### Branching Strategy

We use a simplified Git Flow:

- **`main`** - Production-ready code
- **`develop`** - Integration branch for features
- **`feature/*`** - New features (branch from `develop`)
- **`fix/*`** - Bug fixes (branch from `develop` or `main`)
- **`hotfix/*`** - Critical production fixes (branch from `main`)

### Creating a Branch

```bash
# For a new feature
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# For a bug fix
git checkout -b fix/issue-number-description

# For a hotfix
git checkout main
git checkout -b hotfix/critical-fix-name
```

## Coding Standards

### Python

We follow **PEP 8** with some project-specific conventions:

#### Style Guide

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings (Black default)
- **Import order**: isort with Black profile
- **Type hints**: Use type hints for all public functions

#### Linting and Formatting

Run before committing:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with Ruff
ruff check .

# Type check with MyPy
mypy --ignore-missing-imports core/ tests/
```

#### Docstrings

Use **Google-style docstrings**:

```python
def calculate_poi(claim: str, evidence: List[str]) -> float:
    """Calculate Proof of Integrity score for a claim.
    
    Args:
        claim: The claim statement to verify
        evidence: List of evidence artifacts supporting the claim
        
    Returns:
        PoI score between 0.0 and 1.0
        
    Raises:
        ValueError: If claim or evidence is empty
        
    Example:
        >>> score = calculate_poi("System is secure", ["audit.log"])
        >>> print(f"PoI: {score:.2f}")
    """
```

### Rust

#### Style Guide

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Run `cargo fmt` before committing
- Address all `cargo clippy` warnings

#### Rust Commands

```bash
cd crates/attestation-engine

# Format code
cargo fmt

# Lint
cargo clippy -- -D warnings

# Run tests
cargo test
```

### Security Best Practices

- **Never commit secrets** - Use `.env` files and environment variables
- **Input validation** - Always validate and sanitize user input
- **Dependencies** - Pin all dependency versions
- **Security scanning** - Run `bandit` and `safety` locally before pushing

## Testing Guidelines

### Writing Tests

- **Test organization**: Mirror the structure of the code being tested
- **Test naming**: Use descriptive names that explain what is being tested
- **Coverage**: Aim for >80% code coverage for new code
- **Test types**: Include unit tests, integration tests, and property-based tests

### Running Tests

```bash
# Run all Python tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_core_modules.py -v

# Run tests matching a pattern
pytest tests/ -k "test_jwt" -v
```

### Test Requirements

- All new features **must** include tests
- Bug fixes **should** include regression tests
- Tests **must** pass before merging
- No decrease in overall code coverage

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic changes)
- **refactor**: Code refactoring (no feature changes or bug fixes)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, build, etc.)
- **security**: Security improvements or fixes

### Examples

```bash
# Feature
git commit -m "feat(api): add JWT token refresh endpoint"

# Bug fix
git commit -m "fix(verification): handle empty evidence list gracefully"

# Security fix
git commit -m "security(deps): upgrade cryptography to 45.0.7"

# Documentation
git commit -m "docs(readme): add local development setup instructions"
```

### Commit Best Practices

- **Atomic commits**: One logical change per commit
- **Clear subject**: Imperative mood, present tense ("add" not "added")
- **Descriptive body**: Explain *why* the change was made, not just *what*
- **Reference issues**: Include "Fixes #123" or "Closes #456" in footer

## Pull Request Process

### Before Creating a PR

1. **Update your branch** with the latest changes from the base branch

   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-branch
   git rebase develop
   ```

2. **Run all checks locally**

   ```bash
   # Linting and formatting
   black . && isort . && ruff check .
   
   # Type checking
   mypy --ignore-missing-imports core/ tests/
   
   # Tests
   pytest tests/ -v
   
   # Security scan
   bandit -r core/ -ll
   ```

3. **Verify the build**

   ```bash
   python tools/bizra_verify.py --out evidence --artifact-name bizra --artifact-version local
   ```

### Creating a PR

1. **Push your branch** to GitHub

   ```bash
   git push origin your-branch
   ```

2. **Create the pull request** on GitHub

3. **Fill out the PR template** completely (see `.github/PULL_REQUEST_TEMPLATE.md`)

4. **Link related issues** using keywords like "Fixes #123"

### PR Checklist

- [ ] Code follows the project's coding standards
- [ ] All tests pass locally
- [ ] New tests added for new features/bug fixes
- [ ] Documentation updated (README, docstrings, etc.)
- [ ] Commit messages follow the conventional commits format
- [ ] No secrets or sensitive data in commits
- [ ] Security scan passes (no new vulnerabilities)
- [ ] PR description clearly explains the changes

### Code Review Process

1. **Automated checks**: CI must pass (linting, tests, security scans)
2. **Peer review**: At least one approving review from a maintainer
3. **Address feedback**: Respond to all review comments
4. **Keep PR updated**: Rebase or merge latest changes from base branch

### Merging

- **Squash and merge**: For feature branches (clean history)
- **Merge commit**: For release branches (preserve history)
- **Delete branch**: After merging (keep repository clean)

## Security

### Reporting Vulnerabilities

Please see [SECURITY.md](SECURITY.md) for our security policy and how to report vulnerabilities.

### Security in Development

- Review [SECURITY.md](SECURITY.md) for security best practices
- Never commit credentials, API keys, or private keys
- Use `defusedxml` for XML parsing (already configured)
- Validate all user inputs
- Follow the principle of least privilege

## Build and Release Process

### Local Build

```bash
# Verify build
make verify

# Clean artifacts
make clean
```

### Release Process

Releases are managed by project maintainers:

1. Version bump in relevant files (`pyproject.toml`, `Cargo.toml`)
2. Update `CHANGELOG.md` (if it exists)
3. Create a release branch (`release/vX.Y.Z`)
4. Tag the release (`git tag vX.Y.Z`)
5. Push tag to trigger release workflow

## Getting Help

- **Documentation**: Check the `docs/` directory and README.md
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions and ideas

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see LICENSE_NOTE.txt).

---

Thank you for contributing to BIZRA Scaffold! Your efforts help make this project better for everyone.
