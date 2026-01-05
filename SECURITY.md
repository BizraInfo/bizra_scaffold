# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of `bizra_scaffold` seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report Privately

Instead, please report security vulnerabilities via one of the following methods:

- **GitHub Security Advisories**: Use the [Report a vulnerability](https://github.com/BizraInfo/bizra_scaffold/security/advisories/new) feature (preferred)
- **Email**: Send details to the repository maintainers (check GitHub profile for contact)

### 3. What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., remote code execution, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### 4. Response Timeline

- **Initial Response**: We aim to acknowledge receipt within 48 hours
- **Status Updates**: We'll provide status updates every 5-7 days
- **Resolution**: We aim to release a fix within 90 days for critical vulnerabilities

### 5. Disclosure Policy

- We follow a **coordinated disclosure** approach
- We request that you do not publicly disclose the vulnerability until we've released a fix
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Users

When using `bizra_scaffold`, please follow these security best practices:

### Secrets Management

- **Never commit secrets** to version control (API keys, private keys, credentials)
- Use `.env` files for local development and ensure they're in `.gitignore`
- Use environment variables or secure secret management systems in production
- Reference `.env.template` for required environment variables without sensitive values

### Dependency Security

- Regularly update dependencies to patch known vulnerabilities
- Review the output of `pip install` and `cargo build` for security warnings
- Monitor GitHub Security Advisories and Dependabot alerts

### Authentication & Authorization

- Use strong, unique secrets for JWT signing (see `core/security/jwt_hardened.py`)
- Rotate credentials regularly
- Implement proper RBAC for API access
- Use HTTPS/TLS in production deployments

### Network Security

- Configure CORS appropriately for your use case (see `core/engine/api.py`)
- Use API rate limiting to prevent abuse
- Deploy behind a reverse proxy (nginx, Caddy) with proper security headers
- Enable network policies in Kubernetes deployments

### Data Protection

- Encrypt sensitive data at rest and in transit
- Implement proper input validation and sanitization
- Use defusedxml for XML parsing (already configured)
- Follow the principle of least privilege for database and API access

## Security Features

This repository implements several security features:

### Code Scanning

- **Gitleaks**: Automated secret scanning in CI/CD (`.github/workflows/security-secret-scan.yml`)
- **Bandit**: Python security linter for common vulnerability patterns
- **Safety**: Dependency vulnerability scanner

### Secure Coding Practices

- **JWT Authentication**: Hardened JWT implementation with rotation and revocation
- **Input Validation**: Pydantic models for API input validation
- **XML Security**: defusedxml to prevent XXE attacks
- **Dependency Pinning**: All dependencies pinned to specific versions

### CI/CD Security

- **Least Privilege Permissions**: GitHub Actions workflows use minimal required permissions
- **Dependency Caching**: Secure caching strategies with integrity checks
- **Secret Scanning**: Automated secret detection on every push/PR

## Known Security Considerations

### Development vs Production

- `requirements.txt` uses numpy 2.x for development features
- `requirements-production.txt` uses numpy <2.0 for scipy compatibility
- Ensure you use the correct requirements file for your environment

### Cryptographic Dependencies

This project uses:
- `cryptography==45.0.7` for core cryptographic operations
- `blake3==0.3.3` for fast hashing
- `PyJWT>=2.10.1` for JWT token handling

These are regularly updated to address security vulnerabilities. Monitor for updates.

## Security Audit Trail

Security-related changes and incidents will be documented here:

- **Initial Release**: Security policy established with this repository version
- Future updates will be tracked with dates and descriptions

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Rust Security Guidelines](https://anssi-fr.github.io/rust-guide/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

*Last Updated: January 2026*
