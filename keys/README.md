# ⚠️ KEYS DIRECTORY - SECURITY NOTICE

## 🔴 CRITICAL: NO PRIVATE KEYS ALLOWED

This directory is for **PUBLIC KEYS ONLY** used in cryptographic verification.

### Policy

1. **NEVER** commit private keys, secret keys, or credentials to this directory
2. All files in this directory are assumed to be **PUBLIC** and safe for exposure
3. Private keys must be stored in secure key management systems (HashiCorp Vault, AWS KMS, Azure Key Vault)

### What belongs here

- ✅ Public keys for signature verification
- ✅ Public certificates
- ✅ Example/template keys for development (clearly marked as TEST ONLY)

### What does NOT belong here

- ❌ Private keys (*.key with private data)
- ❌ Secret keys
- ❌ API tokens or credentials
- ❌ Passphrases or passwords
- ❌ Any key material that would compromise security if exposed

### CI Enforcement

This repository includes CI checks that:
1. Scan for common private key patterns (PEM headers, key file signatures)
2. Block commits containing potential secrets
3. Alert on any new files added to this directory

### Reporting

If you discover a private key has been committed, immediately:
1. Rotate the compromised key
2. Contact security team
3. Use `git filter-branch` or BFG Repo-Cleaner to purge from history

---

**Last Audit:** 2025-12-25  
**Status:** ⚠️ PENDING CLEANUP - Remove any non-public keys
