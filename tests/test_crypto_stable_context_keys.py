from __future__ import annotations

from core.memory.crypto import CryptoManager

MASTER_KEY = b"\x11" * 32


def _make_manager() -> CryptoManager:
    return CryptoManager(MASTER_KEY)


def test_context_derived_key_is_stable_across_instances() -> None:
    manager1 = _make_manager()
    manager2 = _make_manager()

    key1 = manager1.derive_key("audit_log")
    key2 = manager2.derive_key("audit_log")

    assert key1 == key2
    assert isinstance(key1, (bytes, bytearray))
    assert len(key1) == 32


def test_hmac_is_stable_across_instances() -> None:
    manager1 = _make_manager()
    manager2 = _make_manager()

    message = b"chain-link-001"
    digest1 = manager1.compute_hmac(message, "audit_log")
    digest2 = manager2.compute_hmac(message, "audit_log")

    assert digest1 == digest2
    assert isinstance(digest1, (bytes, bytearray))
    assert len(digest1) == 32


def test_encryption_keeps_per_blob_randomness() -> None:
    manager = _make_manager()
    plaintext = b"same-plaintext"

    blob1 = manager.encrypt(plaintext, context="data")
    blob2 = manager.encrypt(plaintext, context="data")

    assert (blob1.salt != blob2.salt) or (blob1.nonce != blob2.nonce)
    assert manager.decrypt(blob1) == plaintext
    assert manager.decrypt(blob2) == plaintext
