"""Cryptographic helpers (hashing + Ed25519 signing)."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


@dataclass
class Ed25519Keypair:
    private: Ed25519PrivateKey
    public: Ed25519PublicKey

    @staticmethod
    def load_or_generate(path: Path) -> "Ed25519Keypair":
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = path.read_bytes()
            priv = Ed25519PrivateKey.from_private_bytes(raw)
        else:
            priv = Ed25519PrivateKey.generate()
            raw = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            path.write_bytes(raw)
        pub = priv.public_key()
        return Ed25519Keypair(private=priv, public=pub)

    def public_bytes_raw(self) -> bytes:
        return self.public.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(self, msg: bytes) -> bytes:
        return self.private.sign(msg)

    def verify(self, sig: bytes, msg: bytes) -> None:
        self.public.verify(sig, msg)


@dataclass
class SignatureBundle:
    scheme: str
    public_key_b64: str
    signature_b64: str

    @staticmethod
    def sign(keypair: Ed25519Keypair, msg: bytes) -> "SignatureBundle":
        sig = keypair.sign(msg)
        return SignatureBundle(
            scheme="ed25519",
            public_key_b64=b64(keypair.public_bytes_raw()),
            signature_b64=b64(sig),
        )

    def verify(self, msg: bytes) -> None:
        pub = Ed25519PublicKey.from_public_bytes(b64d(self.public_key_b64))
        pub.verify(b64d(self.signature_b64), msg)
