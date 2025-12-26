#!/usr/bin/env python3
"""
IHSĀN BRIDGE - Cross-Language Vocabulary Unification
=====================================================
Elite Practitioner Grade | SOT Compliant | SAPE Framework Aligned

This module provides a canonical bridge between:
- Python (Arabic terminology): IKHLAS, KARAMA, ADL, KAMAL, ISTIDAMA
- Rust (English terminology): truthfulness, dignity, fairness, excellence, sustainability

REFERENCE: BIZRA_SOT.md Section 3.1 - Ihsan Metric Definition (Canonical)
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class IhsanDimension(Enum):
    """
    Canonical Ihsān dimensions with bilingual support.
    Each dimension has Arabic, English, and Rust field mappings.
    """

    TRUTHFULNESS = ("IKHLAS", "truthfulness", 0.30)
    DIGNITY = ("KARAMA", "dignity", 0.20)
    FAIRNESS = ("ADL", "fairness", 0.20)
    EXCELLENCE = ("KAMAL", "excellence", 0.20)
    SUSTAINABILITY = ("ISTIDAMA", "sustainability", 0.10)

    def __init__(self, arabic: str, english: str, weight: float):
        self.arabic = arabic
        self.english = english
        self.weight = weight

    @classmethod
    def from_arabic(cls, name: str) -> Optional["IhsanDimension"]:
        """Look up dimension by Arabic name."""
        for dim in cls:
            if dim.arabic.upper() == name.upper():
                return dim
        return None

    @classmethod
    def from_english(cls, name: str) -> Optional["IhsanDimension"]:
        """Look up dimension by English name."""
        for dim in cls:
            if dim.english.lower() == name.lower():
                return dim
        return None


@dataclass
class IhsanScore:
    """
    Unified Ihsān score with bilingual accessors.
    Compatible with both Python and Rust attestation-engine.
    """

    truthfulness: float = 0.0  # IKHLAS
    dignity: float = 0.0  # KARAMA
    fairness: float = 0.0  # ADL
    excellence: float = 0.0  # KAMAL
    sustainability: float = 0.0  # ISTIDAMA

    # Canonical threshold from BIZRA_SOT.md
    THRESHOLD: float = 0.95

    # ═══════════════════════════════════════════════════════════════════════
    # ARABIC PROPERTY ALIASES (for Python-native code)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def ikhlas(self) -> float:
        """Arabic accessor for Truthfulness."""
        return self.truthfulness

    @ikhlas.setter
    def ikhlas(self, value: float):
        self.truthfulness = value

    @property
    def karama(self) -> float:
        """Arabic accessor for Dignity."""
        return self.dignity

    @karama.setter
    def karama(self, value: float):
        self.dignity = value

    @property
    def adl(self) -> float:
        """Arabic accessor for Fairness."""
        return self.fairness

    @adl.setter
    def adl(self, value: float):
        self.fairness = value

    @property
    def kamal(self) -> float:
        """Arabic accessor for Excellence."""
        return self.excellence

    @kamal.setter
    def kamal(self, value: float):
        self.excellence = value

    @property
    def istidama(self) -> float:
        """Arabic accessor for Sustainability."""
        return self.sustainability

    @istidama.setter
    def istidama(self, value: float):
        self.sustainability = value

    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTATION METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def total(self) -> float:
        """
        Compute weighted total score.
        Matches Rust IhsanScore::total() implementation.
        """
        return (
            self.truthfulness * IhsanDimension.TRUTHFULNESS.weight
            + self.dignity * IhsanDimension.DIGNITY.weight
            + self.fairness * IhsanDimension.FAIRNESS.weight
            + self.excellence * IhsanDimension.EXCELLENCE.weight
            + self.sustainability * IhsanDimension.SUSTAINABILITY.weight
        )

    def verify(self) -> Tuple[bool, float]:
        """
        Verify Ihsān compliance with fail-closed semantics.
        Returns (passed, score).

        CRITICAL: This must match Rust verify_ihsan() behavior exactly.
        """
        # Fail-closed: reject non-finite or out-of-range values
        for dim in IhsanDimension:
            val = getattr(self, dim.english)
            if not isinstance(val, (int, float)):
                return (False, 0.0)
            if val != val:  # NaN check
                return (False, 0.0)
            if val < 0.0 or val > 1.0:
                return (False, 0.0)

        score = self.total()
        return (score >= self.THRESHOLD, score)

    def is_valid(self) -> bool:
        """Check if all dimensions are within valid range."""
        passed, _ = self.verify()
        return passed

    # ═══════════════════════════════════════════════════════════════════════
    # SERIALIZATION (Rust-compatible)
    # ═══════════════════════════════════════════════════════════════════════

    def to_rust_dict(self) -> Dict[str, float]:
        """
        Serialize to Rust-compatible dictionary.
        Uses English field names to match attestation-engine.
        """
        return {
            "truthfulness": self.truthfulness,
            "dignity": self.dignity,
            "fairness": self.fairness,
            "excellence": self.excellence,
            "sustainability": self.sustainability,
        }

    def to_arabic_dict(self) -> Dict[str, float]:
        """
        Serialize with Arabic field names.
        For Python-native documentation and logging.
        """
        return {
            "ikhlas": self.truthfulness,
            "karama": self.dignity,
            "adl": self.fairness,
            "kamal": self.excellence,
            "istidama": self.sustainability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "IhsanScore":
        """
        Deserialize from dictionary (accepts both Arabic and English keys).
        """
        score = cls()

        for key, value in data.items():
            # Try English first
            dim = IhsanDimension.from_english(key)
            if dim is None:
                # Try Arabic
                dim = IhsanDimension.from_arabic(key)

            if dim is not None:
                setattr(score, dim.english, value)

        return score

    def to_json(self, use_arabic: bool = False) -> str:
        """Serialize to JSON string."""
        data = self.to_arabic_dict() if use_arabic else self.to_rust_dict()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "IhsanScore":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class DimensionScores:
    """
    PoI Dimension scores for evidence bundles.
    Compatible with Rust DimensionScores struct.
    """

    quality: float = 0.0
    utility: float = 0.0
    trust: float = 0.0
    fairness: float = 0.0
    diversity: float = 0.0

    # Weights from BIZRA_SOT.md Section 4
    WEIGHTS = {
        "quality": 0.30,
        "utility": 0.30,
        "trust": 0.20,
        "fairness": 0.10,
        "diversity": 0.10,
    }

    def calculate_poi(self, penalty: float = 0.0) -> float:
        """
        Calculate PoI score.
        Matches Rust calculate_poi() implementation.
        """
        max_penalty = 0.15

        raw_poi = (
            self.quality * self.WEIGHTS["quality"]
            + self.utility * self.WEIGHTS["utility"]
            + self.trust * self.WEIGHTS["trust"]
            + self.fairness * self.WEIGHTS["fairness"]
            + self.diversity * self.WEIGHTS["diversity"]
        )

        clamped_penalty = min(penalty, max_penalty)
        final_poi = raw_poi * (1.0 - clamped_penalty)

        return max(0.0, final_poi)

    def validate(self) -> bool:
        """Validate all dimensions are finite and in [0, 1]."""
        for field in ["quality", "utility", "trust", "fairness", "diversity"]:
            val = getattr(self, field)
            if not isinstance(val, (int, float)):
                return False
            if val != val:  # NaN
                return False
            if val < 0.0 or val > 1.0:
                return False
        return True

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "quality": self.quality,
            "utility": self.utility,
            "trust": self.trust,
            "fairness": self.fairness,
            "diversity": self.diversity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# VOCABULARY TRANSLATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


class IhsanVocabulary:
    """
    Static translation utilities for Ihsān terminology.

    MAPPING TABLE (from BIZRA_SOT.md):
    ═══════════════════════════════════════════════════════════════════════════
    Arabic (روحي)  │ English (Technical) │ Rust Field      │ Weight
    ───────────────────────────────────────────────────────────────────────────
    إخلاص (IKHLAS) │ Truthfulness        │ truthfulness    │ 0.30
    كرامة (KARAMA) │ Dignity             │ dignity         │ 0.20
    عدل (ADL)      │ Fairness            │ fairness        │ 0.20
    كمال (KAMAL)   │ Excellence          │ excellence      │ 0.20
    استدامة (ISTIDAMA) │ Sustainability  │ sustainability  │ 0.10
    ═══════════════════════════════════════════════════════════════════════════
    """

    ARABIC_TO_ENGLISH = {
        "ikhlas": "truthfulness",
        "karama": "dignity",
        "adl": "fairness",
        "kamal": "excellence",
        "itqan": "excellence",  # Alternative Arabic term
        "istidama": "sustainability",
        # Legacy mappings
        "taqwa": "fairness",  # Mindfulness → Fairness
        "rahma": "dignity",  # Compassion → Dignity
    }

    ENGLISH_TO_ARABIC = {
        "truthfulness": "ikhlas",
        "dignity": "karama",
        "fairness": "adl",
        "excellence": "kamal",
        "sustainability": "istidama",
    }

    @classmethod
    def to_english(cls, arabic_term: str) -> Optional[str]:
        """Translate Arabic term to English."""
        return cls.ARABIC_TO_ENGLISH.get(arabic_term.lower())

    @classmethod
    def to_arabic(cls, english_term: str) -> Optional[str]:
        """Translate English term to Arabic."""
        return cls.ENGLISH_TO_ARABIC.get(english_term.lower())

    @classmethod
    def normalize_dict(
        cls, data: Dict[str, Any], target: str = "english"
    ) -> Dict[str, Any]:
        """
        Normalize dictionary keys to target vocabulary (english or arabic).
        """
        result = {}
        mapping = (
            cls.ARABIC_TO_ENGLISH if target == "english" else cls.ENGLISH_TO_ARABIC
        )

        for key, value in data.items():
            normalized_key = mapping.get(key.lower(), key)
            result[normalized_key] = value

        return result


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-LANGUAGE ATTESTATION BRIDGE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AttestationBridge:
    """
    Bridge for creating Python attestations compatible with Rust verification.
    Ensures deterministic serialization matching attestation-engine.
    """

    @staticmethod
    def create_evidence_bundle(
        content_hash: str,
        dimensions: DimensionScores,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create evidence bundle in Rust-compatible format.
        """
        return {
            "content_hash": content_hash,
            "metadata": metadata or {},
            "dimensions": dimensions.to_dict(),
        }

    @staticmethod
    def canonical_json(data: Dict[str, Any]) -> bytes:
        """
        Produce canonical JSON matching Rust serde_jcs behavior.
        RFC 8785 (JSON Canonicalization Scheme):
        - Keys sorted lexicographically
        - No whitespace
        - UTF-8 encoded
        - Numbers normalized (no trailing zeros, no leading zeros except for 0.x)
        """

        def jcs_serialize(obj: Any) -> str:
            """Serialize object to JCS-compliant JSON."""
            if obj is None:
                return "null"
            elif isinstance(obj, bool):
                return "true" if obj else "false"
            elif isinstance(obj, int):
                return str(obj)
            elif isinstance(obj, float):
                # JCS: Use shortest decimal representation
                if obj == int(obj):
                    return str(int(obj))
                return repr(obj)
            elif isinstance(obj, str):
                # JCS: Escape only required characters
                escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
                escaped = (
                    escaped.replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                return f'"{escaped}"'
            elif isinstance(obj, (list, tuple)):
                items = ",".join(jcs_serialize(item) for item in obj)
                return f"[{items}]"
            elif isinstance(obj, dict):
                # JCS: Sort keys lexicographically by UTF-16 code units
                sorted_items = sorted(obj.items(), key=lambda x: x[0])
                pairs = ",".join(f'"{k}":{jcs_serialize(v)}' for k, v in sorted_items)
                return "{" + pairs + "}"
            else:
                return jcs_serialize(str(obj))

        return jcs_serialize(data).encode("utf-8")

    @staticmethod
    def compute_evidence_root(evidence: Dict[str, Any]) -> str:
        """
        Compute evidence root using Blake3, matching Rust crypto.rs.
        """
        import blake3

        canonical = AttestationBridge.canonical_json(evidence)
        return blake3.blake3(canonical).hexdigest()

    @staticmethod
    def compute_attestation_id(contributor: str, epoch: int, evidence_root: str) -> str:
        """
        Compute attestation ID matching Rust implementation exactly.
        Uses Blake3 hash of contributor + epoch (big-endian u64) + evidence_root.

        Must match Rust crypto.rs::compute_attestation_id()
        """
        import blake3

        hasher = blake3.blake3()
        hasher.update(contributor.encode("utf-8"))
        hasher.update(epoch.to_bytes(8, "big"))  # u64 big-endian
        hasher.update(evidence_root.encode("utf-8"))
        return hasher.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════


def verify_vocabulary_consistency() -> bool:
    """
    Verify that vocabulary mappings are consistent and complete.
    This should pass as a CI check.
    """
    # Check bidirectional mapping consistency
    for arabic, english in IhsanVocabulary.ARABIC_TO_ENGLISH.items():
        if arabic not in ["taqwa", "rahma", "itqan"]:  # Exclude legacy/alias
            reverse = IhsanVocabulary.ENGLISH_TO_ARABIC.get(english)
            if reverse is None:
                return False

    # Check all dimensions are covered
    for dim in IhsanDimension:
        if dim.arabic.lower() not in IhsanVocabulary.ARABIC_TO_ENGLISH:
            return False
        if dim.english.lower() not in IhsanVocabulary.ENGLISH_TO_ARABIC:
            return False

    # Check weights sum to 1.0
    total_weight = sum(dim.weight for dim in IhsanDimension)
    if abs(total_weight - 1.0) > 1e-9:
        return False

    return True


if __name__ == "__main__":
    # Self-test
    print("Ihsān Bridge Self-Test")
    print("=" * 50)

    # Test 1: Vocabulary consistency
    assert verify_vocabulary_consistency(), "Vocabulary consistency check failed"
    print("✓ Vocabulary consistency verified")

    # Test 2: Score computation
    score = IhsanScore(
        truthfulness=1.0,
        dignity=1.0,
        fairness=1.0,
        excellence=1.0,
        sustainability=1.0,
    )
    assert abs(score.total() - 1.0) < 1e-9, "Perfect score should equal 1.0"
    print("✓ Score computation verified")

    # Test 3: Verification
    passed, total = score.verify()
    assert passed, "Perfect score should pass verification"
    print("✓ Verification logic verified")

    # Test 4: Arabic accessors
    score.ikhlas = 0.5
    assert score.truthfulness == 0.5, "Arabic accessor should update English field"
    print("✓ Bilingual accessors verified")

    # Test 5: Serialization roundtrip
    original = IhsanScore(0.9, 0.85, 0.88, 0.92, 0.95)
    json_str = original.to_json()
    restored = IhsanScore.from_json(json_str)
    assert original.total() == restored.total(), "Serialization roundtrip failed"
    print("✓ Serialization roundtrip verified")

    # Test 6: DimensionScores PoI calculation
    dims = DimensionScores(
        quality=1.0, utility=1.0, trust=1.0, fairness=1.0, diversity=1.0
    )
    poi = dims.calculate_poi()
    assert abs(poi - 1.0) < 1e-9, "Perfect dimensions should yield PoI = 1.0"
    print("✓ PoI calculation verified")

    print("=" * 50)
    print("All self-tests passed ✓")
