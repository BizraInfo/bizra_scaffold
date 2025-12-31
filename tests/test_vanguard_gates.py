from types import SimpleNamespace

from core.boot import KernelLaws, _calculate_boot_vector


def test_ihsan_gate_failure_without_crypto():
    """
    Fail-closed: if blake3 is missing, correctness=0.0 => Ihsan score below threshold.
    """
    caps = SimpleNamespace(
        force_lite=True,
        numpy=False,
        faiss=False,
        neo4j=False,
        z3=False,
        blake3=False,
        neo4j_configured=False,
        l3_mode="basic",
        l4_mode="disabled",
    )
    score = _calculate_boot_vector(caps)
    assert score < KernelLaws.IHSAN.MIN_SCORE_THRESHOLD


def test_ihsan_gate_success_lite_minimal_with_crypto():
    """
    Lite nodes are valid citizens:
    No numpy/faiss/neo4j needed, but crypto must exist.
    """
    caps = SimpleNamespace(
        force_lite=True,
        numpy=False,
        faiss=False,
        neo4j=False,
        z3=False,
        blake3=True,
        neo4j_configured=False,
        l3_mode="basic",
        l4_mode="disabled",
    )
    score = _calculate_boot_vector(caps)
    assert score >= KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
