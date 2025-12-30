r"""
Tests for deterministic chat-to-knowledge pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipelines.chat_to_knowledge_pipeline import (
    _deterministic_embedding,
    run_chat_to_knowledge_pipeline,
)


def _write_chat_json(path: Path, mapping: dict, title: str = "Test Chat") -> None:
    payload = {"title": title, "mapping": mapping}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_mapping(messages):
    mapping = {}
    parent_id = None
    for message_id, role, content, create_time in messages:
        mapping[message_id] = {
            "message": {
                "author": {"role": role},
                "content": {"parts": [content]},
                "create_time": create_time,
            },
            "parent": parent_id,
        }
        parent_id = message_id
    return mapping


@pytest.mark.asyncio
async def test_pipeline_determinism_and_manifest(tmp_path):
    root = tmp_path / "chat"
    root.mkdir()

    messages = [
        ("a_msg", "user", "hello alpha", 1700000000.0),
        ("b_msg", "assistant", "response beta", 1700000001.0),
    ]
    mapping = _make_mapping(messages)
    _write_chat_json(root / "conv_one.json", mapping)

    result_one = await run_chat_to_knowledge_pipeline(root, evidence_dir=tmp_path)
    result_two = await run_chat_to_knowledge_pipeline(root, evidence_dir=tmp_path)

    assert result_one.dataset_digest == result_two.dataset_digest
    assert [r.stable_id for r in result_one.messages] == [
        r.stable_id for r in result_two.messages
    ]
    assert result_one.manifest["nodes"] == result_two.manifest["nodes"]

    ordered = [
        (r.source_rel_path, r.conversation_id, r.message_id)
        for r in result_one.messages
    ]
    assert ordered == sorted(ordered)

    for node in result_one.manifest["nodes"]:
        assert "content" not in node
        assert "text" not in node


@pytest.mark.asyncio
async def test_recall_returns_expected_episode(tmp_path):
    root = tmp_path / "chat"
    root.mkdir()

    messages = [
        ("m1", "user", "nebula_z1 unique_token", 1700000000.0),
        ("m2", "assistant", "gamma_x9 different_token", 1700000001.0),
    ]
    mapping = _make_mapping(messages)
    _write_chat_json(root / "conv_two.json", mapping)

    result = await run_chat_to_knowledge_pipeline(root)

    query = _deterministic_embedding("nebula_z1 unique_token", 768)
    assert isinstance(query, np.ndarray)
    results = await result.memory.recall_similar(query, k=1)

    assert results
    assert results[0]["episode_id"] == result.messages[0].stable_id
