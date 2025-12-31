r"""
Tests for deterministic chat-to-knowledge pipeline.
"""

import json
import sys
from pathlib import Path

import hashlib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge_bridge import EdgeType
from core.pipelines.chat_to_knowledge_pipeline import (
    _deterministic_embedding,
    _stable_message_id,
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


def _make_mapping_with_parents(messages):
    mapping = {}
    for message_id, role, content, create_time, parent_id in messages:
        mapping[message_id] = {
            "message": {
                "author": {"role": role},
                "content": {"parts": [content]},
                "create_time": create_time,
            },
            "parent": parent_id,
        }
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


@pytest.mark.asyncio
async def test_parent_ordering_sets_next_edges(tmp_path):
    root = tmp_path / "chat"
    root.mkdir()

    conversation_id = "conv_parent"
    messages = [
        ("m2", "user", "second node", 1700000002.0, "m1"),
        ("m1", "user", "root node", 1700000001.0, None),
        ("m3", "assistant", "third node", 1700000003.0, "m2"),
    ]
    mapping = _make_mapping_with_parents(messages)
    payload = {"conversation_id": conversation_id, "mapping": mapping}
    (root / "conv_parent.json").write_text(json.dumps(payload), encoding="utf-8")

    result = await run_chat_to_knowledge_pipeline(root)
    source_rel_path = "conv_parent.json"

    def stable_id(message_id: str, role: str, content: str) -> str:
        text_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return _stable_message_id(
            source_rel_path, conversation_id, message_id, role, text_sha256
        )

    m1_id = stable_id("m1", "user", "root node")
    m2_id = stable_id("m2", "user", "second node")
    m3_id = stable_id("m3", "assistant", "third node")

    assert result.bridge.graph.get_edge(m1_id, m2_id, EdgeType.NEXT) is not None
    assert result.bridge.graph.get_edge(m2_id, m3_id, EdgeType.NEXT) is not None
    assert result.bridge.graph.get_edge(m2_id, m1_id, EdgeType.NEXT) is None


@pytest.mark.asyncio
async def test_skipped_files_include_hash_and_size(tmp_path):
    root = tmp_path / "chat"
    root.mkdir()

    mapping = _make_mapping([("m1", "user", "ok", 1700000000.0)])
    _write_chat_json(root / "conv_ok.json", mapping)
    (root / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")

    result = await run_chat_to_knowledge_pipeline(root)
    skipped = result.manifest["skipped_files"]
    assert skipped

    entry = next(
        item for item in skipped if item["source_rel_path"] == "bad.json"
    )
    assert entry["reason"] == "non_object_json"
    assert entry["sha256"]
    assert entry["size_bytes"] > 0
