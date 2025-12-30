"""
Deterministic chat-to-knowledge pipeline.

Ingests chat history JSON, computes stable IDs and SNR scores,
stores hashed metadata in memory tiers, and emits evidence manifests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.knowledge_bridge import KnowledgeGraphBridge
from core.layers.memory_layers_v2 import L3EpisodicMemoryV2
from core.snr_scorer import SNRThresholds

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RawMessage:
    message_id: str
    role: str
    content: str
    timestamp: float


@dataclass(frozen=True)
class MessageRecord:
    stable_id: str
    conversation_id: str
    message_id: str
    role: str
    created_at: Optional[str]
    text_sha256: str
    snr: float
    snr_level: str
    ihsan: float
    source_rel_path: str

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "role": self.role,
            "created_at": self.created_at,
            "text_sha256": self.text_sha256,
            "snr": self.snr,
            "snr_level": self.snr_level,
            "ihsan": self.ihsan,
            "source_rel_path": self.source_rel_path,
        }

    def to_graph_dict(self) -> Dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "message_id": self.message_id,
            "role": self.role,
            "text_sha256": self.text_sha256,
            "snr": self.snr,
            "ihsan": self.ihsan,
        }

    def to_memory_content(self) -> Dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "role": self.role,
            "created_at": self.created_at,
            "text_sha256": self.text_sha256,
            "snr": self.snr,
            "snr_level": self.snr_level,
            "ihsan": self.ihsan,
            "source_rel_path": self.source_rel_path,
        }


@dataclass
class PipelineResult:
    dataset_digest: str
    manifest_path: Optional[Path]
    manifest: Dict[str, Any]
    messages: List[MessageRecord]
    bridge: KnowledgeGraphBridge
    memory: L3EpisodicMemoryV2
    merkle_tail: str


def _iter_chat_files(root_path: Path) -> List[Path]:
    files = sorted(root_path.glob("**/*.json"))
    return [f for f in files if "manifest" not in str(f).lower()]


def _compute_file_sha256(path: Path, chunk_size: int = 8192) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compute_dataset_digest(files: Sequence[Path], root_path: Path) -> str:
    items = []
    for path in files:
        rel_path = path.relative_to(root_path).as_posix()
        items.append((rel_path, _compute_file_sha256(path)))

    items.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for rel_path, file_hash in items:
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\x00")

    return digest.hexdigest()


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _compute_chat_snr(text: str, thresholds: SNRThresholds) -> Tuple[float, str, float]:
    if not text:
        return 0.0, thresholds.classify(0.0, 0.0, 0.0).name, 0.0

    tokens = _tokenize(text)
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(1, token_count)
    length_factor = min(len(text) / 400.0, 1.0)

    has_code = "```" in text
    has_list = "\n- " in text or "\n1. " in text or "\n* " in text
    structure_bonus = (0.15 if has_code else 0.0) + (0.10 if has_list else 0.0)

    signal = (0.45 * unique_ratio) + (0.35 * length_factor) + structure_bonus
    repetition = 1.0 - unique_ratio
    short_penalty = 0.4 if len(text) < 80 else 0.0
    noise = (0.5 * repetition) + (0.5 * short_penalty)

    snr = signal / (noise + 1e-6)
    ihsan = 0.98
    confidence = 0.9 if token_count >= 5 else 0.6
    level = thresholds.classify(snr, ihsan, confidence)

    return snr, level.name, ihsan


def _stable_message_id(
    source_rel_path: str,
    conversation_id: str,
    message_id: str,
    role: str,
    text_sha256: str,
) -> str:
    payload = "|".join(
        [source_rel_path, conversation_id, message_id, role, text_sha256]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"msg_{digest[:32]}"


def _timestamp_to_iso(timestamp: float) -> Optional[str]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _format_parse_error(error: Exception) -> str:
    if isinstance(error, UnicodeDecodeError):
        return "UnicodeDecodeError"
    if isinstance(error, json.JSONDecodeError):
        return "JSONDecodeError"
    return error.__class__.__name__


def _parse_conversation(
    file_path: Path,
) -> Tuple[Optional[str], List[RawMessage], Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [], _format_parse_error(exc)

    if not isinstance(data, dict):
        return None, [], "non_object_json"

    mapping = data.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return None, [], "missing_mapping"

    conversation_id = (
        str(data.get("conversation_id"))
        if data.get("conversation_id")
        else str(data.get("id") or file_path.stem)
    )
    messages: List[RawMessage] = []

    for node_id, node_data in mapping.items():
        message = node_data.get("message")
        if not message:
            continue

        author = message.get("author", {})
        role = author.get("role", "unknown")
        content_obj = message.get("content", {})
        parts = content_obj.get("parts", [])
        text_content = "".join(str(part) for part in parts)
        timestamp = float(message.get("create_time") or 0.0)

        messages.append(
            RawMessage(
                message_id=str(node_id),
                role=str(role),
                content=text_content,
                timestamp=timestamp,
            )
        )

    if not messages:
        return None, [], "empty_messages"

    return conversation_id, messages, None


def _deterministic_embedding(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm

    return vector


async def run_chat_to_knowledge_pipeline(
    root_path: Path,
    evidence_dir: Optional[Path] = None,
    embedding_dim: int = 768,
    include_generated_at: bool = False,
) -> PipelineResult:
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Chat root not found: {root_path}")

    files = _iter_chat_files(root_path)
    dataset_digest = _compute_dataset_digest(files, root_path)
    thresholds = SNRThresholds()

    bridge = KnowledgeGraphBridge()
    memory = L3EpisodicMemoryV2(embedding_dim=embedding_dim)
    messages: List[MessageRecord] = []
    conversation_ids: set[str] = set()
    skipped_files: List[Dict[str, str]] = []

    for file_path in files:
        source_rel_path = file_path.relative_to(root_path).as_posix()
        conversation_id, raw_messages, error = _parse_conversation(file_path)
        if error or conversation_id is None:
            skipped_files.append(
                {
                    "source_rel_path": source_rel_path,
                    "reason": error or "unknown_error",
                }
            )
            continue

        conversation_ids.add(conversation_id)
        raw_messages = sorted(raw_messages, key=lambda item: item.message_id)

        chat_nodes: List[Dict[str, Any]] = []
        for raw in raw_messages:
            text_sha256 = hashlib.sha256(raw.content.encode("utf-8")).hexdigest()
            snr, snr_level, ihsan = _compute_chat_snr(raw.content, thresholds)

            stable_id = _stable_message_id(
                source_rel_path=source_rel_path,
                conversation_id=conversation_id,
                message_id=raw.message_id,
                role=raw.role,
                text_sha256=text_sha256,
            )

            record = MessageRecord(
                stable_id=stable_id,
                conversation_id=conversation_id,
                message_id=raw.message_id,
                role=raw.role,
                created_at=_timestamp_to_iso(raw.timestamp),
                text_sha256=text_sha256,
                snr=snr,
                snr_level=snr_level,
                ihsan=ihsan,
                source_rel_path=source_rel_path,
            )
            messages.append(record)
            chat_nodes.append(record.to_graph_dict())

            embedding = _deterministic_embedding(raw.content, embedding_dim)
            await memory.store_episode(stable_id, record.to_memory_content(), embedding)

        bridge.ingest_chat_nodes(chat_nodes, conversation_id, source_rel_path)

    messages_sorted = sorted(
        messages,
        key=lambda item: (item.source_rel_path, item.conversation_id, item.message_id),
    )
    merkle_tail = memory.merkle_root.hex()

    manifest = {
        "schema_version": "1.0",
        "input_root": str(root_path),
        "dataset_digest": dataset_digest,
        "nodes": [record.to_manifest_dict() for record in messages_sorted],
        "merkle_tail": merkle_tail,
        "counts": {
            "conversations": len(conversation_ids),
            "messages": len(messages_sorted),
        },
        "skipped_files": skipped_files,
    }
    if include_generated_at:
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    manifest_path: Optional[Path] = None
    if evidence_dir:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        filename = f"chat_to_knowledge_manifest.{dataset_digest}.json"
        manifest_path = evidence_dir / filename
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    return PipelineResult(
        dataset_digest=dataset_digest,
        manifest_path=manifest_path,
        manifest=manifest,
        messages=messages_sorted,
        bridge=bridge,
        memory=memory,
        merkle_tail=merkle_tail,
    )


def run_chat_to_knowledge_pipeline_sync(
    root_path: Path,
    evidence_dir: Optional[Path] = None,
    embedding_dim: int = 768,
    include_generated_at: bool = False,
) -> PipelineResult:
    return asyncio.run(
        run_chat_to_knowledge_pipeline(
            root_path=root_path,
            evidence_dir=evidence_dir,
            embedding_dim=embedding_dim,
            include_generated_at=include_generated_at,
        )
    )


def _parse_args() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic chat-to-knowledge pipeline"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing chat history JSON files",
    )
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="Directory to write evidence manifest",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension for deterministic hashing",
    )
    parser.add_argument(
        "--include-generated-at",
        action="store_true",
        help="Include generated_at timestamp in manifest (non-deterministic)",
    )
    return vars(parser.parse_args())


def main() -> None:
    args = _parse_args()
    run_chat_to_knowledge_pipeline_sync(
        root_path=Path(args["root"]),
        evidence_dir=Path(args["evidence_dir"]) if args["evidence_dir"] else None,
        embedding_dim=args["embedding_dim"],
        include_generated_at=args["include_generated_at"],
    )


if __name__ == "__main__":
    main()
