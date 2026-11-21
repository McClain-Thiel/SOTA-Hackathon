"""Storage interfaces, SHA-256 hashing, and canonical Section 7 S3/local path layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest for arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for hashing and storage."""
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ArtifactWriteReceipt:
    """Receipt returned by an ArtifactStore write operation."""

    key: str
    uri: str
    sha256: str
    size_bytes: int
    skipped_existing: bool = False


class StoragePaths:
    """Canonical Section 7 key layout under an experiment root."""

    @staticmethod
    def experiment_manifest(experiment_id: str) -> str:
        return f"experiments/{experiment_id}/manifest.json"

    @staticmethod
    def task_benchmark(experiment_id: str, task_id: str) -> str:
        return f"experiments/{experiment_id}/tasks/{task_id}/benchmark.json"

    @staticmethod
    def task_image(experiment_id: str, task_id: str) -> str:
        return f"experiments/{experiment_id}/tasks/{task_id}/image.json"

    @staticmethod
    def attempt_prefix(experiment_id: str, task_id: str, attempt_id: str) -> str:
        return f"experiments/{experiment_id}/tasks/{task_id}/attempts/{attempt_id}"

    @classmethod
    def raw_provider_event(cls, experiment_id: str, task_id: str, attempt_id: str, turn_index: int) -> str:
        prefix = cls.attempt_prefix(experiment_id, task_id, attempt_id)
        return f"{prefix}/raw-provider-events/turn-{turn_index:04d}.json"

    @classmethod
    def event_segment(
        cls,
        experiment_id: str,
        task_id: str,
        attempt_id: str,
        seq_index: int,
        content_sha256: str,
    ) -> str:
        prefix = cls.attempt_prefix(experiment_id, task_id, attempt_id)
        return f"{prefix}/events/seg-{seq_index:06d}-{content_sha256[:12]}.jsonl"

    @classmethod
    def attempt_trajectory(cls, experiment_id: str, task_id: str, attempt_id: str) -> str:
        return f"{cls.attempt_prefix(experiment_id, task_id, attempt_id)}/trajectory.json"

    @classmethod
    def attempt_patch(cls, experiment_id: str, task_id: str, attempt_id: str) -> str:
        return f"{cls.attempt_prefix(experiment_id, task_id, attempt_id)}/candidate.patch"

    @classmethod
    def attempt_evaluation(cls, experiment_id: str, task_id: str, attempt_id: str) -> str:
        return f"{cls.attempt_prefix(experiment_id, task_id, attempt_id)}/evaluation.json"

    @classmethod
    def attempt_completion(cls, experiment_id: str, task_id: str, attempt_id: str) -> str:
        return f"{cls.attempt_prefix(experiment_id, task_id, attempt_id)}/completion.json"

    @staticmethod
    def training_prefix(experiment_id: str, model_id: str) -> str:
        safe_model = model_id.replace("/", "__")
        return f"experiments/{experiment_id}/training/qwen/{safe_model}"

    @classmethod
    def training_manifest(cls, experiment_id: str, model_id: str) -> str:
        return f"{cls.training_prefix(experiment_id, model_id)}/manifest.json"

    @classmethod
    def training_conversion_report(cls, experiment_id: str, model_id: str) -> str:
        return f"{cls.training_prefix(experiment_id, model_id)}/conversion-report.json"

    @classmethod
    def training_example(cls, experiment_id: str, model_id: str, example_name: str) -> str:
        return f"{cls.training_prefix(experiment_id, model_id)}/examples/{example_name}"


@runtime_checkable
class ArtifactStore(Protocol):
    """Protocol implemented by local and S3 artifact stores."""

    def put_bytes(self, key: str, data: bytes, *, immutable: bool = False, content_type: str = "application/octet-stream") -> ArtifactWriteReceipt:
        """Persist raw bytes under key. When immutable=True, identical existing objects are not overwritten."""
        ...

    def put_json(self, key: str, payload: Any, *, immutable: bool = False) -> ArtifactWriteReceipt:
        """Serialize payload deterministically as JSON and persist it under key."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Read raw bytes stored under key."""
        ...

    def get_json(self, key: str) -> Any:
        """Read and decode JSON stored under key."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if key exists in the store."""
        ...

    def list_prefix(self, prefix: str) -> list[str]:
        """Return sorted relative keys matching prefix."""
        ...

    def uri_for(self, key: str) -> str:
        """Return canonical URI (file:// or s3://) for key."""
        ...
