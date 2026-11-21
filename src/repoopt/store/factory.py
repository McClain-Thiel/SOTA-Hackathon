"""Factory helper to build local or S3 ArtifactStore from StorageConfig."""

from __future__ import annotations

from repoopt.config import StorageConfig
from repoopt.store.base import ArtifactStore
from repoopt.store.local import LocalArtifactStore
from repoopt.store.s3 import S3ArtifactStore


def build_artifact_store(config: StorageConfig) -> ArtifactStore:
    """Instantiate the configured ArtifactStore backend."""
    if config.backend == "local":
        return LocalArtifactStore(config.local_root)
    if config.backend == "s3":
        return S3ArtifactStore(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            region_name=config.s3_region,
            endpoint_url=config.s3_endpoint_url,
        )
    raise ValueError(f"Unsupported storage backend: {config.backend!r}")
