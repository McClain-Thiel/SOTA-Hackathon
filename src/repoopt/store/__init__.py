"""Artifact persistence backends (local filesystem and S3) for RepoOpt."""

from .base import ArtifactStore, ArtifactWriteReceipt, StoragePaths, sha256_bytes
from .factory import build_artifact_store
from .local import LocalArtifactStore
from .s3 import S3ArtifactStore

__all__ = [
    "ArtifactStore",
    "ArtifactWriteReceipt",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "StoragePaths",
    "build_artifact_store",
    "sha256_bytes",
]
