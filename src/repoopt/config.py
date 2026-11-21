"""Pydantic configuration and experiment manifest models for RepoOpt."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from repoopt import __version__


def _safe_pkg_version(name: str, *fallbacks: str) -> str:
    for pkg in (name, *fallbacks):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return default


def _git_head_commit(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


class ModelConfig(BaseModel):
    """Inference and student model configuration."""

    teacher_provider: str = "openrouter"
    teacher_model: str = "z-ai/glm-5.3"
    teacher_base_url: str = "https://openrouter.ai/api/v1"
    student_model: str = "Qwen/Qwen3.5-9B"
    student_renderer: str = "qwen3_5"
    max_tokens: int = 8192
    temperature: float = 0.7
    reasoning_effort: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None = (
        "medium"
    )
    max_concurrency: int = 8

    @property
    def harbor_teacher_model(self) -> str:
        """Return provider/model identifier expected by Harbor."""
        if self.teacher_model.startswith(f"{self.teacher_provider}/"):
            return self.teacher_model
        return f"{self.teacher_provider}/{self.teacher_model}"


class StorageConfig(BaseModel):
    """Artifact persistence backend configuration."""

    backend: Literal["local", "s3"] = "local"
    local_root: Path = Path("artifacts_store")
    s3_bucket: str = "repoopt"
    s3_prefix: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    flush_interval_sec: float = 15.0

    @classmethod
    def from_env(cls, *, default_local_root: Path = Path("artifacts_store")) -> "StorageConfig":
        """Build StorageConfig from environment variables with safe local defaults."""
        raw_backend = os.environ.get("REPOOPT_STORAGE_BACKEND", "local").strip().lower()
        backend: Literal["local", "s3"] = "s3" if raw_backend == "s3" else "local"
        local_root_env = os.environ.get("REPOOPT_LOCAL_STORE_ROOT", "").strip()
        return cls(
            backend=backend,
            local_root=Path(local_root_env) if local_root_env else default_local_root,
            s3_bucket=os.environ.get("REPOOPT_S3_BUCKET", "").strip() or "repoopt",
            s3_prefix=os.environ.get("REPOOPT_S3_PREFIX", "").strip(),
            s3_region=os.environ.get("AWS_DEFAULT_REGION", "").strip() or None,
            s3_endpoint_url=os.environ.get("REPOOPT_S3_ENDPOINT_URL", "").strip() or None,
        )


class TrackingConfig(BaseModel):
    """MLflow / Databricks run and trace observability configuration."""

    enabled: bool = False
    tracking_uri: str = "databricks"
    experiment_name: str = "/Shared/repoopt"
    log_artifacts: bool = True

    @classmethod
    def from_env(cls) -> "TrackingConfig":
        """Build TrackingConfig from MLFLOW_* and REPOOPT_MLFLOW_* environment variables."""
        return cls(
            enabled=_env_bool("REPOOPT_MLFLOW_ENABLED", False),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "").strip() or "databricks",
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip() or "/Shared/repoopt",
            log_artifacts=_env_bool("REPOOPT_MLFLOW_LOG_ARTIFACTS", True),
        )


class ExecutionConfig(BaseModel):
    """Harbor execution backend and resource budget configuration."""

    environment_backend: Literal["docker", "ray"] = "docker"
    agent_name: str = "pi"
    tool_schema_version: str = "pi-tools-v1"
    pi_tools: tuple[str, ...] = ("read", "write", "edit", "bash")
    agent_timeout_sec: float = 900.0
    verifier_timeout_sec: float = 300.0
    max_turns: int = 40
    skip_agent_install: bool = True
    cpus: int = 2
    memory_mb: int = 4096
    gpus: int = 0


class ExperimentConfig(BaseModel):
    """Top-level configuration for a RepoOpt experiment."""

    experiment_id: str = "local-pilot"
    models: ModelConfig = Field(default_factory=ModelConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class ExperimentManifest(BaseModel):
    """Pinned reproducibility manifest persisted at experiments/<id>/manifest.json."""

    schema_version: str = "repoopt-manifest-v1"
    experiment_id: str
    created_at: str
    repoopt_version: str
    repo_commit: str | None
    pinned_packages: dict[str, str]
    models: ModelConfig
    execution: ExecutionConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        *,
        workspace_root: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "ExperimentManifest":
        root = workspace_root or Path.cwd()
        return cls(
            experiment_id=config.experiment_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            repoopt_version=__version__,
            repo_commit=_git_head_commit(root),
            pinned_packages={
                "harbor": _safe_pkg_version("harbor"),
                "mlflow-skinny": _safe_pkg_version("mlflow-skinny", "mlflow"),
                "tinker": _safe_pkg_version("tinker"),
                "tinker-cookbook": _safe_pkg_version("tinker-cookbook"),
                "transformers": _safe_pkg_version("transformers"),
                "torch": _safe_pkg_version("torch"),
                "pydantic": _safe_pkg_version("pydantic"),
            },
            models=config.models,
            execution=config.execution,
            tracking=config.tracking,
            extra=extra or {},
        )
