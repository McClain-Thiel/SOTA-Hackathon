"""Local filesystem implementation of ArtifactStore with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .base import ArtifactWriteReceipt, canonical_json_bytes, sha256_bytes


class LocalArtifactStore:
    """Stores artifacts under a local root directory with atomic temp-file renames."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        clean = key.lstrip("/")
        target = (self.root / clean).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Key escapes store root: {key!r}")
        return target

    def uri_for(self, key: str) -> str:
        return self._resolve(key).as_uri()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        immutable: bool = False,
        content_type: str = "application/octet-stream",
    ) -> ArtifactWriteReceipt:
        del content_type
        digest = sha256_bytes(data)
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)

        if immutable and target.exists():
            existing_digest = sha256_bytes(target.read_bytes())
            if existing_digest != digest:
                raise FileExistsError(
                    f"Immutable artifact conflict at {key}: existing sha256={existing_digest}, new sha256={digest}"
                )
            return ArtifactWriteReceipt(
                key=key,
                uri=target.as_uri(),
                sha256=digest,
                size_bytes=len(data),
                skipped_existing=True,
            )

        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

        return ArtifactWriteReceipt(
            key=key,
            uri=target.as_uri(),
            sha256=digest,
            size_bytes=len(data),
            skipped_existing=False,
        )

    def put_json(self, key: str, payload: Any, *, immutable: bool = False) -> ArtifactWriteReceipt:
        return self.put_bytes(
            key,
            canonical_json_bytes(payload),
            immutable=immutable,
            content_type="application/json",
        )

    def get_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def list_prefix(self, prefix: str) -> list[str]:
        clean = prefix.lstrip("/")
        base = (self.root / clean).resolve()
        if base != self.root and self.root not in base.parents:
            raise ValueError(f"Prefix escapes store root: {prefix!r}")
        results: list[str] = []
        if base.is_file():
            return [base.relative_to(self.root).as_posix()]
        if not base.exists():
            # Support partial filename prefix matching under parent dir
            parent = base.parent
            if not parent.exists():
                return []
            stem = base.name
            for path in parent.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(self.root).as_posix()
                    if rel.startswith(clean):
                        results.append(rel)
            return sorted(results)
        for path in base.rglob("*"):
            if path.is_file():
                results.append(path.relative_to(self.root).as_posix())
        return sorted(results)
