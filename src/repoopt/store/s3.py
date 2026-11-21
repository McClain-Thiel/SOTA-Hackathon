"""S3 implementation of ArtifactStore with SHA-256 metadata and idempotent immutable puts."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .base import ArtifactWriteReceipt, canonical_json_bytes, sha256_bytes


class S3ArtifactStore:
    """Persists RepoOpt artifacts to Amazon S3 under a configured bucket and key prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client or boto3.client(
            "s3",
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    def _full_key(self, key: str) -> str:
        clean = key.lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full_key(key)}"

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        immutable: bool = False,
        content_type: str = "application/octet-stream",
    ) -> ArtifactWriteReceipt:
        full_key = self._full_key(key)
        digest = sha256_bytes(data)

        if immutable:
            try:
                head = self._client.head_object(Bucket=self.bucket, Key=full_key)
                existing_sha = head.get("Metadata", {}).get("sha256")
                if existing_sha and existing_sha != digest:
                    raise FileExistsError(
                        f"Immutable S3 object conflict at {full_key}: existing sha256={existing_sha}, new={digest}"
                    )
                if existing_sha == digest:
                    return ArtifactWriteReceipt(
                        key=key,
                        uri=self.uri_for(key),
                        sha256=digest,
                        size_bytes=len(data),
                        skipped_existing=True,
                    )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise

        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": full_key,
            "Body": data,
            "ContentType": content_type,
            "Metadata": {"sha256": digest},
        }
        self._client.put_object(**kwargs)
        return ArtifactWriteReceipt(
            key=key,
            uri=self.uri_for(key),
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
        resp = self._client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def list_prefix(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        strip_prefix = f"{self.prefix}/" if self.prefix else ""
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for item in page.get("Contents", []):
                raw_key = item["Key"]
                if strip_prefix and raw_key.startswith(strip_prefix):
                    keys.append(raw_key[len(strip_prefix) :])
                else:
                    keys.append(raw_key)
        return sorted(keys)
