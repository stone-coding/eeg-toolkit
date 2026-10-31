"""Storage abstraction: a local-filesystem backend and an S3-compatible backend.

The catalog and API address files by *key* (a POSIX-style relative path such as
``bids/sub-001/ses-01/eeg/....edf``), so swapping local disk for an object store (S3, MinIO, a
lab NAS gateway) only means constructing a different :class:`Storage`.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


class StorageError(Exception):
    """Base class for storage failures."""


class NotFoundError(StorageError, FileNotFoundError):
    """The requested key does not exist."""


class InvalidKeyError(StorageError, ValueError):
    """The key is absolute, empty or tries to escape the storage root."""


def normalize_key(key: str) -> str:
    """Validate a key and return its normalized POSIX form (no leading slash, no ``..``)."""
    p = PurePosixPath(key)
    if not key or p.is_absolute() or ".." in p.parts or "\\" in key or "\0" in key:
        raise InvalidKeyError(f"invalid storage key: {key!r}")
    return str(p)


class Storage(ABC):
    """Minimal object-store interface."""

    @abstractmethod
    def put_file(self, key: str, src: str | Path) -> None:
        """Store the local file ``src`` under ``key`` (overwriting)."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Open ``key`` for streaming reads (raises :class:`NotFoundError`)."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def list(self, prefix: str = "") -> Iterator[str]:
        """Yield keys beginning with ``prefix`` in sorted order."""

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def read_bytes(self, key: str) -> bytes:
        with self.open(key) as fh:
            return fh.read()

    def get_file(self, key: str, dest: str | Path) -> Path:
        """Download ``key`` to the local path ``dest``."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.open(key) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        return dest


class LocalStorage(Storage):
    """Files under a root directory. Keys can never resolve outside the root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / normalize_key(key)).resolve()
        if self.root not in path.parents and path != self.root:
            raise InvalidKeyError(f"key escapes storage root: {key!r}")
        return path

    def put_file(self, key: str, src: str | Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).resolve() != dest:
            shutil.copyfile(src, dest)

    def put_bytes(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def open(self, key: str) -> BinaryIO:
        path = self._path(key)
        if not path.is_file():
            raise NotFoundError(key)
        return open(path, "rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.is_file():
            raise NotFoundError(key)
        return path.stat().st_size

    def list(self, prefix: str = "") -> Iterator[str]:
        base = self._path(prefix) if prefix else self.root
        start = base if base.is_dir() else base.parent
        for p in sorted(start.rglob("*")):
            if p.is_file():
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    yield key

    def delete(self, key: str) -> None:
        path = self._path(key)
        if not path.is_file():
            raise NotFoundError(key)
        path.unlink()


class S3Storage(Storage):
    """S3-compatible backend (AWS S3, MinIO, Ceph...) built on a boto3 client.

    Pass ``client`` to inject a preconfigured/fake client; otherwise one is created lazily
    (``endpoint_url`` selects a non-AWS service such as MinIO).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3  # optional dependency: pip install eeg-toolkit[s3]

            client = boto3.client("s3", endpoint_url=endpoint_url)
        self.client = client

    def _k(self, key: str) -> str:
        key = normalize_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _strip(self, full: str) -> str:
        return full[len(self.prefix) + 1 :] if self.prefix else full

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def put_file(self, key: str, src: str | Path) -> None:
        self.client.upload_file(str(src), self.bucket, self._k(key))

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def open(self, key: str) -> BinaryIO:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as exc:
            if self._is_missing(exc):
                raise NotFoundError(key) from exc
            raise
        return resp["Body"]

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as exc:
            if self._is_missing(exc):
                return False
            raise
        return True

    def size(self, key: str) -> int:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as exc:
            if self._is_missing(exc):
                raise NotFoundError(key) from exc
            raise
        return int(head["ContentLength"])

    def list(self, prefix: str = "") -> Iterator[str]:
        full = self._k(prefix) if prefix else (self.prefix + "/" if self.prefix else "")
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                yield self._strip(obj["Key"])
            if not resp.get("IsTruncated"):
                return
            token = resp["NextContinuationToken"]

    def delete(self, key: str) -> None:
        if not self.exists(key):
            raise NotFoundError(key)
        self.client.delete_object(Bucket=self.bucket, Key=self._k(key))
