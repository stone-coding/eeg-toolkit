import pytest

from eegkit.storage import (
    InvalidKeyError,
    LocalStorage,
    NotFoundError,
    S3Storage,
    Storage,
    normalize_key,
)


class FakeS3Client:
    """Just enough of the boto3 S3 client API, backed by a dict."""

    class ClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as fh:
            self.objects[f"{bucket}/{key}"] = fh.read()

    def put_object(self, Bucket, Key, Body):
        self.objects[f"{Bucket}/{Key}"] = Body

    def get_object(self, Bucket, Key):
        import io

        try:
            return {"Body": io.BytesIO(self.objects[f"{Bucket}/{Key}"])}
        except KeyError:
            raise self.ClientError("NoSuchKey") from None

    def head_object(self, Bucket, Key):
        try:
            return {"ContentLength": len(self.objects[f"{Bucket}/{Key}"])}
        except KeyError:
            raise self.ClientError("404") from None

    def delete_object(self, Bucket, Key):
        self.objects.pop(f"{Bucket}/{Key}", None)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None, page=2):
        keys = sorted(k.split("/", 1)[1] for k in self.objects if k.startswith(f"{Bucket}/"))
        keys = [k for k in keys if k.startswith(Prefix)]
        start = int(ContinuationToken or 0)
        chunk = keys[start : start + page]
        more = start + page < len(keys)
        resp = {"Contents": [{"Key": k} for k in chunk], "IsTruncated": more}
        if more:
            resp["NextContinuationToken"] = str(start + page)
        return resp


@pytest.fixture(params=["local", "s3"])
def backend(request, tmp_path) -> Storage:
    if request.param == "local":
        return LocalStorage(tmp_path / "root")
    return S3Storage("bucket", prefix="lab", client=FakeS3Client())


def test_roundtrip_exists_size_delete(backend, tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello")
    backend.put_file("x/y/a.bin", src)
    backend.put_bytes("x/b.txt", b"bb")
    assert backend.exists("x/y/a.bin") and not backend.exists("x/none")
    assert backend.size("x/y/a.bin") == 5
    assert backend.read_bytes("x/y/a.bin") == b"hello"
    assert backend.get_file("x/b.txt", tmp_path / "out" / "b.txt").read_bytes() == b"bb"
    backend.delete("x/b.txt")
    assert not backend.exists("x/b.txt")


def test_list_prefix_sorted_and_paginated(backend):
    for k in ["d/3", "d/1", "d/2", "e/1"]:
        backend.put_bytes(k, b"1")
    assert list(backend.list("d/")) == ["d/1", "d/2", "d/3"]
    assert list(backend.list()) == ["d/1", "d/2", "d/3", "e/1"]


def test_missing_key_raises(backend):
    for call in (backend.open, backend.size, backend.delete):
        with pytest.raises(NotFoundError):
            call("nope")


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs", "a/../../b", "", "a\\b"])
def test_invalid_keys_rejected(backend, bad):
    with pytest.raises(InvalidKeyError):
        normalize_key(bad)
    with pytest.raises(InvalidKeyError):
        backend.put_bytes(bad, b"x")


def test_local_symlink_escape_blocked(tmp_path):
    root = tmp_path / "root"
    store = LocalStorage(root)
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(InvalidKeyError):
        store.read_bytes("link/secret.txt")


def test_s3_prefix_applied(tmp_path):
    client = FakeS3Client()
    s3 = S3Storage("bucket", prefix="lab/", client=client)
    s3.put_bytes("k", b"1")
    assert "bucket/lab/k" in client.objects
    assert list(s3.list()) == ["k"]
