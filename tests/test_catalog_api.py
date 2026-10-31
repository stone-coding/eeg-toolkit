import json

import pytest
from fastapi.testclient import TestClient

from eegkit.api import create_app
from eegkit.bids import bids_path, write_run_bids
from eegkit.catalog import Catalog
from eegkit.config import Config
from eegkit.epochs import make_epochs
from eegkit.storage import LocalStorage
from tests.conftest import make_synthetic_raw


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """Two subjects x two runs of BIDS; sub-001 processed ok, sub-002 failed, sub-003 none."""
    tmp = tmp_path_factory.mktemp("lab")
    cfg = Config.model_validate(
        {
            "subjects": [1, 2],
            "paths": {
                "raw_dir": str(tmp / "raw"),
                "bids_root": str(tmp / "data/bids"),
                "output_dir": str(tmp / "out"),
                "cache_dir": str(tmp / "out/cache"),
                "catalog_db": str(tmp / "cat.sqlite"),
                "storage_root": str(tmp / "data"),
            },
        }
    )
    for sub in (1, 2, 3):
        for run in (4, 8):
            raw = make_synthetic_raw(seed=sub * 10 + run)
            write_run_bids(raw, bids_path(sub, run, cfg.paths.bids_root), deid_salt="s")
    cache, out = cfg.paths.cache_dir / "sub-001", cfg.paths.output_dir / "subjects"
    (out / "sub-001").mkdir(parents=True)
    (out / "sub-002").mkdir(parents=True)
    cache.mkdir(parents=True)
    make_epochs(make_synthetic_raw()).save(cache / "epochs-epo.fif", verbose="ERROR")
    (out / "sub-001/bandpower.csv").write_text(
        "subject,condition,channel,band,power\n"
        "1,left,C3,mu,1e-10\n1,right,C3,mu,5e-11\n1,left,C4,beta,2e-11\n"
    )
    (out / "sub-001/decoding.json").write_text("{}")
    (out / "sub-001/erders_summary.csv").write_text("subject\n1\n")
    (out / "sub-001/status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "stage": "outputs",
                "n_epochs": 24,
                "n_dropped": 0,
                "accuracy": 0.9,
                "bad_channels": ["Fz"],
                "updated": "2026-01-01T00:00:00",
            }
        )
    )
    (out / "sub-002/status.json").write_text(
        json.dumps({"status": "failed", "stage": "epochs", "error": "OSError: boom"})
    )
    storage = LocalStorage(cfg.paths.storage_root)
    catalog = Catalog(cfg.paths.catalog_db)
    catalog.index_bids(cfg.paths.bids_root, storage)
    catalog.sync_status(cfg.paths.output_dir)
    catalog.publish_artifacts(cfg, storage)
    return cfg, catalog, storage


@pytest.fixture
def client(lab):
    _, catalog, storage = lab
    return TestClient(create_app(catalog, storage))


def test_indexing_is_idempotent_and_complete(lab):
    cfg, catalog, storage = lab
    assert catalog.index_bids(cfg.paths.bids_root, storage) == 6
    total, runs = catalog.query_runs(limit=100)
    assert total == 6 and len(runs) == 6
    r = runs[0]
    assert r.sampling_rate == 160.0 and r.n_channels == 10 and r.task == "motorimagery"
    assert r.n_events_left == 12 and r.n_events_right == 12
    assert r.file_path.startswith("bids/sub-001/ses-01/eeg/") and storage.exists(r.file_path)
    assert len(catalog.get_run(r.id).channels) == 10
    assert len(catalog.list_subjects()) == 3


def test_catalog_filters(lab):
    _, catalog, _ = lab
    assert catalog.query_runs(subject="sub-002")[0] == 2
    assert catalog.query_runs(task="motorimagery")[0] == 6 and catalog.query_runs(task="x")[0] == 0
    assert catalog.query_runs(status="ok")[0] == 2
    assert catalog.query_runs(status="failed")[0] == 2
    assert catalog.query_runs(status="unprocessed")[0] == 2  # sub-003 has no status.json -> row
    assert catalog.query_runs(min_sampling_rate=200)[0] == 0
    assert catalog.query_runs(max_sampling_rate=160)[0] == 6
    assert catalog.query_runs(min_duration_s=1e6)[0] == 0
    assert catalog.query_runs(min_channels=11)[0] == 0
    total, page = catalog.query_runs(limit=2, offset=4)
    assert total == 6 and len(page) == 2


def test_catalog_has_no_identifying_fields(lab):
    _, catalog, _ = lab
    run = catalog.query_runs(limit=1)[1][0]
    assert run.acq_time is None or run.acq_time.year < 2026
    cols = {
        c.name
        for t in __import__("eegkit.catalog", fromlist=["Base"]).Base.metadata.tables.values()
        for c in t.columns
    }
    assert not {c for c in cols if any(k in c for k in ("name_", "birth", "email", "address"))}


def test_api_health_and_openapi(client):
    assert client.get("/health").json()["status"] == "ok"
    spec = client.get("/openapi.json").json()
    assert {
        "/recordings",
        "/recordings/{run_id}",
        "/subjects/{label}/status",
        "/subjects/{label}/bandpower",
        "/subjects/{label}/epochs",
    } <= set(spec["paths"])
    assert client.get("/docs").status_code == 200


def test_api_list_and_filter_recordings(client):
    page = client.get("/recordings").json()
    assert page["total"] == 6 and len(page["items"]) == 6
    assert client.get("/recordings", params={"subject": "sub-001"}).json()["total"] == 2
    items = client.get("/recordings", params={"status": "ok"}).json()["items"]
    assert {i["subject"] for i in items} == {"sub-001"} and items[0]["status"] == "ok"
    assert (
        client.get("/recordings", params={"limit": 1, "offset": 1}).json()["items"][0]["run_number"]
        == 8
    )
    assert client.get("/recordings", params={"status": "bogus"}).status_code == 422
    assert client.get("/recordings", params={"limit": 0}).status_code == 422
    assert client.get("/recordings", params={"min_sampling_rate": -1}).status_code == 422


def test_api_recording_detail(client):
    rid = client.get("/recordings").json()["items"][0]["id"]
    d = client.get(f"/recordings/{rid}").json()
    assert d["n_samples"] > 0 and len(d["channels"]) == 10 and d["channels"][0]["type"] == "eeg"
    assert client.get("/recordings/9999").status_code == 404


def test_api_subjects_and_status(client):
    assert [s["label"] for s in client.get("/subjects").json()] == ["sub-001", "sub-002", "sub-003"]
    st = client.get("/subjects/sub-001/status").json()
    assert st["status"] == "ok" and st["accuracy"] == 0.9 and st["bad_channels"] == ["Fz"]
    assert client.get("/subjects/sub-002/status").json()["error"] == "OSError: boom"
    assert client.get("/subjects/sub-003/status").json()["status"] == "unprocessed"
    assert client.get("/subjects/sub-999/status").status_code == 404
    kinds = {a["kind"] for a in client.get("/subjects/sub-001").json()["artifacts"]}
    assert kinds == {"epochs", "bandpower", "erders", "decoding"}


def test_api_bandpower(client):
    rows = client.get("/subjects/sub-001/bandpower").json()
    assert len(rows) == 3 and rows[0] == {
        "condition": "left",
        "channel": "C3",
        "band": "mu",
        "power": 1e-10,
    }
    assert len(client.get("/subjects/sub-001/bandpower", params={"band": "mu"}).json()) == 2
    assert len(client.get("/subjects/sub-001/bandpower", params={"channel": "C4"}).json()) == 1
    assert client.get("/subjects/sub-001/bandpower", params={"condition": "up"}).status_code == 422
    assert client.get("/subjects/sub-003/bandpower").status_code == 409  # not processed yet
    assert client.get("/subjects/sub-002/bandpower").status_code == 404  # processed but failed


def test_api_epochs_download(client, lab):
    cfg, _, _ = lab
    r = client.get("/subjects/sub-001/epochs")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert "sub-001_epochs-epo.fif" in r.headers["content-disposition"]
    expected = (cfg.paths.cache_dir / "sub-001/epochs-epo.fif").read_bytes()
    assert r.content == expected and int(r.headers["content-length"]) == len(expected)
    assert client.get("/subjects/sub-003/epochs").status_code == 409
    assert client.get("/subjects/nope/epochs").status_code == 404


def test_api_epochs_missing_from_storage(lab):
    cfg, catalog, storage = lab
    key = "bids/derivatives/eegkit/sub-001/epochs-epo.fif"
    data = storage.read_bytes(key)
    storage.delete(key)
    try:
        assert (
            TestClient(create_app(catalog, storage)).get("/subjects/sub-001/epochs").status_code
            == 404
        )
    finally:
        storage.put_bytes(key, data)
