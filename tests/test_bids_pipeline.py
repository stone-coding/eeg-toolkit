import json

import pytest
from typer.testing import CliRunner

from eegkit import pipeline
from eegkit.bids import bids_path, load_subject_bids, write_run_bids
from eegkit.cli import app
from eegkit.config import Config
from tests.conftest import make_synthetic_raw


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate(
        {
            "subjects": [1, 2],
            "runs": [4],
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "bids_root": str(tmp_path / "bids"),
                "output_dir": str(tmp_path / "out"),
                "cache_dir": str(tmp_path / "out/cache"),
                "catalog_db": str(tmp_path / "out/catalog.sqlite"),
            },
            "analysis": {"decoding": {"n_splits": 4}},
        }
    )


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Pipeline reads a synthetic recording instead of downloaded/BIDS data."""
    src = tmp_path / "S001R04.edf"
    src.write_bytes(b"x" * 10)
    monkeypatch.setattr(pipeline, "_source_files", lambda cfg, subject: [src])
    calls = {"n": 0}

    def fake_load(cfg, subject):
        calls["n"] += 1
        if subject == 2 and calls.get("fail2"):
            raise OSError("disk on fire")
        return make_synthetic_raw(seed=subject)

    monkeypatch.setattr(pipeline, "_load_raw", fake_load)
    return calls


def test_bids_roundtrip(tmp_path):
    raw = make_synthetic_raw()
    bp = bids_path(1, 4, tmp_path / "bids")
    write_run_bids(raw, bp)
    assert bp.fpath.exists()
    assert (bp.fpath.parent / "sub-001_ses-01_task-motorimagery_run-04_events.tsv").exists()
    back = load_subject_bids(1, [4], tmp_path / "bids")
    assert back.ch_names == raw.ch_names
    assert {"left", "right", "rest"} <= set(back.annotations.description)
    assert (tmp_path / "bids" / "participants.tsv").exists()


def test_pipeline_outputs_and_group(cfg, patched):
    results, summary = pipeline.run_batch(cfg)
    assert [r.status for r in results] == ["ok", "ok"]
    out = cfg.paths.output_dir
    for name in ("status.json", "qc.json", "bandpower.csv", "erders_summary.csv", "decoding.json"):
        assert (out / "subjects/sub-001" / name).exists()
    assert (out / "subjects/sub-001/figures/erders.png").exists()
    for name in ("decoding.csv", "erders_summary.csv", "summary.json", "figures/decoding.png"):
        assert (out / "group" / name).exists()
    assert summary["n_subjects_ok"] == 2
    assert summary["decoding"]["mean_accuracy"] > 0.8
    assert summary["erders_mean_pct"]["right/C3/mu"] < -50


def test_cache_skips_work_and_invalidates_by_stage(cfg, patched):
    pipeline.run_batch(cfg)
    assert patched["n"] == 2
    r2, _ = pipeline.run_batch(cfg)
    assert patched["n"] == 2  # nothing reloaded
    assert all(r.cached_stages == ["epochs", "analysis"] for r in r2)

    # analysis-only config change: epochs reused, analysis recomputed
    cfg2 = cfg.model_copy(deep=True)
    cfg2.analysis.bands = {"mu": (9.0, 12.0)}
    r3, _ = pipeline.run_batch(cfg2)
    assert patched["n"] == 2 and all(r.cached_stages == ["epochs"] for r in r3)

    # preprocessing change invalidates everything
    cfg3 = cfg.model_copy(deep=True)
    cfg3.preprocess.h_freq = 35.0
    r4, _ = pipeline.run_batch(cfg3)
    assert patched["n"] == 4 and all(r.cached_stages == [] for r in r4)

    pipeline.run_batch(cfg3, force=True)
    assert patched["n"] == 6


def test_one_failing_subject_does_not_stop_batch(cfg, patched):
    patched["fail2"] = True
    results, summary = pipeline.run_batch(cfg)
    assert [r.status for r in results] == ["ok", "failed"]
    assert "disk on fire" in results[1].error
    assert summary["n_subjects_ok"] == 1 and "2" in {str(k) for k in summary["failed"]}
    status = json.loads((cfg.paths.output_dir / "subjects/sub-002/status.json").read_text())
    assert status["status"] == "failed"


def test_missing_source_is_reported(cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "_source_files", lambda cfg, subject: [])
    res = pipeline.process_subject(cfg, 1)
    assert res.status == "failed" and "no recordings" in res.error


def test_aggregate_requires_a_success(cfg):
    with pytest.raises(RuntimeError):
        pipeline.aggregate(cfg, [pipeline.SubjectResult(1, status="failed")])


def test_report_subject(cfg, patched):
    path = pipeline.report_subject(cfg, 1)
    text = path.read_text()
    assert "Subject 1" in text and "figures/erders.png" in text and "| right" in text


def test_cli(cfg, patched, tmp_path):
    runner = CliRunner()
    assert "eegkit" in runner.invoke(app, ["--version"]).output
    bad = tmp_path / "bad.yaml"
    bad.write_text("subjects: []\n")
    r = runner.invoke(app, ["run", "-c", str(bad)])
    assert r.exit_code == 2 and "Invalid configuration" in r.output
    good = tmp_path / "good.yaml"
    good.write_text(cfg.model_dump_json())  # JSON is valid YAML
    r = runner.invoke(app, ["run", "-c", str(good)])
    assert r.exit_code == 0 and "Group: n=2" in r.output
    r = runner.invoke(app, ["report", "--subject", "1", "-c", str(good)])
    assert r.exit_code == 0 and "report.md" in r.output
