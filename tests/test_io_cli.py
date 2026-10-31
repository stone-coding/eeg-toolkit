import datetime as dt

import mne
import numpy as np
import pytest
from typer.testing import CliRunner

from eegkit import io as eio
from eegkit.bids import bids_runs_available, load_subject_bids, to_bids
from eegkit.cli import app
from eegkit.deidentify import scan_for_identifiers

runner = CliRunner()


def test_load_run_standardizes_names_and_montage(fake_eegbci, tmp_path):
    raw = eio.load_run(1, 4, tmp_path)
    assert (
        "C3" in raw.ch_names and "FC3" in raw.ch_names and not any("." in c for c in raw.ch_names)
    )
    assert not np.isnan(raw.info["chs"][raw.ch_names.index("C3")]["loc"][:3]).any()
    assert {"T0", "T1", "T2"} <= set(raw.annotations.description)


def test_load_subject_concatenates_runs(fake_eegbci, tmp_path):
    one = eio.load_run(2, 4, tmp_path)
    both = eio.load_subject(2, [4, 8], tmp_path)
    assert both.n_times == 2 * one.n_times
    assert both.info["subject_info"]["his_id"] == "sub-002"


def test_download_subjects_returns_paths(fake_eegbci, tmp_path):
    files = eio.download_subjects([1, 2], [4, 8], tmp_path)
    assert set(files) == {(1, 4), (1, 8), (2, 4), (2, 8)}
    assert all(p.exists() for p in files.values())


def test_to_bids_converts_deidentifies_and_skips_existing(fake_eegbci, tmp_path, monkeypatch):
    monkeypatch.setenv("EEGKIT_DEID_SALT", "s")
    bids = tmp_path / "bids"
    paths = to_bids([1], [4, 8], tmp_path, bids)
    assert len(paths) == 2 and bids_runs_available(1, [4, 8], bids)
    before = paths[0].fpath.stat().st_mtime_ns
    to_bids([1], [4, 8], tmp_path, bids)  # existing runs are kept
    assert paths[0].fpath.stat().st_mtime_ns == before
    to_bids([1], [4], tmp_path, bids, overwrite=True)
    raw = load_subject_bids(1, [4, 8], bids)
    assert raw.n_times > 0 and {"left", "right", "rest"} <= set(raw.annotations.description)
    # the source's 2009 measurement date was shifted back by 1-3 years
    shift = dt.datetime(2009, 8, 12, 16, 15, tzinfo=dt.timezone.utc) - raw.info["meas_date"]
    assert 365 <= shift.days <= 3 * 365
    assert scan_for_identifiers(raw, ["2009-08-12"]) == []


def test_demo_salt_warning_when_unset(monkeypatch, caplog):
    from eegkit.deidentify import get_salt

    monkeypatch.delenv("EEGKIT_DEID_SALT", raising=False)
    import logging

    logging.getLogger("eegkit").propagate = True
    with caplog.at_level(logging.WARNING, logger="eegkit"):
        assert "demo" in get_salt()
    assert "demo" in caplog.text


# ----------------------------------------------------------------------------- CLI


@pytest.fixture
def cfg_file(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        f"""
subjects: [1]
runs: [4, 8]
paths:
  raw_dir: {tmp_path / "raw"}
  bids_root: {tmp_path / "data/bids"}
  output_dir: {tmp_path / "out"}
  cache_dir: {tmp_path / "out/cache"}
  catalog_db: {tmp_path / "out/catalog.sqlite"}
  storage_root: {tmp_path / "data"}
analysis:
  decoding: {{n_splits: 3}}
"""
    )
    return f


def test_cli_end_to_end_download_bids_run_index(fake_eegbci, cfg_file, tmp_path, monkeypatch):
    monkeypatch.setenv("EEGKIT_DEID_SALT", "s")
    c = ["-c", str(cfg_file)]
    r = runner.invoke(app, ["download", *c])
    assert r.exit_code == 0 and "2 files available" in r.output
    r = runner.invoke(app, ["to-bids", *c])
    assert r.exit_code == 0 and "2 runs" in r.output
    r = runner.invoke(app, ["run", *c])
    assert r.exit_code == 0, r.output
    assert "sub-001  ok" in r.output and "Group: n=1" in r.output
    r = runner.invoke(app, ["run", *c])  # second time: cached
    assert "cached: epochs,analysis" in r.output
    r = runner.invoke(app, ["index", *c])
    assert r.exit_code == 0 and "Indexed 2 runs" in r.output


def test_cli_run_reports_failure_exit_code(cfg_file, monkeypatch):
    # nothing downloaded/converted -> the subject fails, exit code 1
    r = runner.invoke(app, ["run", "-c", str(cfg_file)])
    assert r.exit_code == 1 and "failed" in r.output


def test_cli_report_failure_message(cfg_file):
    r = runner.invoke(app, ["report", "-s", "1", "-c", str(cfg_file)])
    assert r.exit_code == 1 and "failed" in r.output


def test_cli_subjects_override_and_invalid(cfg_file):
    r = runner.invoke(app, ["download", "-c", str(cfg_file), "-s", "0"])
    assert r.exit_code == 2 and "Invalid configuration" in r.output


def test_cli_deidentify_command(tmp_path):
    import datetime as dt

    from tests.conftest import make_synthetic_raw

    raw = make_synthetic_raw()
    raw.info["subject_info"] = {"his_id": "MRN1", "last_name": "Doe", "first_name": "Jane"}
    raw.set_meas_date(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
    src, dest = tmp_path / "in_raw.fif", tmp_path / "out_raw.fif"
    raw.save(src, verbose="ERROR")
    r = runner.invoke(app, ["deidentify", str(src), str(dest), "--label", "sub-009"])
    assert r.exit_code == 0, r.output
    out = mne.io.read_raw_fif(dest, verbose="ERROR")
    assert scan_for_identifiers(out, ["Jane", "Doe", "MRN1", "2020-01-01"]) == []


def test_cli_serve_and_app_launch(cfg_file, monkeypatch):
    import subprocess

    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda a, **kw: seen.update(app=a, **kw))
    r = runner.invoke(app, ["serve", "-c", str(cfg_file), "--port", "9001"])
    assert r.exit_code == 0 and seen["port"] == 9001 and seen["app"].title == "eegkit API"

    monkeypatch.setattr(subprocess, "call", lambda cmd, env: seen.update(cmd=cmd, env=env) or 0)
    r = runner.invoke(app, ["app", "-c", str(cfg_file), "--headless", "--port", "8123"])
    assert r.exit_code == 0 and "streamlit" in seen["cmd"] and "8123" in seen["cmd"]
    assert seen["env"]["EEGKIT_CONFIG"] == str(cfg_file)


def test_structured_logging_formats(capsys):
    import json

    from eegkit.logging import configure_logging, get_logger

    configure_logging("INFO")
    get_logger("x").info("hello", extra={"subject": 3})
    assert "hello subject=3" in capsys.readouterr().err
    configure_logging("INFO", as_json=True)
    get_logger("x").info("hello", extra={"subject": 3})
    rec = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert rec["event"] == "hello" and rec["subject"] == 3 and rec["level"] == "INFO"
