"""Shared synthetic-EEG fixtures (no downloads): mu rhythm with contralateral ERD."""

from __future__ import annotations

import datetime as dt

import matplotlib
import mne
import pytest

from eegkit.synthetic import CH_NAMES, SFREQ, make_synthetic_raw  # noqa: F401

matplotlib.use("Agg")
mne.set_log_level("ERROR")


@pytest.fixture(scope="session")
def synthetic_raw() -> mne.io.RawArray:
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def synthetic_epochs(synthetic_raw):
    from eegkit.epochs import make_epochs

    return make_epochs(synthetic_raw)


@pytest.fixture
def lab_env(tmp_path, monkeypatch):
    """A tiny, fully processed lab (synthetic BIDS + pipeline outputs + catalog) on disk,
    selected via $EEGKIT_CONFIG - for GUI / end-to-end tests."""
    from eegkit import pipeline
    from eegkit.bids import bids_path, write_run_bids
    from eegkit.catalog import Catalog
    from eegkit.config import Config
    from eegkit.storage import LocalStorage

    cfg = Config.model_validate(
        {
            "subjects": [1],
            "runs": [4, 8],
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "bids_root": str(tmp_path / "data/bids"),
                "output_dir": str(tmp_path / "out"),
                "cache_dir": str(tmp_path / "out/cache"),
                "catalog_db": str(tmp_path / "out/catalog.sqlite"),
                "storage_root": str(tmp_path / "data"),
            },
            "analysis": {"decoding": {"n_splits": 4}},
        }
    )
    for run in (4, 8):
        write_run_bids(make_synthetic_raw(seed=run), bids_path(1, run, cfg.paths.bids_root))
    results, _ = pipeline.run_batch(cfg)
    assert results[0].status == "ok"
    storage = LocalStorage(cfg.paths.storage_root)
    cat = Catalog(cfg.paths.catalog_db)
    cat.index_bids(cfg.paths.bids_root, storage)
    cat.sync_status(cfg.paths.output_dir)
    cat.publish_artifacts(cfg, storage)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg.model_dump_json())
    monkeypatch.setenv("EEGKIT_CONFIG", str(cfg_file))
    return cfg


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run tests that need the real dataset")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs the real EEGBCI download (use --runslow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def _eegbci_name(name: str) -> str:
    """'FC3' -> 'Fc3.', 'C3' -> 'C3..' (the dotted, odd-case labels in the EEGBCI EDFs)."""
    base = name[0] + name[1:].lower()
    return base.ljust(4, ".")


@pytest.fixture
def fake_eegbci(tmp_path, monkeypatch):
    """Synthetic EDFs laid out like the EEGBCI download, and ``eegbci.load_data`` pointed at
    them, so download/convert/load code paths run offline."""
    from mne.export import export_raw

    from eegkit import io as eio

    root = tmp_path / "raw" / "files"
    calls: list[tuple] = []

    def load_data(subjects, runs, path=None, **kwargs):
        calls.append((subjects, tuple(runs)))
        subjects = [subjects] if isinstance(subjects, int) else subjects
        out = []
        for s in subjects:
            for r in runs:
                f = root / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf"
                if not f.exists():
                    f.parent.mkdir(parents=True, exist_ok=True)
                    raw = make_synthetic_raw(n_per_class=3, seed=s * 100 + r)
                    raw.rename_channels({n: _eegbci_name(n) for n in raw.ch_names})
                    raw.set_montage(None)
                    raw.set_meas_date(dt.datetime(2009, 8, 12, 16, 15, tzinfo=dt.timezone.utc))
                    export_raw(f, raw, fmt="edf", overwrite=True, verbose="ERROR")
                out.append(str(f))
        return out

    monkeypatch.setattr(eio.eegbci, "load_data", load_data)
    return calls
