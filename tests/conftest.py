"""Shared synthetic-EEG fixtures (no downloads): mu rhythm with contralateral ERD."""

from __future__ import annotations

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
