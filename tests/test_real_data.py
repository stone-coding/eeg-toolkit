"""End-to-end checks on the real EEGBCI data. Slow; need the download (``--runslow``)."""

from pathlib import Path

import pytest

from eegkit import pipeline
from eegkit.config import Config

RAW = Path("data/raw")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RAW.exists(), reason="run `eegkit download` first"),
]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    return Config.model_validate(
        {
            "subjects": [1, 2],
            "paths": {
                "raw_dir": str(RAW),
                "bids_root": str(out / "bids"),
                "output_dir": str(out / "out"),
                "cache_dir": str(out / "out/cache"),
            },
        }
    )


def test_real_subject_loads_standardized():
    from eegkit.io import load_run

    raw = load_run(1, 4, RAW)
    assert raw.info["sfreq"] == 160.0 and len(raw.ch_names) == 64
    assert "C3" in raw.ch_names and raw.get_montage() is not None


def test_pipeline_on_real_subjects(cfg):
    results, summary = pipeline.run_batch(cfg)
    assert all(r.status == "ok" for r in results)
    assert results[0].n_epochs >= 40
    assert results[0].accuracy > 0.6 and results[1].accuracy > 0.6  # known-good subjects
    # contralateral mu ERD: right-hand imagery desynchronizes C3 more than C4
    erd = summary["erders_mean_pct"]
    assert erd["right/C3/mu"] < erd["right/C4/mu"]
