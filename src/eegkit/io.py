"""Loading EEGBCI motor-imagery recordings with standardized channel names and montage."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mne
from mne.datasets import eegbci

from eegkit.logging import get_logger

log = get_logger(__name__)

# MNE >= 1.13 deprecates the name "standard_1005" (same coordinates as "colin27_1005").
MONTAGE = (
    "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
)


def download_subjects(
    subjects: Iterable[int], runs: Iterable[int], data_dir: str | Path
) -> dict[tuple[int, int], Path]:
    """Download (or find cached) EDF files; return ``{(subject, run): path}``."""
    out: dict[tuple[int, int], Path] = {}
    runs = list(runs)
    for sub in subjects:
        paths = eegbci.load_data(sub, runs, path=str(data_dir), update_path=False, verbose="ERROR")
        for run, p in zip(runs, paths, strict=True):
            out[(sub, run)] = Path(p)
        log.info("subject available", extra={"subject": sub, "runs": runs})
    return out


def standardize_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Clean EEGBCI channel names (``Fc5.`` -> ``FC5``) and apply the standard_1005 montage.

    Works in place and returns ``raw`` for chaining.
    """
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage(MONTAGE), on_missing="warn")
    return raw


def load_run(subject: int, run: int, data_dir: str | Path) -> mne.io.BaseRaw:
    """Load one EEGBCI run into memory with standardized names and montage."""
    (path,) = eegbci.load_data(
        subject, [run], path=str(data_dir), update_path=False, verbose="ERROR"
    )
    return read_edf(path)


def read_edf(path: str | Path) -> mne.io.BaseRaw:
    """Read an EEGBCI-style EDF file from disk and standardize it."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return standardize_raw(raw)


def load_subject(subject: int, runs: Iterable[int], data_dir: str | Path) -> mne.io.BaseRaw:
    """Load and concatenate several runs of one subject."""
    raws = [load_run(subject, r, data_dir) for r in runs]
    raw = mne.concatenate_raws(raws, verbose="ERROR") if len(raws) > 1 else raws[0]
    raw.info["subject_info"] = {"his_id": f"sub-{subject:03d}", "id": subject}
    log.info(
        "loaded subject",
        extra={"subject": subject, "n_channels": len(raw.ch_names), "sfreq": raw.info["sfreq"]},
    )
    return raw
