"""BIDS conversion (mne-bids) and reading of the converted dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

from eegkit.io import load_run
from eegkit.logging import get_logger

log = get_logger(__name__)

TASK = "motorimagery"
SESSION = "01"
EVENT_ID = {"rest": 1, "left": 2, "right": 3}
RENAME = {"T0": "rest", "T1": "left", "T2": "right"}


def bids_path(subject: int, run: int, root: str | Path) -> BIDSPath:
    """BIDSPath for one EEGBCI run (run label = the original EEGBCI run number)."""
    return BIDSPath(
        subject=f"{subject:03d}",
        session=SESSION,
        task=TASK,
        run=f"{run:02d}",
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
        root=Path(root),
    )


def to_bids(
    subjects: Iterable[int],
    runs: Iterable[int],
    raw_dir: str | Path,
    bids_root: str | Path,
    overwrite: bool = False,
) -> list[BIDSPath]:
    """Convert downloaded EEGBCI EDFs to a BIDS dataset. Existing files are kept unless
    ``overwrite``. Returns the BIDS paths of all runs."""
    out: list[BIDSPath] = []
    for sub in subjects:
        for run in runs:
            bp = bids_path(sub, run, bids_root)
            out.append(bp)
            if bp.fpath.exists() and not overwrite:
                log.info("bids run exists, skipping", extra={"subject": sub, "run": run})
                continue
            raw = load_run(sub, run, raw_dir)
            raw.annotations.rename(RENAME)
            raw.info["line_freq"] = 60  # recorded in the US
            write_raw_bids(
                raw,
                bp,
                event_id=EVENT_ID,
                format="EDF",
                allow_preload=True,
                overwrite=True,
                verbose="ERROR",
            )
            log.info("wrote bids run", extra={"subject": sub, "run": run, "path": str(bp.fpath)})
    return out


def write_run_bids(raw: mne.io.BaseRaw, bp: BIDSPath) -> None:
    """Write one recording (T0/T1/T2 or rest/left/right annotations) as BIDS EDF."""
    raw = raw.copy()
    raw.annotations.rename(
        {k: v for k, v in RENAME.items() if k in set(raw.annotations.description)}
    )
    raw.info["line_freq"] = 60  # recorded in the US
    write_raw_bids(
        raw,
        bp,
        event_id=EVENT_ID,
        format="EDF",
        allow_preload=True,
        overwrite=True,
        verbose="ERROR",
    )


def bids_runs_available(subject: int, runs: Iterable[int], root: str | Path) -> bool:
    return all(bids_path(subject, r, root).fpath.exists() for r in runs)


def load_subject_bids(subject: int, runs: Iterable[int], root: str | Path) -> mne.io.BaseRaw:
    """Read and concatenate a subject's BIDS runs (preloaded)."""
    raws = []
    for r in runs:
        raw = read_raw_bids(bids_path(subject, r, root), verbose="ERROR")
        raw.load_data(verbose="ERROR")
        raws.append(raw)
    return mne.concatenate_raws(raws, verbose="ERROR") if len(raws) > 1 else raws[0]
