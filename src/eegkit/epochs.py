"""Event extraction, epoching and amplitude-based epoch rejection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mne
import numpy as np

from eegkit.logging import get_logger

log = get_logger(__name__)

# EEGBCI runs 4/8/12: T0 = rest, T1 = left fist imagery, T2 = right fist imagery
# BIDS-converted data uses the descriptive names instead of the raw T-codes.
ANNOTATION_MAP = {"T1": 1, "T2": 2, "left": 1, "right": 2}
EVENT_ID = {"left": 1, "right": 2}


def extract_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    """Return ``(events, event_id)`` for left (T1) / right (T2) imagery cues."""
    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP, verbose="ERROR")
    if len(events) == 0:
        raise ValueError("no T1/T2 events found in recording")
    present = {k: v for k, v in EVENT_ID.items() if v in events[:, 2]}
    return events, present


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = -1.5,
    tmax: float = 4.5,
    baseline: tuple[float, float] | None = (-1.5, 0.0),
) -> mne.Epochs:
    """Epoch EEG channels around the cues (no rejection; see :func:`reject_epochs`)."""
    events, event_id = extract_events(raw)
    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )


@dataclass
class RejectionReport:
    n_total: int
    n_kept: int
    threshold_uv: float
    dropped_indices: list[int] = field(default_factory=list)
    dropped_by_class: dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_total - self.n_kept

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_dropped"] = self.n_dropped
        return d

    def __str__(self) -> str:
        return (
            f"kept {self.n_kept}/{self.n_total} epochs "
            f"(dropped {self.n_dropped} above {self.threshold_uv:g} µV peak-to-peak)"
        )


def reject_epochs(
    epochs: mne.Epochs, reject_uv: float = 300.0, channels: list[str] | None = None
) -> tuple[mne.Epochs, RejectionReport]:
    """Drop epochs whose peak-to-peak amplitude exceeds ``reject_uv`` (µV).

    The criterion is evaluated on ``channels`` (default: all good EEG channels); channels
    missing from the data are ignored.
    """
    if channels is not None:
        channels = [c for c in channels if c in epochs.ch_names]
        if not channels:
            raise ValueError("none of the rejection channels are present in the epochs")
    data = epochs.get_data(picks=channels if channels is not None else "eeg")
    ptp_uv = np.ptp(data, axis=2).max(axis=1) * 1e6
    bad = ptp_uv > reject_uv
    inv = {v: k for k, v in epochs.event_id.items()}
    by_class: dict[str, int] = {}
    for code in epochs.events[bad, 2]:
        by_class[inv[int(code)]] = by_class.get(inv[int(code)], 0) + 1
    report = RejectionReport(
        n_total=len(epochs),
        n_kept=int((~bad).sum()),
        threshold_uv=reject_uv,
        dropped_indices=[int(i) for i in np.flatnonzero(bad)],
        dropped_by_class=by_class,
    )
    log.info("epoch rejection", extra=report.as_dict())
    return epochs[~bad], report
