"""Event-related (de)synchronization: time-frequency power as percent change vs. baseline."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class ErdersResult:
    """ERD/ERS in percent: negative = desynchronization (ERD), positive = ERS."""

    condition: str
    ch_names: list[str]
    freqs: np.ndarray
    times: np.ndarray
    data: np.ndarray  # (n_ch, n_freq, n_times)


def compute_erders(
    epochs: mne.Epochs,
    channels: list[str] | tuple[str, ...] = ("C3", "C4"),
    freqs: tuple[float, float] = (6.0, 30.0),
    baseline: tuple[float, float] = (-1.0, 0.0),
    crop: tuple[float, float] | None = (-1.0, 4.0),
) -> dict[str, ErdersResult]:
    """Morlet time-frequency ERD/ERS per condition (Pfurtscheller: ``(P - R) / R * 100``).

    Power is averaged over epochs first, then expressed relative to the mean power in the
    ``baseline`` window at each frequency. Edge effects are avoided by computing on the full
    epoch and cropping afterwards.
    """
    picks = [c for c in channels if c in epochs.ch_names]
    if not picks:
        raise ValueError(f"none of {list(channels)} found in epochs")
    f = np.arange(freqs[0], freqs[1] + 1.0, 1.0)
    out: dict[str, ErdersResult] = {}
    for cond in epochs.event_id:
        sub = epochs[cond]
        if len(sub) == 0:
            continue
        tfr = sub.compute_tfr(
            "morlet",
            f,
            n_cycles=f / 2.0,
            picks=picks,
            average=True,
            return_itc=False,
            decim=2,
            verbose="ERROR",
        )
        power, times = tfr.data, tfr.times
        bmask = (times >= baseline[0]) & (times <= baseline[1])
        if not bmask.any():
            raise ValueError("baseline window outside the epoch time range")
        ref = power[..., bmask].mean(axis=-1, keepdims=True)
        erd = (power - ref) / ref * 100.0
        if crop is not None:
            m = (times >= crop[0]) & (times <= crop[1])
            erd, times = erd[..., m], times[m]
        out[cond] = ErdersResult(cond, list(tfr.ch_names), f, times, erd)
    return out


def band_erders(
    res: ErdersResult, band: tuple[float, float], window: tuple[float, float] = (0.5, 3.5)
) -> dict[str, float]:
    """Mean ERD/ERS (%) per channel within a band and time window."""
    fm = (res.freqs >= band[0]) & (res.freqs <= band[1])
    tm = (res.times >= window[0]) & (res.times <= window[1])
    vals = res.data[:, fm][:, :, tm].mean(axis=(1, 2))
    return dict(zip(res.ch_names, (float(v) for v in vals), strict=True))
