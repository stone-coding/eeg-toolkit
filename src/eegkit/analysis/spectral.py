"""Power spectral density and band power."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def _psd(epochs: mne.Epochs, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Epoch-averaged Welch PSD (V²/Hz) -> ``(psd[n_ch, n_freq], freqs)``."""
    n_fft = min(256, epochs.get_data().shape[-1])
    spec = epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, picks="eeg", verbose="ERROR"
    )
    data, freqs = spec.get_data(return_freqs=True)
    return data.mean(axis=0), freqs


def compute_psd_by_condition(
    epochs: mne.Epochs,
    fmin: float = 2.0,
    fmax: float = 40.0,
    tmin: float | None = 0.5,
    tmax: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Mean PSD per condition over the task window ``[tmin, tmax]``.

    Returns ``({condition: psd[n_ch, n_freq]}, freqs, ch_names)``.
    """
    cropped = epochs.copy().crop(tmin, tmax)
    out: dict[str, np.ndarray] = {}
    freqs = np.array([])
    for cond in epochs.event_id:
        if len(cropped[cond]) == 0:
            continue
        out[cond], freqs = _psd(cropped[cond], fmin, fmax)
    names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
    return out, freqs, names


def band_power(
    psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float], relative: bool = False
) -> np.ndarray:
    """Integrate PSD over ``band`` along the last axis (trapezoid rule).

    With ``relative=True`` the result is divided by the total power over ``freqs``.
    """
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        raise ValueError(f"frequency grid has fewer than 2 points inside band {band}")
    bp = trapezoid(psd[..., mask], freqs[mask], axis=-1)
    if relative:
        bp = bp / trapezoid(psd, freqs, axis=-1)
    return np.asarray(bp)


def band_power_table(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]],
    window: tuple[float, float] = (0.5, 3.5),
) -> pd.DataFrame:
    """Long-format band power: columns ``condition, channel, band, power`` (V²)."""
    fmax = max(hi for _, hi in bands.values()) + 2
    psds, freqs, names = compute_psd_by_condition(
        epochs, fmin=1.0, fmax=fmax, tmin=window[0], tmax=window[1]
    )
    rows = []
    for cond, psd in psds.items():
        for band, rng in bands.items():
            for ch, p in zip(names, band_power(psd, freqs, rng), strict=True):
                rows.append({"condition": cond, "channel": ch, "band": band, "power": float(p)})
    return pd.DataFrame(rows)
