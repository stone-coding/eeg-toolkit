"""Preprocessing: filtering, bad-channel detection, re-referencing. All functions copy."""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np

from eegkit.config import PreprocessConfig
from eegkit.logging import get_logger

log = get_logger(__name__)

MIN_LOG_VAR_SPREAD = 0.1  # log10(variance) units; see detect_bad_channels


def filter_raw(
    raw: mne.io.BaseRaw,
    l_freq: float | None = 1.0,
    h_freq: float | None = 40.0,
    notch_freqs: list[float] | None = None,
) -> mne.io.BaseRaw:
    """Band-pass (+ optional notch) filter. Notches at/above Nyquist are skipped."""
    out = raw.copy()
    out.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    nyq = out.info["sfreq"] / 2
    notch = [f for f in (notch_freqs or []) if f < nyq]
    if notch:
        out.notch_filter(notch, fir_design="firwin", verbose="ERROR")
    return out


@dataclass
class BadChannelReport:
    flat: list[str] = field(default_factory=list)
    noisy: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return sorted(set(self.flat) | set(self.noisy))


def detect_bad_channels(
    raw: mne.io.BaseRaw, flat_std_v: float = 1e-7, noisy_z: float = 4.0
) -> BadChannelReport:
    """Flag flat channels (std below threshold) and high-variance channels.

    Noisy = robust z-score (median/MAD) of log10 variance across channels above ``noisy_z``.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    names = [raw.ch_names[i] for i in picks]
    std = data.std(axis=1)
    flat = std < flat_std_v
    report = BadChannelReport(flat=[n for n, f in zip(names, flat, strict=True) if f])
    live = ~flat
    if live.sum() >= 3:
        logv = np.full(len(std), np.nan)
        logv[live] = np.log10(std[live] ** 2)
        med = np.nanmedian(logv)
        # Floor the MAD so near-identical channels don't make trivial differences "outliers".
        mad = max(np.nanmedian(np.abs(logv - med)) * 1.4826, MIN_LOG_VAR_SPREAD)
        z = (logv - med) / mad
        noisy = live & (z > noisy_z)
        report.noisy = [n for n, f in zip(names, noisy, strict=True) if f]
    return report


def mark_bad_channels(raw: mne.io.BaseRaw, bads: list[str]) -> mne.io.BaseRaw:
    """Return a copy with ``info['bads']`` extended by ``bads``."""
    out = raw.copy()
    out.info["bads"] = sorted(set(out.info["bads"]) | set(bads))
    return out


def set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a copy re-referenced to the average of good EEG channels."""
    return raw.copy().set_eeg_reference("average", projection=False, verbose="ERROR")


def preprocess(
    raw: mne.io.BaseRaw, cfg: PreprocessConfig | None = None
) -> tuple[mne.io.BaseRaw, BadChannelReport]:
    """Filter -> detect/mark bads -> average reference -> (optionally) interpolate bads."""
    cfg = cfg or PreprocessConfig()
    out = filter_raw(raw, cfg.l_freq, cfg.h_freq, cfg.notch_freqs)
    report = detect_bad_channels(out, cfg.flat_std_v, cfg.noisy_z)
    out = mark_bad_channels(out, report.all)
    out = set_average_reference(out)
    if report.all and cfg.interpolate_bads:
        out.interpolate_bads(reset_bads=True, verbose="ERROR")
    log.info("preprocessed", extra={"flat": report.flat, "noisy": report.noisy})
    return out, report
