"""Left-vs-right imagery decoding: CSP + LDA with stratified cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


@dataclass
class DecodingResult:
    accuracy: float
    std: float
    fold_scores: list[float]
    n_epochs: int
    chance: float = 0.5


def decode_csp_lda(
    epochs: mne.Epochs,
    tmin: float = 0.5,
    tmax: float = 2.5,
    fmin: float = 8.0,
    fmax: float = 30.0,
    channels: list[str] | None = None,
    n_components: int = 4,
    n_splits: int = 5,
    seed: int = 42,
) -> DecodingResult:
    """Cross-validated accuracy of CSP(log-variance) + LDA on a band/time window.

    Band-pass filtering is stateless across epochs, so it is applied once before CV; CSP and
    LDA are fitted inside each training fold only (no leakage).
    """
    labels = epochs.events[:, 2]
    classes, counts_arr = np.unique(labels, return_counts=True)
    counts = [int(c) for c in counts_arr]
    if len(classes) < 2:
        raise ValueError("decoding needs at least two classes")
    if min(counts) < n_splits:
        raise ValueError(f"smallest class has {min(counts)} epochs, fewer than n_splits={n_splits}")
    ep = epochs.copy()
    if channels:
        ep.pick([c for c in channels if c in ep.ch_names])
    ep.filter(fmin, fmax, method="iir", verbose="ERROR")
    ep.crop(tmin, tmax)
    x = ep.get_data()
    clf = make_pipeline(
        CSP(n_components=n_components, reg="ledoit_wolf", log=True, cov_est="concat"),
        LinearDiscriminantAnalysis(),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with mne.use_log_level("ERROR"):
        scores = cross_val_score(clf, x, labels, cv=cv)
    return DecodingResult(
        accuracy=float(scores.mean()),
        std=float(scores.std()),
        fold_scores=[float(s) for s in scores],
        n_epochs=len(labels),
        chance=float(max(counts) / sum(counts)),
    )
