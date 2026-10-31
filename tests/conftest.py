"""Shared synthetic-EEG fixtures (no downloads): mu rhythm with contralateral ERD."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")
mne.set_log_level("ERROR")

SFREQ = 160.0
CH_NAMES = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Fz", "Pz", "Oz"]


def make_synthetic_raw(
    n_per_class: int = 12, seed: int = 0, mu_amp_uv: float = 20.0, noise_uv: float = 4.0
) -> mne.io.RawArray:
    """EEGBCI-like run: T0 rest (4.2 s) alternating with T1/T2 cues (4.1 s).

    A 10 Hz mu rhythm is present everywhere; during imagery its amplitude drops to 20% over the
    contralateral sensorimotor cortex (T1/left -> C4/FC4/CP4, T2/right -> C3/FC3/CP3).
    """
    rng = np.random.default_rng(seed)
    labels = ["T1"] * n_per_class + ["T2"] * n_per_class
    rng.shuffle(labels)
    rest, task = 4.2, 4.1
    t_total = rest + len(labels) * (task + rest)
    n = int(t_total * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.normal(0, noise_uv, (len(CH_NAMES), n))
    env = np.ones((len(CH_NAMES), n))
    onsets, descs, durs = [0.0], ["T0"], [rest]
    cur = rest
    right_hemi = [CH_NAMES.index(c) for c in ("FC4", "C4", "CP4")]
    left_hemi = [CH_NAMES.index(c) for c in ("FC3", "C3", "CP3")]
    for lab in labels:
        s, e = int(cur * SFREQ), int((cur + task) * SFREQ)
        for ch in right_hemi if lab == "T1" else left_hemi:
            env[ch, s:e] = 0.2
        onsets += [cur, cur + task]
        descs += [lab, "T0"]
        durs += [task, rest]
        cur += task + rest
    # independent phase per channel (a phase-locked sine would vanish under average reference)
    phase = rng.uniform(0, 2 * np.pi, (len(CH_NAMES), 1))
    data += mu_amp_uv * env * np.sin(2 * np.pi * 10 * t + phase)
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onsets, durs, descs))
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw


@pytest.fixture(scope="session")
def synthetic_raw() -> mne.io.RawArray:
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def synthetic_epochs(synthetic_raw):
    from eegkit.epochs import make_epochs

    return make_epochs(synthetic_raw)
