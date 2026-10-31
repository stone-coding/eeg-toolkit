import mne
import numpy as np
import pandas as pd

from eegkit import viz
from eegkit.analysis import band_power_table, compute_erders, compute_psd_by_condition
from eegkit.io import MONTAGE, standardize_raw


def test_standardize_cleans_names_and_sets_montage():
    names = ["Fc5.", "C3..", "Cz..", "Fpz."]
    raw = mne.io.RawArray(np.zeros((4, 100)), mne.create_info(names, 160.0, "eeg"))
    standardize_raw(raw)
    assert raw.ch_names == ["FC5", "C3", "Cz", "Fpz"]
    assert not np.isnan(raw.info["chs"][1]["loc"][:3]).any()
    assert MONTAGE in {"colin27_1005", "standard_1005"}


def test_figures_build(synthetic_epochs):
    psds, f, n = compute_psd_by_condition(synthetic_epochs)
    figs = [
        viz.plot_psd_comparison(psds, f, n),
        viz.plot_erders(compute_erders(synthetic_epochs)),
        viz.plot_band_topomaps(
            band_power_table(synthetic_epochs, {"mu": (8, 13), "beta": (13, 30)}),
            synthetic_epochs.info,
        ),
        viz.plot_decoding_accuracy(
            pd.DataFrame({"subject": [1, 2], "accuracy": [0.7, 0.9], "std": [0.1, 0.05]})
        ),
    ]
    assert all(len(fg.axes) > 0 for fg in figs)
