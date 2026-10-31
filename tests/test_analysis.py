import mne
import numpy as np
import pytest

from eegkit.analysis import (
    band_erders,
    band_power,
    band_power_table,
    compute_erders,
    compute_psd_by_condition,
    decode_csp_lda,
)
from eegkit.analysis.spectral import _psd


def _sine_epochs(freq: float, amp_uv: float = 10.0):
    sf, n = 160.0, 160 * 4
    t = np.arange(n) / sf
    data = np.tile(amp_uv * 1e-6 * np.sin(2 * np.pi * freq * t), (6, 2, 1))
    info = mne.create_info(["C3", "C4"], sf, "eeg")
    events = np.array([[i * n, 0, 1] for i in range(6)])
    return mne.EpochsArray(data, info, events, tmin=0, event_id={"left": 1}, verbose="ERROR")


def test_band_power_of_known_sine():
    """A 10 Hz sine of amplitude A has power A²/2, all inside the mu band."""
    ep = _sine_epochs(10.0, 10.0)
    psd, freqs = _psd(ep, 1, 40)
    mu = band_power(psd, freqs, (8, 13))
    beta = band_power(psd, freqs, (13, 30))
    expected = (10e-6) ** 2 / 2
    assert mu[0] == pytest.approx(expected, rel=0.1)
    assert beta[0] < 0.02 * mu[0]
    assert band_power(psd, freqs, (8, 13), relative=True)[0] > 0.9


def test_band_power_needs_two_points():
    with pytest.raises(ValueError):
        band_power(np.ones((1, 3)), np.array([1.0, 2.0, 3.0]), (10, 11))


def test_band_power_table_shapes(synthetic_epochs):
    df = band_power_table(synthetic_epochs, {"mu": (8, 13), "beta": (13, 30)})
    assert set(df.columns) == {"condition", "channel", "band", "power"}
    assert len(df) == 2 * 10 * 2
    assert (df.power > 0).all()


def test_mu_power_lower_over_contralateral_cortex(synthetic_epochs):
    df = band_power_table(synthetic_epochs, {"mu": (8, 13)})
    p = df.pivot(index="channel", columns="condition", values="power")
    assert p.loc["C3", "right"] < 0.3 * p.loc["C3", "left"]
    assert p.loc["C4", "left"] < 0.3 * p.loc["C4", "right"]


def test_psd_by_condition_keys(synthetic_epochs):
    psds, freqs, names = compute_psd_by_condition(synthetic_epochs)
    assert set(psds) == {"left", "right"} and psds["left"].shape == (10, len(freqs))
    assert names == synthetic_epochs.ch_names


def test_erders_sign_and_laterality(synthetic_epochs):
    res = compute_erders(synthetic_epochs)
    mu = (8, 13)
    right_c3 = band_erders(res["right"], mu)["C3"]
    right_c4 = band_erders(res["right"], mu)["C4"]
    left_c4 = band_erders(res["left"], mu)["C4"]
    assert right_c3 < -50  # strong ERD over contralateral C3
    assert left_c4 < -50
    assert abs(right_c4) < 20  # ipsilateral ~ baseline
    r = res["right"]
    assert r.data.shape == (2, 25, len(r.times)) and r.times[0] >= -1.0 - 1e-9


def test_erders_validation(synthetic_epochs):
    with pytest.raises(ValueError):
        compute_erders(synthetic_epochs, channels=["XX"])
    with pytest.raises(ValueError):
        compute_erders(synthetic_epochs, baseline=(-9, -8))


def test_decoding_on_separable_synthetic(synthetic_epochs):
    res = decode_csp_lda(synthetic_epochs, n_splits=4)
    assert res.accuracy > 0.85 and len(res.fold_scores) == 4 and res.n_epochs == 24


def test_decoding_guards(synthetic_epochs):
    with pytest.raises(ValueError):
        decode_csp_lda(synthetic_epochs["left"])
    with pytest.raises(ValueError):
        decode_csp_lda(synthetic_epochs, n_splits=20)
