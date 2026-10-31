import numpy as np

from eegkit.preprocess import (
    detect_bad_channels,
    filter_raw,
    mark_bad_channels,
    preprocess,
    set_average_reference,
)
from tests.conftest import make_synthetic_raw


def _power_at(raw, ch, f):
    x = raw.get_data(picks=ch)[0]
    spec = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / raw.info["sfreq"])
    return spec[np.argmin(abs(freqs - f))]


def test_filter_attenuates_line_noise_and_drift_and_copies(synthetic_raw):
    raw = synthetic_raw.copy()
    t = raw.times
    raw._data += 30e-6 * np.sin(2 * np.pi * 60 * t) + 50e-6 * np.sin(2 * np.pi * 0.1 * t)
    before = raw.get_data().copy()
    out = filter_raw(raw, 1.0, 40.0, [60.0])
    assert out is not raw
    np.testing.assert_array_equal(raw.get_data(), before)  # input untouched
    assert _power_at(out, "C3", 60) < 0.05 * _power_at(raw, "C3", 60)
    assert _power_at(out, "C3", 0.1) < 0.05 * _power_at(raw, "C3", 0.1)
    assert _power_at(out, "Cz", 10) > 0.8 * _power_at(raw, "Cz", 10)  # mu preserved


def test_notch_above_nyquist_is_skipped(synthetic_raw):
    filter_raw(synthetic_raw, 1.0, 40.0, [200.0])  # must not raise


def test_detect_flat_and_noisy():
    raw = make_synthetic_raw()
    raw._data[raw.ch_names.index("Fz")] = 0.0
    raw._data[raw.ch_names.index("Oz")] *= 40
    rep = detect_bad_channels(raw, flat_std_v=1e-7, noisy_z=4)
    assert rep.flat == ["Fz"]
    assert rep.noisy == ["Oz"]
    assert rep.all == ["Fz", "Oz"]


def test_mark_bads_returns_copy(synthetic_raw):
    out = mark_bad_channels(synthetic_raw, ["Pz"])
    assert out.info["bads"] == ["Pz"] and synthetic_raw.info["bads"] == []


def test_average_reference_zero_mean(synthetic_raw):
    out = set_average_reference(synthetic_raw)
    assert np.allclose(out.get_data().mean(axis=0), 0, atol=1e-12)


def test_preprocess_pipeline_interpolates():
    raw = make_synthetic_raw()
    raw._data[raw.ch_names.index("Oz")] *= 40
    out, rep = preprocess(raw)
    assert rep.noisy == ["Oz"]
    assert out.info["bads"] == []  # interpolated and reset
    assert out.get_data(picks="Oz").std() < raw.get_data(picks="Oz").std()
