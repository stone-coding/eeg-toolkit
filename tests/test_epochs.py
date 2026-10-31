import numpy as np
import pytest

from eegkit.epochs import EVENT_ID, extract_events, make_epochs, reject_epochs


def test_event_extraction_ignores_rest(synthetic_raw):
    events, event_id = extract_events(synthetic_raw)
    assert event_id == EVENT_ID
    assert len(events) == 24 and set(events[:, 2]) == {1, 2}


def test_epoch_counts_and_window(synthetic_epochs):
    assert len(synthetic_epochs["left"]) == 12
    assert len(synthetic_epochs["right"]) == 12
    assert synthetic_epochs.tmin == pytest.approx(-1.5)
    assert synthetic_epochs.tmax == pytest.approx(4.5, abs=0.01)


def test_no_events_raises(synthetic_raw):
    raw = synthetic_raw.copy()
    raw.set_annotations(None)
    with pytest.raises(ValueError):
        extract_events(raw)


def test_rejection_drops_artifact_epoch_and_reports(synthetic_raw):
    epochs = make_epochs(synthetic_raw)
    epochs._data[3, 2, 100] += 2e-3  # 2 mV spike on C3
    kept, rep = reject_epochs(epochs, reject_uv=300)
    assert rep.n_total == 24 and rep.n_kept == 23 and rep.n_dropped == 1
    assert rep.dropped_indices == [3]
    assert sum(rep.dropped_by_class.values()) == 1
    assert len(kept) == 23 and len(epochs) == 24  # original untouched
    assert "dropped 1" in str(rep)


def test_rejection_channel_subset_ignores_other_channels(synthetic_raw):
    epochs = make_epochs(synthetic_raw)
    epochs._data[0, epochs.ch_names.index("Fz"), 50] += 2e-3
    _, rep_all = reject_epochs(epochs, 300)
    _, rep_roi = reject_epochs(epochs, 300, channels=["C3", "C4"])
    assert rep_all.n_dropped == 1 and rep_roi.n_dropped == 0


def test_rejection_requires_present_channels(synthetic_epochs):
    with pytest.raises(ValueError):
        reject_epochs(synthetic_epochs, 300, channels=["NOPE"])


def test_clean_data_not_rejected(synthetic_epochs):
    _, rep = reject_epochs(synthetic_epochs, 300)
    assert rep.n_dropped == 0 and np.isclose(rep.n_kept, 24)
