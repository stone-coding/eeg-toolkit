import datetime as dt

import mne
import pytest

from eegkit.bids import bids_path, write_run_bids
from eegkit.deidentify import (
    MAX_SHIFT_DAYS,
    MIN_SHIFT_DAYS,
    age_years,
    date_offset_days,
    deidentify_metadata,
    deidentify_raw,
    pseudonym,
    scan_for_identifiers,
)
from tests.conftest import make_synthetic_raw

SALT = "unit-test-salt"
MEAS = dt.datetime(2020, 3, 1, 10, 30, tzinfo=dt.timezone.utc)
# every string an attacker could use to re-identify the subject
IDENTIFIERS = [
    "Jane",
    "Doe",
    "Quincy",
    "MRN12345",
    "Dr. Smith",
    "SN998",
    "General Hospital",
    "1990",
    "Doe lab study",
    "2020-03-01",
]


@pytest.fixture
def identified_raw():
    raw = make_synthetic_raw()
    raw.info["subject_info"] = {
        "id": 7,
        "his_id": "MRN12345",
        "last_name": "Doe",
        "first_name": "Jane",
        "middle_name": "Quincy",
        "birthday": dt.date(1990, 5, 17),
        "sex": 2,
        "hand": 1,
    }
    raw.set_meas_date(MEAS)
    raw.info["experimenter"] = "Dr. Smith"
    raw.info["description"] = "Jane Doe resting session"
    raw.info["proj_name"] = "Doe lab study"
    raw.info["device_info"] = {
        "type": "amp",
        "model": "x",
        "serial": "SN998",
        "site": "General Hospital",
    }
    raw.set_annotations(raw.annotations)
    return raw


def test_scanner_detects_planted_identifiers(identified_raw):
    hits = scan_for_identifiers(identified_raw, IDENTIFIERS)
    text = " ".join(hits)
    for term in [
        "jane",
        "doe",
        "mrn12345",
        "smith",
        "sn998",
        "general hospital",
        "1990",
        "2020-03-01",
    ]:
        assert term in text.lower(), term


def test_no_identifiers_remain_in_memory(identified_raw):
    out = deidentify_raw(identified_raw, "sub-001", SALT)
    assert scan_for_identifiers(out, IDENTIFIERS) == []
    assert out.info["subject_info"].get("birthday") is None
    assert out.info["experimenter"] is None and out.info["description"] is None
    assert out.info["subject_info"]["his_id"] == pseudonym("sub-001", SALT)
    # research-relevant variables survive
    assert out.info["subject_info"]["sex"] == 2 and out.info["subject_info"]["hand"] == 1


def test_input_not_modified(identified_raw):
    deidentify_raw(identified_raw, "sub-001", SALT)
    assert identified_raw.info["subject_info"]["last_name"] == "Doe"
    assert identified_raw.info["meas_date"] == MEAS


def test_no_identifiers_remain_in_saved_file(identified_raw, tmp_path):
    out = deidentify_raw(identified_raw, "sub-001", SALT)
    path = tmp_path / "x_raw.fif"
    out.save(path, verbose="ERROR")
    blob = path.read_bytes()
    for term in ["Jane", "Doe", "Quincy", "MRN12345", "Smith", "SN998", "General Hospital"]:
        assert term.encode() not in blob, term
    back = mne.io.read_raw_fif(path, verbose="ERROR")
    assert scan_for_identifiers(back, IDENTIFIERS) == []


def test_no_identifiers_remain_in_bids_dataset(identified_raw, tmp_path):
    bp = bids_path(1, 4, tmp_path / "bids")
    write_run_bids(identified_raw, bp, deid_salt=SALT)
    for f in (tmp_path / "bids").rglob("*"):
        if f.is_file():
            blob = f.read_bytes()
            for term in [
                "Jane",
                "Doe",
                "Quincy",
                "MRN12345",
                "Smith",
                "SN998",
                "General Hospital",
                "2020-03-01",
                "1990-05-17",
            ]:
                assert term.encode() not in blob, f"{term} found in {f.name}"


def test_dates_shifted_consistently(identified_raw):
    out = deidentify_raw(identified_raw, "sub-001", SALT)
    shift = MEAS - out.info["meas_date"]
    assert MIN_SHIFT_DAYS <= shift.days <= MAX_SHIFT_DAYS
    assert shift.days == date_offset_days("sub-001", SALT)
    # annotations stay aligned with the shifted measurement date
    assert out.annotations.orig_time == out.info["meas_date"]
    # same subject, same shift (intervals between its runs are preserved)...
    again = deidentify_raw(identified_raw, "sub-001", SALT)
    assert again.info["meas_date"] == out.info["meas_date"]
    # ...a different salt or subject gives a different shift
    assert date_offset_days("sub-001", "other") != date_offset_days("sub-001", SALT)
    assert len({date_offset_days(f"sub-{i:03d}", SALT) for i in range(30)}) > 10


def test_age_capped_and_birthday_free(identified_raw):
    assert age_years(identified_raw) == 29
    identified_raw.info["subject_info"]["birthday"] = dt.date(1900, 1, 1)
    assert age_years(identified_raw) == 89
    identified_raw.info["subject_info"]["birthday"] = None
    assert age_years(identified_raw) is None


def test_metadata_dict_deidentification():
    meta = {
        "participant_id": "sub-001",
        "first_name": "Jane",
        "last_name": "Doe",
        "birthdate": "1990-05-17",
        "email": "jane@example.org",
        "MRN": "12345",
        "acq_time": "2020-03-01T10:30:00",
        "date_recorded": dt.date(2020, 3, 1),
        "sex": "F",
        "sampling_rate": 160.0,
        "nested": {"phone": "555-1234", "scan_date": "2020-03-02T00:00:00", "n": 3},
    }
    out = deidentify_metadata(meta, "sub-001", SALT)
    blob = repr(out)
    for term in ["Jane", "Doe", "1990", "example.org", "12345", "555-1234", "sub-001"]:
        assert term not in blob, term
    assert out["participant_id"] == pseudonym("sub-001", SALT)
    shift = dt.timedelta(days=date_offset_days("sub-001", SALT))
    assert dt.datetime.fromisoformat(out["acq_time"]) == dt.datetime(2020, 3, 1, 10, 30) - shift
    assert out["date_recorded"] == dt.date(2020, 3, 1) - shift
    assert out["sex"] == "F" and out["sampling_rate"] == 160.0 and out["nested"]["n"] == 3
