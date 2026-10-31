"""De-identification of recording metadata (a demonstration, not a HIPAA certification).

Approach (see ``docs/privacy.md``):

* direct identifiers (names, birthday, record numbers, experimenter, device serial/site,
  free-text description) are removed or replaced by a keyed pseudonym;
* measurement dates are shifted by a per-subject offset derived from an HMAC of the subject
  label with a secret salt, so intervals *within* a subject (and between its runs) are preserved
  but absolute dates cannot be recovered without the salt;
* the birthday is dropped; :func:`age_years` gives an integer age at recording (capped at 89);
* :func:`scan_for_identifiers` lets callers (and tests) prove that given terms no longer occur.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import mne

from eegkit.logging import get_logger

log = get_logger(__name__)

SALT_ENV = "EEGKIT_DEID_SALT"
_DEMO_SALT = "eegkit-demo-salt-do-not-use-in-production"
MIN_SHIFT_DAYS, MAX_SHIFT_DAYS = 365, 3 * 365
MAX_AGE = 89  # HIPAA safe-harbor style top-coding

# keys dropped from free-form metadata dicts (matched case-insensitively, substring)
IDENTIFYING_KEYS = (
    "name",
    "birth",
    "dob",
    "address",
    "phone",
    "email",
    "mrn",
    "ssn",
    "his_id",
    "experimenter",
    "operator",
    "serial",
    "site",
    "institution",
    "comment",
    "description",
)
DATE_KEYS = ("date", "acq_time", "meas")


def get_salt() -> str:
    """Secret salt from ``$EEGKIT_DEID_SALT`` (a demo salt, with a warning, if unset)."""
    salt = os.environ.get(SALT_ENV)
    if not salt:
        log.warning("using demo de-identification salt; set %s in production", SALT_ENV)
        return _DEMO_SALT
    return salt


def _hmac(label: str, salt: str, purpose: str) -> bytes:
    return hmac.new(salt.encode(), f"{purpose}:{label}".encode(), hashlib.sha256).digest()


def date_offset_days(subject_label: str, salt: str) -> int:
    """Deterministic per-subject backwards shift in ``[MIN_SHIFT_DAYS, MAX_SHIFT_DAYS]``."""
    span = MAX_SHIFT_DAYS - MIN_SHIFT_DAYS + 1
    return MIN_SHIFT_DAYS + int.from_bytes(_hmac(subject_label, salt, "shift")[:8], "big") % span


def pseudonym(subject_label: str, salt: str) -> str:
    """Stable keyed pseudonym (``P-xxxxxxxx``) that cannot be reversed without the salt."""
    return "P-" + _hmac(subject_label, salt, "id").hex()[:8]


def _age_at(birthday: Any, when: dt.datetime | None) -> int | None:
    if birthday is None or when is None:
        return None
    if isinstance(birthday, tuple):  # raw FIF stores (y, m, d)
        birthday = dt.date(*birthday)
    years = when.year - birthday.year - ((when.month, when.day) < (birthday.month, birthday.day))
    return max(0, min(int(years), MAX_AGE))


def age_years(raw: mne.io.BaseRaw) -> int | None:
    """Age at recording in whole years (capped at 89), or None if unknown.

    Call this *before* :func:`deidentify_raw` (which drops the birthday) if the age is needed,
    e.g. for ``participants.tsv``.
    """
    sinfo = raw.info.get("subject_info") or {}
    return _age_at(sinfo.get("birthday"), raw.info["meas_date"])


def deidentify_raw(
    raw: mne.io.BaseRaw, subject_label: str, salt: str | None = None
) -> mne.io.BaseRaw:
    """Return a de-identified copy of ``raw`` (the input is not modified)."""
    salt = salt or get_salt()
    offset = date_offset_days(subject_label, salt)
    sinfo = raw.info.get("subject_info") or {}
    keep = {k: sinfo[k] for k in ("sex", "hand") if k in sinfo}

    out = raw.copy()
    if out.info["meas_date"] is not None:
        out.anonymize(daysback=offset, keep_his=False, verbose="ERROR")
    code = pseudonym(subject_label, salt)
    out.info["subject_info"] = {"id": 0, "his_id": code, **keep}
    out.info["experimenter"] = None
    out.info["description"] = None
    out.info["proj_name"] = None
    out.info["proj_id"] = None
    out.info["device_info"] = None
    log.info("deidentified recording", extra={"subject": subject_label, "shift_days": offset})
    return out


def _walk(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    """Yield ``(path, text)`` for every scalar reachable in nested dict/list/tuple structures."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple | set):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif obj is not None:
        yield path, str(obj)


def scan_for_identifiers(raw: mne.io.BaseRaw, terms: Iterable[str]) -> list[str]:
    """Locations where any of ``terms`` (case-insensitive) still appear in the metadata.

    Searches every field of ``raw.info`` (including nested dicts), the annotation reference time
    and annotation descriptions. An empty list means none of the terms were found.
    """
    needles = [t.lower() for t in terms if t]
    hits: list[str] = []
    fields: list[tuple[str, str]] = list(_walk(dict(raw.info), "info"))
    fields.append(("annotations.orig_time", str(raw.annotations.orig_time)))
    fields += [("annotations.description", d) for d in set(raw.annotations.description)]
    for path, text in fields:
        low = text.lower()
        hits += [f"{path} contains {n!r}" for n in needles if n in low]
    return hits


def deidentify_metadata(
    meta: Mapping[str, Any], subject_label: str, salt: str | None = None
) -> dict[str, Any]:
    """De-identify a flat/nested metadata dict (e.g. a participants row or catalog record).

    Identifying keys are dropped, date-like values are shifted by the subject's offset, and the
    subject id is replaced by its pseudonym.
    """
    salt = salt or get_salt()
    offset = dt.timedelta(days=date_offset_days(subject_label, salt))
    out: dict[str, Any] = {}
    for key, val in meta.items():
        k = key.lower()
        if any(s in k for s in IDENTIFYING_KEYS):
            continue
        if isinstance(val, Mapping):
            out[key] = deidentify_metadata(val, subject_label, salt)
        elif any(s in k for s in DATE_KEYS):
            out[key] = _shift_value(val, offset)
        else:
            out[key] = val
    if "subject" in out or "participant_id" in out:
        out["subject" if "subject" in out else "participant_id"] = pseudonym(subject_label, salt)
    return out


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _shift_value(val: Any, offset: dt.timedelta) -> Any:
    if isinstance(val, dt.datetime | dt.date):
        return val - offset
    if isinstance(val, str) and _ISO.match(val):
        parsed = dt.datetime.fromisoformat(val)
        return (parsed - offset).isoformat()
    return val
