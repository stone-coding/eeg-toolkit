# Privacy and de-identification

!!! warning "This is a demonstration, not a HIPAA certification"
    `eegkit.deidentify` shows a sound *approach* to protecting recording metadata. It has not
    been validated by a privacy officer, an IRB or an expert-determination process, and it must
    not be treated as making data HIPAA-, GDPR- or institution-compliant. The demo dataset
    (PhysioNet EEGBCI) is already public and pseudonymous.

## What counts as identifying in EEG data?

EEG files carry more than signal. Typical sources of identifying information:

| Where | Examples | How `eegkit` handles it |
|---|---|---|
| Subject block in the file header | first / middle / last name, birthday, hospital ID (MRN) | removed; hospital ID replaced by a keyed pseudonym; birthday **dropped** (not shifted) |
| Measurement date/time | `meas_date`, annotation reference time, BIDS `scans.tsv` `acq_time` | shifted back by a per-subject offset (1-3 years) |
| Free text | experimenter, session description, project name | cleared |
| Device info | amplifier serial number, site | cleared |
| Free-form metadata dicts | participants rows, catalog records | identifying keys dropped, date-like values shifted |
| The signal itself | rare: a name spoken or typed into an annotation | **not handled** - review annotations |

## Design

**Keyed, deterministic shifts.** The date offset and the pseudonym are derived with
`HMAC-SHA256(salt, label)`:

* *deterministic*: all runs of one subject get the same shift, so within-subject intervals
  (days between sessions, run order, time of day) are preserved for analysis;
* *keyed*: without the secret salt the offset cannot be recomputed from the label, and a
  different salt gives different shifts;
* *per subject*: offsets differ between subjects, so recordings cannot be re-aligned by shifting
  everything the same amount.

The salt is read from `$EEGKIT_DEID_SALT`. If it is unset the code falls back to a demo salt and
logs a warning - **never rely on the demo salt**.

**Birthday dropped, age capped.** A shifted birthday is still a date that can be attacked, so
the birthday is removed. `age_years()` returns whole years at recording, capped at 89, for the
cases where age is a needed covariate.

**Sex and handedness are kept** because they are legitimate research variables; with small
cohorts they can still contribute to re-identification (see limitations).

**Verifiable.** `scan_for_identifiers(raw, terms)` walks every field of the recording's `info`
(including nested dictionaries), the annotation reference time and annotation text, and reports
where any term still occurs. The test-suite plants names, an MRN, a birthday, device serials and
dates in a recording and proves that none remain

* in the de-identified in-memory object,
* in the raw **bytes** of a saved FIF file,
* in **every file** of a BIDS dataset written from it.

## Where de-identification happens in the pipeline

```
raw EDF  --(eegkit to-bids, privacy.deidentify: true)-->  BIDS dataset  -->  catalog / API
```

De-identification runs at the *ingest boundary*: everything downstream (batch pipeline, SQLite
catalog, REST API, GUI) only ever sees de-identified data. The catalog schema has no columns
for names or birth dates by design. A stand-alone command is also available:

```bash
EEGKIT_DEID_SALT=... eegkit deidentify raw.fif clean.fif --label sub-001
```

## Limitations and what a real deployment needs

* **Subject labels must be study codes**, never medical record numbers or initials.
* **Linkage risk**: rare combinations of sex, handedness, age and date-of-study can re-identify
  people in small cohorts. A real project needs a formal risk assessment (expert determination
  or safe-harbor review).
* **Date shifting is not anonymization**: whoever holds the salt can reverse it. Store the salt
  separately from the data (secrets manager), restrict access, and rotate per study.
* **Signal content** (e.g. ECG-like artifacts, speech in audio channels, free text in
  annotations) is not scrubbed.
* **The API has no authentication or audit log.** Put it behind an authenticating proxy, add
  role-based access, TLS and access logging before serving anything but public data.
* **Storage**: encrypt at rest (S3 SSE-KMS / disk encryption) and keep the identified originals,
  if they must exist, in a separate access-controlled location.
* **Consent and data-use agreements** determine what may be shared; software cannot decide that.
