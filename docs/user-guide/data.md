# Data: `download` and `to-bids`

## `eegkit download`

```bash
eegkit download [-c config.yaml] [-s 1,2,3]
```

Downloads the configured runs (default 4, 8, 12 = left vs. right fist imagery) of the
configured subjects from PhysioNet through `mne.datasets.eegbci`. Existing files are reused.
`-s` overrides the subject list for this call only.

## `eegkit to-bids`

```bash
eegkit to-bids [-c config.yaml] [-s ...] [--overwrite]
```

Converts the EDFs to a [BIDS](https://bids.neuroimaging.io) EEG dataset with `mne-bids`:

```
data/bids/
  dataset_description.json  participants.tsv  README
  sub-001/ses-01/
    sub-001_ses-01_scans.tsv
    eeg/sub-001_ses-01_task-motorimagery_run-04_eeg.edf
        ..._events.tsv  ..._channels.tsv  ..._electrodes.tsv  ..._eeg.json
```

* Channel names are standardized (`Fc5.` -> `FC5`) and the standard 10-05 montage is applied.
* Events are named `rest`, `left`, `right`.
* With `privacy.deidentify: true` (default) recordings are de-identified first - dates are
  shifted and identifiers removed. Set `EEGKIT_DEID_SALT` (see [Privacy](../privacy.md)).
* Existing runs are skipped unless `--overwrite`.
