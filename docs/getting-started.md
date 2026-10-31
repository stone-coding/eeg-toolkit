# Getting started

## Install

```bash
git clone https://github.com/stone-coding/eeg-toolkit && cd eeg-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or choose extras: api, gui, s3, dev, docs
```

Python 3.10+ is required. Core dependencies: MNE-Python, mne-bids, scikit-learn, pydantic, Typer.

## Run the whole workflow

```bash
export EEGKIT_DEID_SALT="$(openssl rand -hex 16)"   # secret used for date shifting
eegkit download                  # PhysioNet EEGBCI, subjects 1-10 (~100 MB)
eegkit to-bids                   # BIDS dataset in data/bids (de-identified)
eegkit run -c configs/default.yaml   # preprocessing -> epochs -> analysis, per subject + group
eegkit index                     # build the metadata catalog
eegkit serve                     # REST API on :8000 (OpenAPI docs at /docs)
eegkit app                       # Streamlit GUI on :8501
```

Results appear under `outputs/`:

```
outputs/
  subjects/sub-001/{status.json, qc.json, bandpower.csv, erders_summary.csv,
                    decoding.json, figures/, report.md}
  group/{decoding.csv, erders_summary.csv, summary.json, figures/}
  cache/            intermediate results (epochs, PSD, ERD maps)
  catalog.sqlite
```

## Docker

```bash
docker compose up --build        # API :8000, GUI :8501 (mounts ./data and ./outputs)
```

## Python API

```python
from eegkit.io import load_subject
from eegkit.preprocess import preprocess
from eegkit.epochs import make_epochs, reject_epochs

raw = load_subject(1, [4, 8, 12], "data/raw")
clean, bads = preprocess(raw)
epochs, report = reject_epochs(make_epochs(clean))
print(report)
```

See the [tutorial](tutorial.ipynb) for the complete analysis.

## Run the tests

```bash
pytest                    # synthetic signals only, no downloads
pytest --runslow          # + tests on the real dataset (needs `eegkit download`)
```
