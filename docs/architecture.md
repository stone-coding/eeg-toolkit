# Architecture

![Architecture](img/architecture.png)

```mermaid
flowchart LR
    EDF[PhysioNet EEGBCI EDF] -->|to-bids + deidentify| BIDS[(BIDS dataset)]
    BIDS --> P[pipeline: preprocess → epochs → analysis]
    P --> C[(cache: epochs, PSD, ERD)]
    P --> O[outputs: CSV + figures]
    BIDS --> CAT[(Catalog - SQLite)]
    O --> CAT
    C --> ST[[Storage: Local / S3]]
    CAT --> API[FastAPI]
    ST --> API
    CAT --> GUI1[Streamlit Explorer]
    DEV[AcquisitionDevice ABC] --- MOCK[MockAmplifier]
    MOCK --> ONL[OnlineProcessor]
    ONL --> GUI2[Streamlit Live monitor]
```

## Layers and dependencies

| Layer | Modules | Depends on |
|---|---|---|
| Core analysis | `io`, `preprocess`, `epochs`, `analysis`, `viz` | MNE, scikit-learn |
| Configuration / logging | `config`, `logging` | pydantic |
| Data management | `bids`, `pipeline`, `results`, `cli` | core |
| Data access | `catalog`, `storage`, `api`, `deidentify` | SQLAlchemy, FastAPI |
| Real time | `realtime.*` | numpy/scipy (+ core `analysis.spectral`) |
| GUI | `gui.*` | everything above |

Dependencies point one way: the core knows nothing about the catalog, API or GUI.

## Design decisions

* **BIDS is the system of record.** The catalog indexes BIDS; it can always be rebuilt.
* **De-identify at the ingest boundary**, so downstream components only see safe data.
* **Content-hash caching** per stage instead of timestamps: results are reproducible and
  stale outputs cannot survive a config change.
* **Storage addressed by key** so local disk and S3 are interchangeable.
* **Abstract device interface** so a real amplifier driver replaces `MockAmplifier` without
  touching consumers; a second `ScriptedDevice` in the tests proves it.
* **Gap-aware online processing** - correctness over continuity.
