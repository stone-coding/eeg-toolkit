# Catalog and API: `index` and `serve`

## `eegkit index`

Builds/refreshes `outputs/catalog.sqlite` from the BIDS dataset and the pipeline outputs:

* **runs** (sampling rate, channel count, duration, cue counts, de-identified acquisition time,
  storage key), **channels**, **subjects**, **sessions**;
* **processing status** from each subject's `status.json` (`unprocessed` / `ok` / `failed`);
* **artifacts** - epochs, band power, ERD maps, PSDs - copied into storage under
  `bids/derivatives/eegkit/sub-XXX/`.

It is idempotent: run it again after `eegkit run`.

## `eegkit serve`

```bash
eegkit serve [-c config.yaml] [--host 127.0.0.1] [--port 8000]
```

Interactive OpenAPI docs: <http://localhost:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `GET /recordings` | filter by `subject`, `task`, `status`, `min/max_sampling_rate`, `min_duration_s`, `min_channels`; paginated |
| `GET /recordings/{id}` | one recording with channels |
| `GET /subjects`, `/subjects/{label}` | subjects and registered result files |
| `GET /subjects/{label}/status` | processing status, epochs kept, accuracy, bad channels, error |
| `GET /subjects/{label}/bandpower` | band power rows, filter by `band`, `condition`, `channel` |
| `GET /subjects/{label}/epochs` | streamed download of the cleaned epochs (`.fif`) |

```bash
curl "localhost:8000/recordings?status=ok&min_channels=64&limit=5"
curl "localhost:8000/subjects/sub-001/bandpower?band=mu&channel=C3"
curl -OJ localhost:8000/subjects/sub-001/epochs
```

Status codes: `404` unknown subject/recording or missing result, `409` subject not processed
yet, `422` invalid parameter.

!!! warning
    The demo API has **no authentication**. See [Privacy](../privacy.md).

## Storage backends

The catalog stores *keys*, resolved by a `Storage`: `LocalStorage` (default, rejects keys that
escape the root, including via symlinks) or `S3Storage(bucket, prefix, endpoint_url=...)` for
AWS S3 / MinIO. Implement `eegkit.storage.Storage` to add another backend.
