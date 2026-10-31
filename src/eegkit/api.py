"""FastAPI service over the catalog and storage (interactive docs at ``/docs``)."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
from collections.abc import Iterator
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eegkit import __version__
from eegkit.catalog import STATUS_UNPROCESSED, Catalog, Run, Subject
from eegkit.config import load_config
from eegkit.storage import LocalStorage, NotFoundError, Storage

DESCRIPTION = """
Read-only access to an EEG metadata catalog and its processed results.

* **recordings** - list/filter runs by subject, task, sampling rate, duration, channel count and
  processing status
* **subjects** - per-subject processing status and registered result files
* **results** - band power per channel (mu/beta) and the cleaned epochs file (`.fif`)

Data served here is de-identified (see the privacy notes in the docs). This demo has **no
authentication**; put it behind an authenticating proxy before exposing it.
"""

Status = Literal["unprocessed", "ok", "failed"]


class ChannelOut(BaseModel):
    name: str
    type: str


class RunOut(BaseModel):
    id: int
    subject: str = Field(examples=["sub-001"])
    session: str = Field(examples=["ses-01"])
    task: str
    run_number: int | None
    sampling_rate: float = Field(description="Hz")
    n_channels: int
    duration_s: float
    n_events_left: int
    n_events_right: int
    acq_time: dt.datetime | None = Field(description="De-identified (shifted) acquisition time")
    file_path: str = Field(description="Storage key of the recording")
    status: Status


class RunDetail(RunOut):
    n_samples: int
    channels: list[ChannelOut]


class RunPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[RunOut]


class StatusOut(BaseModel):
    subject: str
    status: Status
    stage: str | None
    n_epochs: int | None
    n_dropped: int | None
    accuracy: float | None = Field(description="Cross-validated CSP+LDA accuracy")
    bad_channels: list[str]
    error: str | None
    updated: str | None


class ArtifactOut(BaseModel):
    kind: str
    storage_key: str
    size_bytes: int


class SubjectOut(BaseModel):
    label: str
    sex: str | None
    hand: str | None
    status: Status
    artifacts: list[ArtifactOut] = []


class BandPowerRow(BaseModel):
    condition: str
    channel: str
    band: str
    power: float = Field(description="V^2 (integrated PSD)")


def _run_out(run: Run, status: str | None = None) -> dict:
    return {
        "id": run.id,
        "subject": run.session.subject.label,
        "session": run.session.label,
        "task": run.task,
        "run_number": run.run_number,
        "sampling_rate": run.sampling_rate,
        "n_channels": run.n_channels,
        "duration_s": run.duration_s,
        "n_events_left": run.n_events_left,
        "n_events_right": run.n_events_right,
        "acq_time": run.acq_time,
        "file_path": run.file_path,
        "status": status or STATUS_UNPROCESSED,
    }


def _status_of(subj: Subject) -> str:
    return subj.status.status if subj.status else STATUS_UNPROCESSED


def _get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def _get_storage(request: Request) -> Storage:
    return request.app.state.storage


C = Annotated[Catalog, Depends(_get_catalog)]
S = Annotated[Storage, Depends(_get_storage)]


def create_app(catalog: Catalog, storage: Storage) -> FastAPI:
    """Build the application around a catalog and a storage backend."""
    app = FastAPI(
        title="eegkit API",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=[
            {"name": "meta", "description": "Service health"},
            {"name": "recordings", "description": "Browse and filter recordings"},
            {"name": "subjects", "description": "Subjects and processing status"},
            {"name": "results", "description": "Analysis outputs"},
        ],
    )
    app.state.catalog, app.state.storage = catalog, storage

    def subject_or_404(c: Catalog, label: str) -> Subject:
        subj = c.get_subject(label)
        if subj is None:
            raise HTTPException(404, f"unknown subject {label!r}")
        return subj

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/recordings", response_model=RunPage, tags=["recordings"])
    def list_recordings(
        c: C,
        subject: Annotated[str | None, Query(description="e.g. sub-001")] = None,
        task: str | None = None,
        status: Annotated[Status | None, Query(description="processing status")] = None,
        min_sampling_rate: Annotated[float | None, Query(ge=0)] = None,
        max_sampling_rate: Annotated[float | None, Query(ge=0)] = None,
        min_duration_s: Annotated[float | None, Query(ge=0)] = None,
        min_channels: Annotated[int | None, Query(ge=1)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        """List recordings matching all given metadata filters."""
        total, runs = c.query_runs(
            subject,
            task,
            status,
            min_sampling_rate,
            max_sampling_rate,
            min_duration_s,
            min_channels,
            limit,
            offset,
        )
        statuses = {s.label: _status_of(s) for s in c.list_subjects()}
        items = [_run_out(r, statuses.get(r.session.subject.label)) for r in runs]
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/recordings/{run_id}", response_model=RunDetail, tags=["recordings"])
    def get_recording(run_id: int, c: C) -> dict:
        """One recording with its channel list."""
        run = c.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"unknown recording {run_id}")
        subj = c.get_subject(run.session.subject.label)
        out = _run_out(run, _status_of(subj) if subj else None)
        out["n_samples"] = run.n_samples
        out["channels"] = [{"name": ch.name, "type": ch.type} for ch in run.channels]
        return out

    @app.get("/subjects", response_model=list[SubjectOut], tags=["subjects"])
    def list_subjects(c: C) -> list[dict]:
        """All subjects with their processing status."""
        return [
            {"label": s.label, "sex": s.sex, "hand": s.hand, "status": _status_of(s)}
            for s in c.list_subjects()
        ]

    @app.get("/subjects/{label}", response_model=SubjectOut, tags=["subjects"])
    def get_subject(label: str, c: C) -> dict:
        """One subject including its registered result files."""
        s = subject_or_404(c, label)
        return {
            "label": s.label,
            "sex": s.sex,
            "hand": s.hand,
            "status": _status_of(s),
            "artifacts": [
                {"kind": a.kind, "storage_key": a.storage_key, "size_bytes": a.size_bytes}
                for a in s.artifacts
            ],
        }

    @app.get("/subjects/{label}/status", response_model=StatusOut, tags=["subjects"])
    def get_status(label: str, c: C) -> dict:
        """Processing status of a subject (``unprocessed`` until the pipeline has run)."""
        s = subject_or_404(c, label)
        st = s.status
        return {
            "subject": s.label,
            "status": _status_of(s),
            "stage": st.stage if st else None,
            "n_epochs": st.n_epochs if st else None,
            "n_dropped": st.n_dropped if st else None,
            "accuracy": st.accuracy if st else None,
            "bad_channels": json.loads(st.bad_channels) if st and st.bad_channels else [],
            "error": st.error if st else None,
            "updated": st.updated if st else None,
        }

    def artifact_key(c: Catalog, label: str, kind: str) -> str:
        s = subject_or_404(c, label)
        art = next((a for a in s.artifacts if a.kind == kind), None)
        if art is None:
            raise HTTPException(
                404 if _status_of(s) != "unprocessed" else 409,
                f"no {kind} result for {label} (status: {_status_of(s)})",
            )
        return art.storage_key

    @app.get("/subjects/{label}/bandpower", response_model=list[BandPowerRow], tags=["results"])
    def get_bandpower(
        label: str,
        c: C,
        st: S,
        band: Annotated[str | None, Query(description="e.g. mu, beta")] = None,
        condition: Annotated[Literal["left", "right"] | None, Query()] = None,
        channel: Annotated[str | None, Query(description="e.g. C3")] = None,
    ) -> list[dict]:
        """Band power per condition/channel/band, optionally filtered."""
        key = artifact_key(c, label, "bandpower")
        try:
            df = pd.read_csv(io.BytesIO(st.read_bytes(key)))
        except NotFoundError as exc:
            raise HTTPException(404, "result file missing from storage") from exc
        for col, val in (("band", band), ("condition", condition), ("channel", channel)):
            if val is not None:
                df = df[df[col] == val]
        return df[["condition", "channel", "band", "power"]].to_dict("records")

    @app.get(
        "/subjects/{label}/epochs",
        tags=["results"],
        response_class=StreamingResponse,
        responses={200: {"content": {"application/octet-stream": {}}, "description": "FIF file"}},
    )
    def download_epochs(label: str, c: C, st: S) -> StreamingResponse:
        """Download the cleaned epochs (``*-epo.fif``) for a subject."""
        key = artifact_key(c, label, "epochs")
        try:
            size = st.size(key)
            stream = st.open(key)
        except NotFoundError as exc:
            raise HTTPException(404, "epochs file missing from storage") from exc

        def chunks() -> Iterator[bytes]:
            with stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{label}_epochs-epo.fif"',
                "Content-Length": str(size),
            },
        )

    return app


def create_app_from_config(config_path: str | None = None) -> FastAPI:
    """Application factory for ``uvicorn --factory`` (config path via ``$EEGKIT_CONFIG``)."""
    cfg = load_config(config_path or os.environ.get("EEGKIT_CONFIG"))
    return create_app(Catalog(cfg.paths.catalog_db), LocalStorage(cfg.paths.storage_root))
