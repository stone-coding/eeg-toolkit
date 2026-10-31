"""Metadata catalog (SQLite via SQLAlchemy): subjects, sessions, runs, channels, status.

The catalog indexes a BIDS dataset (file *keys* relative to a :class:`~eegkit.storage.Storage`)
and the outputs of the batch pipeline. It never stores names or birth dates; measurement times
come from the (already de-identified) BIDS files.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mne
from mne_bids import get_entities_from_fname, read_raw_bids
from sqlalchemy import ForeignKey, String, UniqueConstraint, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from eegkit.bids import bids_path
from eegkit.config import Config
from eegkit.logging import get_logger
from eegkit.storage import Storage

log = get_logger(__name__)

STATUS_UNPROCESSED = "unprocessed"
DERIV_PREFIX = "bids/derivatives/eegkit"


class Base(DeclarativeBase):
    pass


class Subject(Base):
    __tablename__ = "subjects"
    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(32), unique=True, index=True)  # sub-001
    sex: Mapped[str | None] = mapped_column(String(8))
    hand: Mapped[str | None] = mapped_column(String(8))
    sessions: Mapped[list[SessionRow]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    status: Mapped[ProcessingStatus | None] = relationship(
        back_populates="subject", cascade="all, delete-orphan", uselist=False
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )


class SessionRow(Base):
    __tablename__ = "sessions"
    __table_args__ = (UniqueConstraint("subject_id", "label"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"))
    label: Mapped[str] = mapped_column(String(32))  # ses-01
    subject: Mapped[Subject] = relationship(back_populates="sessions")
    runs: Mapped[list[Run]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    task: Mapped[str] = mapped_column(String(64), index=True)
    run_number: Mapped[int | None]
    sampling_rate: Mapped[float]
    n_channels: Mapped[int]
    n_samples: Mapped[int]
    duration_s: Mapped[float]
    n_events_left: Mapped[int] = mapped_column(default=0)
    n_events_right: Mapped[int] = mapped_column(default=0)
    acq_time: Mapped[dt.datetime | None]  # de-identified (shifted) measurement time
    file_path: Mapped[str] = mapped_column(String(512), unique=True)  # storage key
    session: Mapped[SessionRow] = relationship(back_populates="runs")
    channels: Mapped[list[Channel]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(16))
    type: Mapped[str] = mapped_column(String(16))
    run: Mapped[Run] = relationship(back_populates="channels")


class ProcessingStatus(Base):
    __tablename__ = "processing_status"
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"), unique=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # unprocessed|ok|failed
    stage: Mapped[str | None] = mapped_column(String(16))
    n_epochs: Mapped[int | None]
    n_dropped: Mapped[int | None]
    accuracy: Mapped[float | None]
    bad_channels: Mapped[str | None] = mapped_column(String(256))  # JSON list
    error: Mapped[str | None] = mapped_column(String(512))
    updated: Mapped[str | None] = mapped_column(String(40))
    subject: Mapped[Subject] = relationship(back_populates="status")


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("subject_id", "kind"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"))
    kind: Mapped[str] = mapped_column(String(32))  # epochs | bandpower | erders | decoding
    storage_key: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int]
    subject: Mapped[Subject] = relationship(back_populates="artifacts")


ARTIFACT_FILES = {
    "epochs": ("cache", "epochs-epo.fif"),
    "bandpower": ("subjects", "bandpower.csv"),
    "erders": ("subjects", "erders_summary.csv"),
    "decoding": ("subjects", "decoding.json"),
}


class Catalog:
    """Thin repository around the SQLAlchemy engine."""

    def __init__(self, url: str | Path = "sqlite://") -> None:
        url = str(url)
        if "://" not in url:
            Path(url).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{url}"
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory connection
        self.engine: Engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self._sessionmaker = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessionmaker() as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise

    # ---- ingestion -------------------------------------------------------------------

    def index_bids(self, bids_root: str | Path, storage: Storage, bids_prefix: str = "bids") -> int:
        """Index every EEG run in a BIDS dataset. Idempotent; returns the number of runs seen."""
        root = Path(bids_root)
        files = sorted(root.glob("sub-*/ses-*/eeg/*_eeg.edf"))
        n = 0
        with self.session() as s:
            for f in files:
                ent = get_entities_from_fname(f.name)
                sub_label = f"sub-{ent['subject']}"
                key = f"{bids_prefix}/{f.relative_to(root).as_posix()}"
                bp = bids_path(int(ent["subject"]), int(ent["run"] or 0), root)
                bp.update(task=ent["task"], session=ent["session"], run=ent["run"])
                raw = read_raw_bids(bp, verbose="ERROR")
                self._upsert_run(s, sub_label, f"ses-{ent['session']}", key, ent, raw)
                n += 1
        log.info("indexed bids", extra={"runs": n, "root": str(root)})
        return n

    def _upsert_run(
        self, s: Session, sub_label: str, ses_label: str, key: str, ent: dict, raw: mne.io.BaseRaw
    ) -> None:
        subj = s.scalar(select(Subject).where(Subject.label == sub_label))
        if subj is None:
            info = raw.info.get("subject_info") or {}
            subj = Subject(
                label=sub_label,
                sex={0: None, 1: "M", 2: "F"}.get(info.get("sex", 0)),
                hand={0: None, 1: "R", 2: "L", 3: "A"}.get(info.get("hand", 0)),
            )
            s.add(subj)
            s.flush()
        ses = s.scalar(
            select(SessionRow).where(
                SessionRow.subject_id == subj.id, SessionRow.label == ses_label
            )
        )
        if ses is None:
            ses = SessionRow(subject_id=subj.id, label=ses_label)
            s.add(ses)
            s.flush()
        run = s.scalar(select(Run).where(Run.file_path == key))
        desc = list(raw.annotations.description)
        fields = {
            "task": ent["task"] or "unknown",
            "run_number": int(ent["run"]) if ent["run"] else None,
            "sampling_rate": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "n_samples": int(raw.n_times),
            "duration_s": float(raw.times[-1] + 1 / raw.info["sfreq"]),
            "n_events_left": desc.count("left"),
            "n_events_right": desc.count("right"),
            "acq_time": raw.info["meas_date"].replace(tzinfo=None)
            if raw.info["meas_date"]
            else None,
        }
        if run is None:
            run = Run(session_id=ses.id, file_path=key, **fields)
            run.channels = [
                Channel(name=n, type=t)
                for n, t in zip(raw.ch_names, raw.get_channel_types(), strict=True)
            ]
            s.add(run)
        else:
            for k, v in fields.items():
                setattr(run, k, v)

    def sync_status(self, output_dir: str | Path) -> int:
        """Load ``subjects/*/status.json`` written by the pipeline; returns rows updated."""
        n = 0
        with self.session() as s:
            for subj in s.scalars(select(Subject)):
                f = Path(output_dir) / "subjects" / subj.label / "status.json"
                data: dict[str, Any] = (
                    json.loads(f.read_text()) if f.exists() else {"status": STATUS_UNPROCESSED}
                )
                row = subj.status or ProcessingStatus(subject=subj, status=STATUS_UNPROCESSED)
                row.status = data["status"]
                row.stage = data.get("stage")
                row.n_epochs = data.get("n_epochs")
                row.n_dropped = data.get("n_dropped")
                row.accuracy = data.get("accuracy")
                row.bad_channels = json.dumps(data.get("bad_channels", []))
                row.error = (data.get("error") or None) and str(data["error"])[:512]
                row.updated = data.get("updated")
                s.add(row)
                n += 1
        return n

    def publish_artifacts(self, cfg: Config, storage: Storage) -> int:
        """Copy per-subject results into storage (BIDS derivatives) and register them."""
        n = 0
        with self.session() as s:
            for subj in s.scalars(select(Subject).options(selectinload(Subject.artifacts))):
                for kind, (area, name) in ARTIFACT_FILES.items():
                    base = cfg.paths.cache_dir if area == "cache" else cfg.paths.output_dir / area
                    src = base / subj.label / name
                    if not src.exists():
                        continue
                    key = f"{DERIV_PREFIX}/{subj.label}/{name}"
                    storage.put_file(key, src)
                    existing = next((a for a in subj.artifacts if a.kind == kind), None)
                    if existing:
                        existing.storage_key, existing.size_bytes = key, storage.size(key)
                    else:
                        s.add(
                            Artifact(
                                subject=subj,
                                kind=kind,
                                storage_key=key,
                                size_bytes=storage.size(key),
                            )
                        )
                    n += 1
        return n

    # ---- queries ---------------------------------------------------------------------

    def query_runs(
        self,
        subject: str | None = None,
        task: str | None = None,
        status: str | None = None,
        min_sampling_rate: float | None = None,
        max_sampling_rate: float | None = None,
        min_duration_s: float | None = None,
        min_channels: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[Run]]:
        """Filter runs by metadata; returns ``(total_matches, page)``."""
        stmt = select(Run).join(SessionRow).join(Subject).outerjoin(ProcessingStatus)
        if subject:
            stmt = stmt.where(Subject.label == subject)
        if task:
            stmt = stmt.where(Run.task == task)
        if status:
            stmt = stmt.where(func.coalesce(ProcessingStatus.status, STATUS_UNPROCESSED) == status)
        if min_sampling_rate is not None:
            stmt = stmt.where(Run.sampling_rate >= min_sampling_rate)
        if max_sampling_rate is not None:
            stmt = stmt.where(Run.sampling_rate <= max_sampling_rate)
        if min_duration_s is not None:
            stmt = stmt.where(Run.duration_s >= min_duration_s)
        if min_channels is not None:
            stmt = stmt.where(Run.n_channels >= min_channels)
        with self.session() as s:
            total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            page = s.scalars(
                stmt.options(selectinload(Run.session).selectinload(SessionRow.subject))
                .order_by(Subject.label, Run.run_number)
                .limit(limit)
                .offset(offset)
            ).all()
            return total, list(page)

    def get_subject(self, label: str) -> Subject | None:
        with self.session() as s:
            return s.scalar(
                select(Subject)
                .where(Subject.label == label)
                .options(selectinload(Subject.status), selectinload(Subject.artifacts))
            )

    def list_subjects(self) -> list[Subject]:
        with self.session() as s:
            return list(
                s.scalars(
                    select(Subject).options(selectinload(Subject.status)).order_by(Subject.label)
                )
            )

    def get_run(self, run_id: int) -> Run | None:
        with self.session() as s:
            return s.scalar(
                select(Run)
                .where(Run.id == run_id)
                .options(
                    selectinload(Run.channels),
                    selectinload(Run.session).selectinload(SessionRow.subject),
                )
            )
