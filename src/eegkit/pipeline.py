"""Batch pipeline: preprocess -> epoch -> analyze for many subjects, with caching.

Layout under ``paths.output_dir``::

    cache/sub-001/{epochs-epo.fif, stage1.json, bandpower.csv, psd.npz, erders.npz, ...}
    subjects/sub-001/{status.json, qc.json, bandpower.csv, erders_summary.csv,
                      decoding.json, figures/*.png, report.md}
    group/{decoding.csv, erders_summary.csv, summary.json, figures/*.png}

Each stage stores a hash of everything it depends on (config sections + input files); a stage
is skipped when the stored hash matches and its outputs exist.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eegkit import __version__, viz
from eegkit.analysis import (
    band_erders,
    band_power_table,
    compute_erders,
    compute_psd_by_condition,
    decode_csp_lda,
)
from eegkit.bids import bids_path, bids_runs_available, load_subject_bids
from eegkit.config import Config
from eegkit.epochs import make_epochs, reject_epochs
from eegkit.io import load_subject
from eegkit.logging import get_logger
from eegkit.preprocess import preprocess
from eegkit.results import (
    load_erders,
    load_psd,
    read_json,
    save_erders,
    save_psd,
    write_json,
)

log = get_logger(__name__)


def sub_label(subject: int) -> str:
    return f"sub-{subject:03d}"


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _source_files(cfg: Config, subject: int) -> list[Path]:
    if bids_runs_available(subject, cfg.runs, cfg.paths.bids_root):
        return [bids_path(subject, r, cfg.paths.bids_root).fpath for r in cfg.runs]
    return sorted(cfg.paths.raw_dir.rglob(f"S{subject:03d}R*.edf"))


def _fingerprint(files: list[Path]) -> list[tuple[str, int]]:
    return [(f.name, f.stat().st_size) for f in files if f.exists()]


@dataclass
class SubjectResult:
    subject: int
    status: str = "pending"  # ok | failed
    stage: str = ""
    n_epochs: int = 0
    n_dropped: int = 0
    accuracy: float | None = None
    accuracy_std: float | None = None
    bad_channels: list[str] = field(default_factory=list)
    cached_stages: list[str] = field(default_factory=list)
    error: str | None = None
    updated: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _load_raw(cfg: Config, subject: int) -> mne.io.BaseRaw:
    if bids_runs_available(subject, cfg.runs, cfg.paths.bids_root):
        return load_subject_bids(subject, cfg.runs, cfg.paths.bids_root)
    log.info("BIDS not found, reading raw EDF", extra={"subject": subject})
    return load_subject(subject, cfg.runs, cfg.paths.raw_dir)


def _dirs(cfg: Config, subject: int) -> tuple[Path, Path]:
    cdir = cfg.paths.cache_dir / sub_label(subject)
    sdir = cfg.paths.output_dir / "subjects" / sub_label(subject)
    for d in (cdir, sdir / "figures"):
        d.mkdir(parents=True, exist_ok=True)
    return cdir, sdir


def process_subject(cfg: Config, subject: int, force: bool = False) -> SubjectResult:
    """Run (or reuse cached) preprocessing, epoching and analysis for one subject."""
    res = SubjectResult(subject)
    try:
        _run_subject(cfg, subject, force, res)
        res.status = "ok"
    except Exception as exc:  # noqa: BLE001 - one bad subject must not kill the batch
        res.status = "failed"
        res.error = f"{type(exc).__name__}: {exc}"
        log.error("subject failed", extra={"subject": subject, "error": res.error})
    res.updated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _, sdir = _dirs(cfg, subject)
    write_json(sdir / "status.json", res.as_dict())
    return res


def _run_subject(cfg: Config, subject: int, force: bool, res: SubjectResult) -> None:
    cdir, sdir = _dirs(cfg, subject)
    a = cfg.analysis

    # ---- stage 1: preprocess + epoch + reject (cached as epochs-epo.fif) ----------------
    res.stage = "epochs"
    files = _source_files(cfg, subject)
    if not files:
        raise FileNotFoundError(f"no recordings found for subject {subject}")
    key1 = _hash(
        {
            "v": __version__,
            "pre": cfg.preprocess.model_dump(),
            "ep": cfg.epochs.model_dump(),
            "runs": cfg.runs,
            "src": _fingerprint(files),
        }
    )
    epo_path, m1_path = cdir / "epochs-epo.fif", cdir / "stage1.json"
    if not force and m1_path.exists() and epo_path.exists() and read_json(m1_path)["key"] == key1:
        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        qc = read_json(m1_path)["qc"]
        res.cached_stages.append("epochs")
        log.info("cache hit", extra={"subject": subject, "stage": "epochs"})
    else:
        raw = _load_raw(cfg, subject)
        clean, bads = preprocess(raw, cfg.preprocess)
        epochs = make_epochs(clean, cfg.epochs.tmin, cfg.epochs.tmax, cfg.epochs.baseline)
        epochs, rej = reject_epochs(epochs, cfg.epochs.reject_uv, cfg.epochs.reject_channels)
        qc = {"bad_channels": {"flat": bads.flat, "noisy": bads.noisy}, "rejection": rej.as_dict()}
        epochs.save(epo_path, overwrite=True, verbose="ERROR")
        write_json(m1_path, {"key": key1, "qc": qc})
    write_json(sdir / "qc.json", qc)
    res.n_epochs = len(epochs)
    res.n_dropped = qc["rejection"]["n_dropped"]
    res.bad_channels = qc["bad_channels"]["flat"] + qc["bad_channels"]["noisy"]

    # ---- stage 2: analysis (cached as csv/npz/json) -------------------------------------
    res.stage = "analysis"
    key2 = _hash({"k1": key1, "an": a.model_dump()})
    m2_path = cdir / "stage2.json"
    outputs = [cdir / n for n in ("bandpower.csv", "psd.npz", "erders.npz", "decoding.json")]
    if (
        not force
        and m2_path.exists()
        and all(o.exists() for o in outputs)
        and read_json(m2_path)["key"] == key2
    ):
        res.cached_stages.append("analysis")
        log.info("cache hit", extra={"subject": subject, "stage": "analysis"})
    else:
        bp = band_power_table(epochs, a.bands, a.band_window)
        bp.insert(0, "subject", subject)
        bp.to_csv(cdir / "bandpower.csv", index=False)
        psds, freqs, names = compute_psd_by_condition(epochs, tmin=a.band_window[0])
        save_psd(cdir / "psd.npz", psds, freqs, names)
        save_erders(
            cdir / "erders.npz",
            compute_erders(
                epochs, a.erders_channels, a.erders_freqs, a.erders_baseline, a.erders_crop
            ),
        )
        try:
            dec = decode_csp_lda(epochs, **a.decoding.model_dump())
            dec_out: dict[str, Any] = {"subject": subject, **dec.__dict__}
        except ValueError as exc:
            log.warning("decoding skipped", extra={"subject": subject, "reason": str(exc)})
            dec_out = {"subject": subject, "accuracy": None, "std": None, "error": str(exc)}
        write_json(cdir / "decoding.json", dec_out)
        write_json(m2_path, {"key": key2})

    dec_out = read_json(cdir / "decoding.json")
    res.accuracy, res.accuracy_std = dec_out.get("accuracy"), dec_out.get("std")

    # ---- per-subject outputs (cheap; always refreshed from cache) -----------------------
    res.stage = "outputs"
    erd = load_erders(cdir / "erders.npz")
    rows = [
        {
            "subject": subject,
            "condition": cond,
            "channel": ch,
            "band": band,
            "window": f"{a.band_window[0]}-{a.band_window[1]}s",
            "erd_pct": v,
        }
        for cond, r in erd.items()
        for band, rng in a.bands.items()
        for ch, v in band_erders(r, rng, a.band_window).items()
    ]
    pd.DataFrame(rows).to_csv(sdir / "erders_summary.csv", index=False)
    pd.read_csv(cdir / "bandpower.csv").to_csv(sdir / "bandpower.csv", index=False)
    write_json(sdir / "decoding.json", dec_out)
    _subject_figures(cfg, subject, epochs.info, cdir, sdir / "figures")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _subject_figures(cfg: Config, subject: int, info: mne.Info, cdir: Path, fdir: Path) -> None:
    a = cfg.analysis
    psds, freqs, names = load_psd(cdir / "psd.npz")
    chans = tuple(a.erders_channels)
    t = f"Subject {subject}"
    _save(viz.plot_psd_comparison(psds, freqs, names, chans, a.bands, t), fdir / "psd.png")
    _save(viz.plot_erders(load_erders(cdir / "erders.npz"), title=t), fdir / "erders.png")
    _save(
        viz.plot_band_topomaps(pd.read_csv(cdir / "bandpower.csv"), info, title=t),
        fdir / "topomaps.png",
    )


# ----------------------------------------------------------------------------------------
# group level
# ----------------------------------------------------------------------------------------


def aggregate(cfg: Config, results: list[SubjectResult]) -> dict[str, Any]:
    """Group-level tables, summary.json and figures from the cached per-subject results."""
    ok = [r for r in results if r.status == "ok"]
    gdir = cfg.paths.output_dir / "group"
    (gdir / "figures").mkdir(parents=True, exist_ok=True)
    a = cfg.analysis
    if not ok:
        raise RuntimeError("no subject processed successfully; cannot aggregate")

    caches = {r.subject: cfg.paths.cache_dir / sub_label(r.subject) for r in ok}
    dec = pd.DataFrame([read_json(c / "decoding.json") for c in caches.values()])
    dec = dec[["subject", "accuracy", "std", "n_epochs"]].copy()
    dec["n_dropped"] = [r.n_dropped for r in ok]
    dec.to_csv(gdir / "decoding.csv", index=False)
    dec_valid = dec.dropna(subset=["accuracy"])

    erd_tab = pd.concat(
        [
            pd.read_csv(
                cfg.paths.output_dir / "subjects" / sub_label(r.subject) / "erders_summary.csv"
            )
            for r in ok
        ]
    )
    erd_group = (
        erd_tab.groupby(["condition", "channel", "band"])["erd_pct"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    erd_group.to_csv(gdir / "erders_summary.csv", index=False)

    # group-mean time-frequency maps and PSDs
    erd_by_sub = [load_erders(c / "erders.npz") for c in caches.values()]
    group_erd = {
        cond: type(erd_by_sub[0][cond])(
            cond,
            erd_by_sub[0][cond].ch_names,
            erd_by_sub[0][cond].freqs,
            erd_by_sub[0][cond].times,
            np.mean([e[cond].data for e in erd_by_sub], axis=0),
        )
        for cond in erd_by_sub[0]
    }
    psd_by_sub = [load_psd(c / "psd.npz") for c in caches.values()]
    freqs, names = psd_by_sub[0][1], psd_by_sub[0][2]
    group_psd = {c: np.mean([p[0][c] for p in psd_by_sub], axis=0) for c in psd_by_sub[0][0]}
    power = pd.concat([pd.read_csv(c / "bandpower.csv") for c in caches.values()])
    gpower = (
        power.assign(logp=np.log(power["power"]))
        .groupby(["condition", "channel", "band"])["logp"]
        .mean()
        .reset_index()
    )
    gpower["power"] = np.exp(gpower.pop("logp"))  # geometric mean across subjects
    info = mne.read_epochs(
        next(iter(caches.values())) / "epochs-epo.fif", preload=False, verbose="ERROR"
    ).info

    fdir = gdir / "figures"
    n = len(ok)
    _save(
        viz.plot_psd_comparison(
            group_psd, freqs, names, tuple(a.erders_channels), a.bands, f"Group mean PSD (n={n})"
        ),
        fdir / "psd.png",
    )
    _save(
        viz.plot_erders(group_erd, vlim=50, title=f"Group mean ERD/ERS (n={n})"),
        fdir / "erders.png",
    )
    _save(
        viz.plot_band_topomaps(gpower, info, title=f"Group band power (n={n})"),
        fdir / "topomaps.png",
    )
    _save(
        viz.plot_decoding_accuracy(dec_valid, "Left vs right imagery: CSP+LDA, 5-fold CV"),
        fdir / "decoding.png",
    )

    summary = {
        "n_subjects_requested": len(results),
        "n_subjects_ok": len(ok),
        "failed": {r.subject: r.error for r in results if r.status != "ok"},
        "decoding": {
            "mean_accuracy": float(dec_valid["accuracy"].mean()),
            "sd_accuracy": float(dec_valid["accuracy"].std()),
            "n_above_chance_70": int((dec_valid["accuracy"] >= 0.7).sum()),
            "per_subject": {
                int(s): float(v)
                for s, v in zip(dec_valid["subject"], dec_valid["accuracy"], strict=True)
            },
        },
        "erders_mean_pct": {
            f"{c}/{ch}/{b}": round(float(m), 2)
            for c, ch, b, m in zip(
                erd_group["condition"],
                erd_group["channel"],
                erd_group["band"],
                erd_group["mean"],
                strict=True,
            )
        },
        "total_epochs": int(dec["n_epochs"].sum()),
        "total_dropped": int(dec["n_dropped"].sum()),
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(gdir / "summary.json", summary)
    return summary


def run_batch(
    cfg: Config, subjects: list[int] | None = None, force: bool = False
) -> tuple[list[SubjectResult], dict[str, Any] | None]:
    """Process all subjects (cache-aware) then write the group-level outputs."""
    subs = subjects or cfg.subjects
    results = [process_subject(cfg, s, force=force) for s in subs]
    summary = aggregate(cfg, results) if any(r.status == "ok" for r in results) else None
    return results, summary


def report_subject(cfg: Config, subject: int) -> Path:
    """(Re)build one subject's outputs from cache and write ``report.md``; return its path."""
    res = process_subject(cfg, subject)
    if res.status != "ok":
        raise RuntimeError(f"subject {subject} failed: {res.error}")
    _, sdir = _dirs(cfg, subject)
    qc = read_json(sdir / "qc.json")
    erd = pd.read_csv(sdir / "erders_summary.csv")
    mu = erd[(erd.band == "mu") & erd.channel.isin(["C3", "C4"])]
    table = mu.pivot(index="condition", columns="channel", values="erd_pct").round(1)
    acc = "n/a" if res.accuracy is None else f"{res.accuracy:.2f} ± {res.accuracy_std:.2f}"
    md = f"""# Subject {subject}

- Runs: {cfg.runs} (motor imagery, left vs right fist)
- Epochs kept: **{res.n_epochs}** (dropped {res.n_dropped} above
  {qc["rejection"]["threshold_uv"]:g} µV)
- Bad channels (interpolated): {res.bad_channels or "none"}
- Decoding accuracy (CSP+LDA, {cfg.analysis.decoding.n_splits}-fold): **{acc}**

## Mu-band ERD/ERS (%), {cfg.analysis.band_window[0]}–{cfg.analysis.band_window[1]} s

{table.to_markdown()}

## Figures

![PSD](figures/psd.png)

![ERD/ERS](figures/erders.png)

![Topomaps](figures/topomaps.png)
"""
    path = sdir / "report.md"
    path.write_text(md)
    return path
