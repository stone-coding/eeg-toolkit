"""Command-line interface: ``eegkit download | to-bids | run | report``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from eegkit import __version__
from eegkit.config import Config, load_config
from eegkit.logging import configure_logging

app = typer.Typer(help="eegkit: manage, access and analyze human EEG data.", no_args_is_help=True)

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config (defaults if omitted).")
]
SubjectsOpt = Annotated[
    str | None, typer.Option("--subjects", "-s", help="Comma-separated subject ids, e.g. 1,2,3.")
]


def _cfg(config: Path | None, subjects: str | None = None) -> Config:
    try:
        cfg = load_config(config)
        if subjects:
            cfg = Config.model_validate(
                {**cfg.model_dump(), "subjects": [int(s) for s in subjects.split(",")]}
            )
    except (ValidationError, OSError, ValueError) as exc:
        typer.secho(f"Invalid configuration: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    configure_logging(cfg.log_level)
    return cfg


def _version(value: bool) -> None:
    if value:
        typer.echo(f"eegkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """eegkit command-line interface."""


@app.command()
def download(config: ConfigOpt = None, subjects: SubjectsOpt = None) -> None:
    """Download the EEGBCI motor-imagery runs (PhysioNet) for the configured subjects."""
    from eegkit.io import download_subjects

    cfg = _cfg(config, subjects)
    files = download_subjects(cfg.subjects, cfg.runs, cfg.paths.raw_dir)
    typer.echo(f"{len(files)} files available under {cfg.paths.raw_dir}")


@app.command("to-bids")
def to_bids_cmd(
    config: ConfigOpt = None,
    subjects: SubjectsOpt = None,
    overwrite: Annotated[bool, typer.Option(help="Rewrite existing BIDS files.")] = False,
) -> None:
    """Convert downloaded recordings into a BIDS dataset (mne-bids)."""
    from eegkit.bids import to_bids

    cfg = _cfg(config, subjects)
    paths = to_bids(
        cfg.subjects,
        cfg.runs,
        cfg.paths.raw_dir,
        cfg.paths.bids_root,
        overwrite,
        cfg.privacy.deidentify,
    )
    typer.echo(f"BIDS dataset at {cfg.paths.bids_root} ({len(paths)} runs)")


@app.command()
def run(
    config: ConfigOpt = None,
    subjects: SubjectsOpt = None,
    force: Annotated[bool, typer.Option(help="Ignore the cache and recompute.")] = False,
) -> None:
    """Run preprocessing -> epoching -> analysis for all subjects, then the group summary."""
    from eegkit.pipeline import run_batch

    cfg = _cfg(config, subjects)
    results, summary = run_batch(cfg, force=force)
    for r in results:
        acc = "  n/a" if r.accuracy is None else f"{r.accuracy:.2f}"
        cached = f" (cached: {','.join(r.cached_stages)})" if r.cached_stages else ""
        typer.echo(
            f"sub-{r.subject:03d}  {r.status:<6} epochs={r.n_epochs:<3} "
            f"dropped={r.n_dropped:<3} acc={acc}{cached}{'  ' + r.error if r.error else ''}"
        )
    if summary:
        d = summary["decoding"]
        typer.echo(
            f"\nGroup: n={summary['n_subjects_ok']}  decoding "
            f"{d['mean_accuracy']:.3f} ± {d['sd_accuracy']:.3f}  -> {cfg.paths.output_dir}/group"
        )
    if any(r.status != "ok" for r in results):
        raise typer.Exit(1)


@app.command()
def report(
    subject: Annotated[int, typer.Option("--subject", "-s", help="Subject id.")],
    config: ConfigOpt = None,
) -> None:
    """Write a markdown report (with figures) for one subject."""
    from eegkit.pipeline import report_subject

    cfg = _cfg(config)
    try:
        path = report_subject(cfg, subject)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Report written to {path}")


@app.command()
def index(config: ConfigOpt = None) -> None:
    """Build/refresh the metadata catalog from the BIDS dataset and pipeline outputs."""
    from eegkit.catalog import Catalog
    from eegkit.storage import LocalStorage

    cfg = _cfg(config)
    storage = LocalStorage(cfg.paths.storage_root)
    catalog = Catalog(cfg.paths.catalog_db)
    n_runs = catalog.index_bids(cfg.paths.bids_root, storage)
    n_status = catalog.sync_status(cfg.paths.output_dir)
    n_art = catalog.publish_artifacts(cfg, storage)
    typer.echo(
        f"Indexed {n_runs} runs, {n_status} subject statuses, {n_art} result files "
        f"-> {cfg.paths.catalog_db}"
    )


@app.command()
def serve(
    config: ConfigOpt = None,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Serve the REST API (OpenAPI docs at /docs)."""
    import uvicorn

    from eegkit.api import create_app
    from eegkit.catalog import Catalog
    from eegkit.storage import LocalStorage

    cfg = _cfg(config)
    app_ = create_app(Catalog(cfg.paths.catalog_db), LocalStorage(cfg.paths.storage_root))
    uvicorn.run(app_, host=host, port=port, log_level=cfg.log_level.lower())


@app.command()
def deidentify(
    src: Annotated[Path, typer.Argument(help="Input recording (.fif/.edf).")],
    dest: Annotated[Path, typer.Argument(help="Output .fif file.")],
    subject_label: Annotated[str, typer.Option("--label", help="Subject label, e.g. sub-001.")],
) -> None:
    """De-identify a recording: strip identifiers and shift dates (demonstration utility)."""
    import mne

    from eegkit.deidentify import deidentify_raw

    raw = mne.io.read_raw(src, preload=True, verbose="ERROR")
    deidentify_raw(raw, subject_label).save(dest, overwrite=True, verbose="ERROR")
    typer.echo(f"Wrote de-identified recording to {dest}")


if __name__ == "__main__":  # pragma: no cover
    app()
