"""Shared GUI plumbing: configuration, catalog and storage access."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mne
import numpy as np
import streamlit as st

from eegkit.catalog import Catalog
from eegkit.config import Config, load_config
from eegkit.storage import LocalStorage, Storage


@st.cache_resource
def context() -> tuple[Config, Catalog, Storage]:
    """Config (``$EEGKIT_CONFIG`` or defaults), catalog and storage, created once per server."""
    path = os.environ.get("EEGKIT_CONFIG")
    cfg = load_config(path if path and Path(path).exists() else None)
    return cfg, Catalog(cfg.paths.catalog_db), LocalStorage(cfg.paths.storage_root)


@st.cache_data(show_spinner="Loading recording...")
def load_run_arrays(file_key: str) -> tuple[np.ndarray, float, list[str], list[tuple[float, str]]]:
    """Read one BIDS run through the storage layer: ``(data, sfreq, ch_names, markers)``."""
    _, _, storage = context()
    with tempfile.TemporaryDirectory() as tmp:
        local = storage.get_file(file_key, Path(tmp) / Path(file_key).name)
        raw = mne.io.read_raw_edf(local, preload=True, verbose="ERROR")
    markers = [
        (float(o), str(d))
        for o, d in zip(raw.annotations.onset, raw.annotations.description, strict=True)
        if d in ("left", "right")
    ]
    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names), markers


@st.cache_data(show_spinner="Loading results...")
def load_artifact_bytes(storage_key: str) -> bytes:
    return context()[2].read_bytes(storage_key)


def empty_catalog_hint() -> None:
    st.info(
        "The catalog is empty. Download and process the data first:\n\n"
        "```bash\neegkit download && eegkit to-bids && eegkit run && eegkit index\n```"
    )
