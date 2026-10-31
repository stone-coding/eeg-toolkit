"""Small helpers to persist intermediate analysis results (npz / json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eegkit.analysis.erders import ErdersResult


def save_erders(path: Path, results: dict[str, ErdersResult]) -> None:
    first = next(iter(results.values()))
    arrays: dict[str, Any] = {f"data__{c}": r.data for c, r in results.items()}
    np.savez_compressed(
        path, freqs=first.freqs, times=first.times, ch_names=np.array(first.ch_names), **arrays
    )


def load_erders(path: Path) -> dict[str, ErdersResult]:
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["ch_names"]]
        return {
            k.split("__", 1)[1]: ErdersResult(
                k.split("__", 1)[1], names, z["freqs"], z["times"], z[k]
            )
            for k in z.files
            if k.startswith("data__")
        }


def save_psd(path: Path, psds: dict[str, np.ndarray], freqs: np.ndarray, ch: list[str]) -> None:
    arrays: dict[str, Any] = {f"psd__{c}": p for c, p in psds.items()}
    np.savez_compressed(path, freqs=freqs, ch_names=np.array(ch), **arrays)


def load_psd(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        psds = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("psd__")}
        return psds, z["freqs"], [str(n) for n in z["ch_names"]]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
