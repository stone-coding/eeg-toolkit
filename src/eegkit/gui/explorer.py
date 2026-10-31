"""Explorer page: browse the catalog; raw traces, PSD, ERD/ERS maps and topomaps."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st

from eegkit import viz
from eegkit.analysis.erders import ErdersResult
from eegkit.gui.common import context, empty_catalog_hint, load_artifact_bytes, load_run_arrays
from eegkit.io import MONTAGE


def _fig(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _npz(storage_key: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(load_artifact_bytes(storage_key)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def render() -> None:
    cfg, catalog, _ = context()
    st.title("Recording explorer")
    subjects = catalog.list_subjects()
    if not subjects:
        empty_catalog_hint()
        return

    with st.sidebar:
        label = st.selectbox("Subject", [s.label for s in subjects]) or subjects[0].label
        _, runs = catalog.query_runs(subject=label)
        run = st.selectbox(
            "Run", runs, format_func=lambda r: f"run-{r.run_number:02d}  ({r.duration_s:.0f} s)"
        )
    subject = catalog.get_subject(label)
    assert subject is not None and run is not None
    status = subject.status.status if subject.status else "unprocessed"
    arts = {a.kind: a.storage_key for a in subject.artifacts}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampling rate", f"{run.sampling_rate:g} Hz")
    c2.metric("Channels", run.n_channels)
    c3.metric("Cues (L / R)", f"{run.n_events_left} / {run.n_events_right}")
    st_ = subject.status
    c4.metric(
        "Decoding accuracy",
        f"{st_.accuracy:.2f}" if st_ and st_.accuracy is not None else "n/a",
        help=f"Processing status: {status}",
    )

    tabs = st.tabs(["Raw traces", "PSD", "ERD/ERS", "Topomaps"])

    with tabs[0]:
        data, sfreq, names, markers = load_run_arrays(run.file_path)
        default = [c for c in ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4", "Oz") if c in names]
        chans = st.multiselect("Channels", names, default=default)
        col_a, col_b = st.columns(2)
        win = col_a.slider("Window (s)", 2, 30, 10)
        scale = col_b.slider("Scale (µV per trace)", 20, 400, 100, step=10)
        t0 = st.slider("Start (s)", 0.0, max(float(run.duration_s - win), 1.0), 0.0, step=0.5)
        if chans:
            i0, i1 = int(t0 * sfreq), int((t0 + win) * sfreq)
            rows = [names.index(c) for c in chans]
            times = np.arange(i0, min(i1, data.shape[1])) / sfreq
            _fig(viz.plot_traces(data[rows, i0:i1], times, chans, scale, markers))
        st.caption("Dashed lines: imagery cues (left / right). Display only - unfiltered signal.")

    needs = "Run `eegkit run && eegkit index` to compute and register results for this subject."
    with tabs[1]:
        if "psd" not in arts:
            st.warning(needs)
        else:
            z = _npz(arts["psd"])
            names_p = [str(n) for n in z["ch_names"]]
            psds = {k.split("__", 1)[1]: z[k] for k in z if k.startswith("psd__")}
            pick = st.multiselect(
                "Channels", names_p, default=[c for c in ("C3", "C4") if c in names_p], key="psdch"
            )
            if pick:
                _fig(
                    viz.plot_psd_comparison(
                        psds, z["freqs"], names_p, tuple(pick), cfg.analysis.bands, label
                    )
                )

    with tabs[2]:
        if "erders_maps" not in arts:
            st.warning(needs)
        else:
            z = _npz(arts["erders_maps"])
            names_e = [str(n) for n in z["ch_names"]]
            res = {
                k.split("__", 1)[1]: ErdersResult(
                    k.split("__", 1)[1], names_e, z["freqs"], z["times"], z[k]
                )
                for k in z
                if k.startswith("data__")
            }
            vlim = st.slider("Colour scale (±%)", 20, 150, 80, step=10)
            _fig(viz.plot_erders(res, vlim=float(vlim), title=label))
            st.caption("Blue = ERD (power decrease vs. -1…0 s baseline); red = ERS.")

    with tabs[3]:
        if "bandpower" not in arts:
            st.warning(needs)
        else:
            power = pd.read_csv(io.BytesIO(load_artifact_bytes(arts["bandpower"])))
            eeg = list(dict.fromkeys(power["channel"]))
            info = mne.create_info(eeg, run.sampling_rate, "eeg")
            info.set_montage(mne.channels.make_standard_montage(MONTAGE), on_missing="ignore")
            _fig(viz.plot_band_topomaps(power, info, title=label))
