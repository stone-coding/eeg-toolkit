"""Live monitor page: stream a recording through the MockAmplifier and show online band power."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from eegkit import viz
from eegkit.bids import load_subject_bids
from eegkit.gui.common import context
from eegkit.realtime import DeviceState, MockAmplifier, OnlineProcessor, PowerResult
from eegkit.synthetic import make_synthetic_raw

SHOW = ("C3", "Cz", "C4")


def _build(source: str, speed: float) -> tuple[MockAmplifier, OnlineProcessor, list[PowerResult]]:
    cfg, _, _ = context()
    if source == "Synthetic demo signal":
        raw = make_synthetic_raw(n_per_class=6)
    else:
        sub = int(source.split("-")[1])
        raw = load_subject_bids(sub, cfg.runs, cfg.paths.bids_root)
    amp = MockAmplifier.from_raw(raw, speed=speed, chunk_samples=16, buffer_seconds=30)
    proc = OnlineProcessor(amp.ch_names, amp.sfreq, window_s=1.0, hop_s=0.25)
    results: list[PowerResult] = []
    amp.start()

    def on_result(r: PowerResult) -> None:
        results.append(r)
        del results[:-200]  # keep the latest 200 windows

    proc.start(amp, on_result)
    return amp, proc, results


def _teardown() -> None:
    proc, amp = st.session_state.pop("proc", None), st.session_state.pop("amp", None)
    if proc:
        proc.stop()
    if amp:
        amp.stop()
    st.session_state.pop("results", None)


@st.fragment(run_every=0.3)
def _live_view(seconds: int) -> None:
    amp: MockAmplifier | None = st.session_state.get("amp")
    proc: OnlineProcessor | None = st.session_state.get("proc")
    if amp is None or proc is None:
        st.info("Press **Start** to stream.")
        return
    results: list[PowerResult] = st.session_state["results"]
    stats = amp.stats()

    a, b, c, d = st.columns(4)
    a.metric("Device", amp.state.value)
    b.metric("Processor", proc.stats.status)
    c.metric("Latency", f"{1e3 * proc.stats.last_latency_s:.1f} ms")
    d.metric("Dropped / overflowed", f"{proc.stats.dropped_chunks} / {stats.chunks_overflowed}")
    if amp.state is DeviceState.DISCONNECTED:
        st.error("Device disconnected. Press Stop, then Start to reconnect.")

    n = int(seconds * amp.sfreq)
    names = [c_ for c_ in SHOW if c_ in amp.ch_names]
    seg = amp.buffer.latest(n)
    if seg.shape[1] > 1 and names:
        rows = [amp.ch_names.index(c_) for c_ in names]
        end = amp.buffer.total_written / amp.sfreq
        times = end - (seg.shape[1] - 1 - np.arange(seg.shape[1])) / amp.sfreq
        marks = [(m / amp.sfreq, lab) for m, lab in list(amp.markers)]
        fig = viz.plot_traces(seg[rows], times, names, 100.0, marks, "Scrolling signal")
        left, right = st.columns([3, 1.4])
        left.pyplot(fig, clear_figure=True)
        plt.close(fig)
        if results:
            fig2 = viz.plot_band_bars(results[-1].powers, "Band power (last 1 s)")
            right.pyplot(fig2, clear_figure=True)
            plt.close(fig2)
        else:
            right.caption("Waiting for the first full 1 s window...")


def render() -> None:
    cfg, catalog, _ = context()
    st.title("Live monitor")
    st.caption(
        "A recorded session is replayed through a simulated amplifier at its real sampling rate; "
        "mu/beta power at C3/C4 is computed online over a sliding 1 s window."
    )
    labels = ["Synthetic demo signal"] + [
        f"sub-{s.label.split('-')[1]}" for s in catalog.list_subjects()
    ]
    with st.sidebar:
        source = st.selectbox("Source", labels) or labels[0]
        speed = st.select_slider(
            "Speed", [0.5, 1.0, 2.0], value=1.0, format_func=lambda x: f"{x:g}x"
        )
        seconds = st.slider("Display window (s)", 3, 20, 8)
    running = "amp" in st.session_state

    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("Start", type="primary", disabled=running):
        try:
            amp, proc, results = _build(source, speed)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start: {exc}")
        else:
            st.session_state.update(amp=amp, proc=proc, results=results)
            st.rerun()
    if c2.button("Stop", disabled=not running):
        _teardown()
        st.rerun()
    if running and c3.button("Simulate disconnect"):
        amp = st.session_state["amp"]
        amp.inject_disconnect(amp.stats().chunks_emitted + 1)
        time.sleep(0.2)
    streaming = running and st.session_state["amp"].state is DeviceState.STREAMING
    if streaming and st.button("Insert marker"):
        st.session_state["amp"].send_marker("user")

    _live_view(seconds)
