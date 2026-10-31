"""Build and execute docs/tutorial.ipynb (needs the downloaded dataset in data/raw)."""

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Tutorial: from raw EEG to ERD/ERS and decoding

This notebook walks through the `eegkit` building blocks on **subject 1** of the PhysioNet
EEGBCI motor-imagery dataset (imagining closing the left vs. right fist, 3 runs, 64 channels,
160 Hz). Every step is also what `eegkit run` does for all subjects."""),
    code("""from pathlib import Path

import numpy as np

from eegkit import viz
from eegkit.analysis import (band_erders, band_power_table, compute_erders,
                             compute_psd_by_condition, decode_csp_lda)
from eegkit.config import Config
from eegkit.epochs import make_epochs, reject_epochs
from eegkit.io import load_subject
from eegkit.preprocess import preprocess

DATA = next(p for p in (Path("data/raw"), Path("../data/raw")) if p.exists())
cfg = Config()          # defaults; see configs/default.yaml
raw = load_subject(1, cfg.runs, DATA)
raw"""),
    md("## 1. Look at the raw signal\nCue markers (dashed) are the *left*/*right* imagery onsets."),
    code("""names = ["C3", "Cz", "C4", "FC3", "FC4", "Oz"]
picks = [raw.ch_names.index(c) for c in names]
sf = raw.info["sfreq"]
marks = [(o, "left" if d == "T1" else "right")
         for o, d in zip(raw.annotations.onset, raw.annotations.description) if d != "T0"]
viz.plot_traces(raw.get_data(picks=picks, start=0, stop=int(20 * sf)),
                np.arange(int(20 * sf)) / sf, names, 150, marks);"""),
    md("""## 2. Preprocess
Band-pass 1-40 Hz, 60 Hz notch, robust bad-channel detection (flat / high variance), average
reference. Every function returns a **copy**."""),
    code("""clean, bads = preprocess(raw, cfg.preprocess)
print("flat:", bads.flat, " noisy:", bads.noisy)"""),
    md(
        "## 3. Epoch and reject artifacts\nEpochs span -1.5...4.5 s around each cue; rejection looks at the sensorimotor channels."
    ),
    code("""epochs = make_epochs(clean, cfg.epochs.tmin, cfg.epochs.tmax, cfg.epochs.baseline)
epochs, report = reject_epochs(epochs, cfg.epochs.reject_uv, cfg.epochs.reject_channels)
print(report)
epochs"""),
    md(
        "## 4. Spectra and band power\nMu (8-13 Hz) at C3 drops during *right*-hand imagery - the contralateral hemisphere."
    ),
    code("""psds, freqs, ch = compute_psd_by_condition(epochs)
viz.plot_psd_comparison(psds, freqs, ch, ("C3", "C4"), cfg.analysis.bands);"""),
    code("""power = band_power_table(epochs, cfg.analysis.bands)
power.query("channel in ['C3', 'C4']").pivot_table(
    index=["band", "channel"], columns="condition", values="power").mul(1e12).round(1)"""),
    md("""## 5. ERD/ERS
Percent power change relative to the -1...0 s baseline: **negative = event-related
desynchronization (ERD)**. The brief red patch right after the cue is the evoked response,
not an ERS."""),
    code("""erd = compute_erders(epochs)
viz.plot_erders(erd);"""),
    code(
        """{cond: {k: round(v) for k, v in band_erders(r, (8, 13)).items()} for cond, r in erd.items()}"""
    ),
    md("## 6. Topographies"),
    code("""viz.plot_band_topomaps(power, epochs.info);"""),
    md("""## 7. Decoding left vs right
CSP spatial filters + LDA, 5-fold stratified cross-validation; filters are fitted inside each
training fold only."""),
    code("""res = decode_csp_lda(epochs, channels=cfg.analysis.decoding.channels)
print(f"accuracy {res.accuracy:.2f} +/- {res.std:.2f}  over {res.n_epochs} epochs", res.fold_scores)"""),
    md("""## 8. The same signal, online
Replay the recording through the simulated amplifier (8x faster than real time here) and
compute sliding-window band power at C3/C4 as a real-time system would."""),
    code("""import time
import matplotlib.pyplot as plt
from eegkit.realtime import MockAmplifier, OnlineProcessor

amp = MockAmplifier.from_raw(clean, speed=8.0, loop=False, chunk_samples=16)
proc = OnlineProcessor(amp.ch_names, amp.sfreq, window_s=1.0, hop_s=0.25)
out = []
amp.start(); proc.run(amp, out.append)       # blocks until the finite stream ends
t = np.array([r.time_s for r in out])
fig, ax = plt.subplots(figsize=(10, 3))
for chn, col in (("C3", "#009E73"), ("C4", "#CC79A7")):
    ax.plot(t, [r.powers[chn]["mu"] * 1e12 for r in out], color=col, label=f"{chn} mu")
for s, lab in amp._markers:
    ax.axvline(s / amp.sfreq, color=viz.COLORS[lab], alpha=.35, lw=1)
ax.set(xlabel="Stream time (s)", ylabel="Mu power (uV^2)", xlim=(0, 125)); ax.legend(frameon=False)
print(f"{proc.stats.windows} windows, mean latency {1e3*np.mean(proc.stats.latencies):.1f} ms,",
      "status:", proc.stats.status)"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "docs/tutorial.ipynb")
print("notebook executed")
