"""Figures: PSD comparison, ERD/ERS maps, band-power topomaps, decoding accuracy."""

from __future__ import annotations

import matplotlib

# headless-safe default, but leave Jupyter's inline backend alone
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from eegkit.analysis.erders import ErdersResult  # noqa: E402

# Okabe-Ito pair (colorblind-safe) for the two imagery classes.
COLORS = {"left": "#0072B2", "right": "#D55E00"}
INK = "#333333"
MUTED = "#8a8a8a"
BAND_SHADE = {"mu": "#E8E8E8", "beta": "#F3F3F3"}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 150,
    }
)


def plot_psd_comparison(
    psds: dict[str, np.ndarray],
    freqs: np.ndarray,
    ch_names: list[str],
    channels: tuple[str, ...] = ("C3", "C4"),
    bands: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
) -> Figure:
    """PSD (µV²/Hz, log axis) per condition at the given channels; bands shaded."""
    bands = bands or {"mu": (8, 13), "beta": (13, 30)}
    fig, axes = plt.subplots(
        1, len(channels), figsize=(4.6 * len(channels), 3.6), sharey=True, squeeze=False
    )
    for ax, ch in zip(axes[0], channels, strict=True):
        idx = ch_names.index(ch)
        for name, (lo, hi) in bands.items():
            ax.axvspan(lo, hi, color=BAND_SHADE.get(name, "#EEE"), zorder=0)
            ax.text(
                (lo + hi) / 2,
                1.0,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color=MUTED,
                fontsize=9,
            )
        for cond, psd in psds.items():
            ax.semilogy(freqs, psd[idx] * 1e12, color=COLORS.get(cond), lw=2, label=cond)
        ax.set(xlabel="Frequency (Hz)", title=ch)
        ax.margins(x=0)
    axes[0][0].set_ylabel("PSD (µV²/Hz)")
    axes[0][-1].legend(frameon=False, title="imagery")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_erders(
    results: dict[str, ErdersResult], vlim: float = 80.0, title: str | None = None
) -> Figure:
    """ERD/ERS maps: rows = conditions, columns = channels. Diverging scale centered on 0%."""
    conds = list(results)
    chans = results[conds[0]].ch_names
    fig, axes = plt.subplots(
        len(conds),
        len(chans),
        figsize=(4.8 * len(chans), 3.0 * len(conds)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    im = None
    for i, cond in enumerate(conds):
        r = results[cond]
        for j, ch in enumerate(chans):
            ax = axes[i][j]
            im = ax.imshow(
                r.data[r.ch_names.index(ch)],
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
                interpolation="bilinear",
                extent=(r.times[0], r.times[-1], r.freqs[0], r.freqs[-1]),
            )
            ax.axvline(0, color=INK, lw=1, ls="--")
            ax.set_title(f"{ch} · {cond} imagery")
            if i == len(conds) - 1:
                ax.set_xlabel("Time from cue (s)")
            if j == 0:
                ax.set_ylabel("Frequency (Hz)")
    assert im is not None
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("ERD (−) / ERS (+)  %")
    if title:
        fig.suptitle(title)
    return fig


def plot_band_topomaps(
    power: pd.DataFrame,
    info: mne.Info,
    bands: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Topomaps of log band power per condition plus the right-vs-left contrast.

    ``power`` is long-format (``condition, channel, band, power``). The last column shows
    10·log10(right/left) in dB, on a diverging scale centered on zero.
    """
    bands = bands or list(power["band"].unique())
    conds = [c for c in ("left", "right") if c in set(power["condition"])]
    eeg = [info.ch_names[i] for i in mne.pick_types(info, eeg=True)]
    pinfo = mne.pick_info(info, mne.pick_channels(info.ch_names, eeg, ordered=True))
    fig, axes = plt.subplots(
        len(bands),
        len(conds) + 1,
        figsize=(3.2 * (len(conds) + 1), 3.1 * len(bands)),
        squeeze=False,
    )

    def vec(cond: str, band: str) -> np.ndarray:
        d = power[(power.condition == cond) & (power.band == band)].set_index("channel")
        return d.loc[eeg, "power"].to_numpy()

    for i, band in enumerate(bands):
        logp = {c: 10 * np.log10(vec(c, band) * 1e12) for c in conds}
        lo = min(v.min() for v in logp.values())
        hi = max(v.max() for v in logp.values())
        for j, cond in enumerate(conds):
            im, _ = mne.viz.plot_topomap(
                logp[cond],
                pinfo,
                axes=axes[i][j],
                show=False,
                cmap="viridis",
                vlim=(lo, hi),
                contours=4,
            )
            axes[i][j].set_title(f"{band} · {cond} (dB µV²)")
            fig.colorbar(im, ax=axes[i][j], shrink=0.7)
        if len(conds) == 2:
            diff = 10 * np.log10(vec("right", band) / vec("left", band))
            lim = max(abs(diff).max(), 1e-6)
            im, _ = mne.viz.plot_topomap(
                diff,
                pinfo,
                axes=axes[i][-1],
                show=False,
                cmap="RdBu_r",
                vlim=(-lim, lim),
                contours=4,
            )
            axes[i][-1].set_title(f"{band} · right − left (dB)")
            fig.colorbar(im, ax=axes[i][-1], shrink=0.7)
        else:
            axes[i][-1].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_decoding_accuracy(df: pd.DataFrame, title: str | None = None) -> Figure:
    """Bar chart of CV accuracy per subject (error bars = fold SD) with chance line.

    ``df`` needs columns ``subject, accuracy`` and optionally ``std``.
    """
    df = df.sort_values("subject")
    fig, ax = plt.subplots(figsize=(max(5.5, 0.7 * len(df) + 2), 3.8))
    x = np.arange(len(df))
    ax.bar(
        x, df["accuracy"], color="#0072B2", width=0.6, yerr=df.get("std"), capsize=3, ecolor=MUTED
    )
    ax.axhline(0.5, color=INK, lw=1, ls="--")
    ax.annotate(
        "chance",
        (1.01, 0.5),
        xycoords=("axes fraction", "data"),
        va="center",
        color=MUTED,
        fontsize=9,
        annotation_clip=False,
    )
    mean = float(df["accuracy"].mean())
    ax.axhline(mean, color="#D55E00", lw=1.5)
    ax.annotate(
        f"mean {mean:.2f}",
        (1.01, mean),
        xycoords=("axes fraction", "data"),
        va="center",
        color="#D55E00",
        fontsize=9,
        annotation_clip=False,
    )
    ax.set_xticks(x, [f"S{int(s)}" for s in df["subject"]])
    ax.set(ylabel="CV accuracy", ylim=(0, 1.05), xlabel="Subject")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_traces(
    data: np.ndarray,
    times: np.ndarray,
    ch_names: list[str],
    scale_uv: float = 100.0,
    markers: list[tuple[float, str]] | None = None,
    title: str | None = None,
) -> Figure:
    """Stacked EEG traces (``data`` in volts, one row per channel) with event markers."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(ch_names) + 1.2))
    step = scale_uv * 1e-6
    for i in range(len(ch_names)):
        y = data[i] - data[i].mean()
        ax.plot(times, y + (len(ch_names) - 1 - i) * step, color=INK, lw=0.7)
    ax.set_yticks([(len(ch_names) - 1 - i) * step for i in range(len(ch_names))], ch_names)
    ax.set_ylim(-step, len(ch_names) * step)
    ax.set_xlim(times[0], times[-1])
    seen: set[str] = set()
    for t, lab in markers or []:
        if times[0] <= t <= times[-1]:
            ax.axvline(
                t, color=COLORS.get(lab, MUTED), lw=1.5, ls="--", label=None if lab in seen else lab
            )
            seen.add(lab)
    if seen:
        ax.legend(frameon=False, loc="upper right", ncol=len(seen))
    ax.set(xlabel="Time (s)")
    ax.text(1.005, 0.02, f"{scale_uv:g} µV", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


CHANNEL_COLORS = ("#009E73", "#CC79A7", "#E69F00", "#56B4E9")  # Okabe-Ito, distinct from L/R


def plot_band_bars(
    powers: dict[str, dict[str, float]], title: str | None = None, ymax_uv2: float | None = None
) -> Figure:
    """Grouped bars of band power (µV²): one group per band, one bar per channel."""
    chans = list(powers)
    bands = list(next(iter(powers.values())))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    w = 0.8 / len(chans)
    for i, ch in enumerate(chans):
        vals = [powers[ch][b] * 1e12 for b in bands]
        xs = np.arange(len(bands)) + (i - (len(chans) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.9, color=CHANNEL_COLORS[i % 4], label=ch)
    ax.set_xticks(np.arange(len(bands)), bands)
    ax.set(ylabel="Band power (µV²)")
    if ymax_uv2:
        ax.set_ylim(0, ymax_uv2)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
