"""Render docs/img/architecture.png (used by the README, docs and demo deck)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#222222", "#6b6b6b"
FILL = {"data": "#E8F1FA", "core": "#EAF5EE", "access": "#FBF0E3", "ui": "#F3ECF7", "rt": "#FDECEC"}
EDGE = {"data": "#0072B2", "core": "#009E73", "access": "#D55E00", "ui": "#7B5EA7", "rt": "#B03A48"}

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, title, sub="", kind="core"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=FILL[kind],
            ec=EDGE[kind],
            lw=1.6,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.14 if sub else 0),
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    if sub:
        ax.text(
            x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.5, color=MUTED
        )


def arrow(p, q, rad=0.0, style="-|>", color=MUTED, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=13,
            lw=1.4,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
        )
    )


def label(x, y, text):
    ax.text(x, y, text, fontsize=9, fontweight="bold", color=MUTED, ha="left")


# --- lane titles
label(0.2, 6.62, "DATA IN")
label(3.05, 6.62, "ANALYSIS CORE  (eegkit.io / preprocess / epochs / analysis / viz)")
label(9.15, 6.62, "DATA ACCESS")

# --- data in
box(0.2, 5.35, 2.3, 0.95, "PhysioNet EEGBCI", "EDF, 109 subjects", "data")
box(0.2, 3.95, 2.3, 0.95, "BIDS dataset", "mne-bids · de-identified", "data")
box(0.2, 2.55, 2.3, 0.95, "Cache + outputs", "epochs.fif · CSV · PNG", "data")
arrow((1.35, 5.35), (1.35, 4.9))
arrow((1.35, 3.95), (1.35, 3.5), style="<|-|>")

# --- core pipeline
xs = [3.05, 4.55, 6.05, 7.55]
names = [
    ("io", "load · montage"),
    ("preprocess", "filter · bad ch · ref"),
    ("epochs", "events · reject"),
    ("analysis", "PSD · ERD · CSP"),
]
for x, (t, s) in zip(xs, names, strict=True):
    box(x, 5.35, 1.35, 0.95, t, s, "core")
for a, b in zip(xs[:-1], xs[1:], strict=True):
    arrow((a + 1.35, 5.825), (b, 5.825))
arrow((2.5, 5.825), (3.05, 5.825))
box(
    3.05,
    3.95,
    5.85,
    0.95,
    "pipeline  (batch · cached · per-subject failure isolation)",
    "stage hashes: config + input files  →  skip finished work",
    "core",
)
arrow((3.7, 5.35), (3.7, 4.9), style="<|-|>")
arrow((8.2, 5.35), (8.2, 4.9), style="<|-|>")
arrow((3.05, 4.425), (2.5, 4.425), style="<|-|>")
arrow((3.05, 4.1), (2.5, 3.2), rad=0.0)
box(3.05, 2.55, 2.75, 0.95, "viz", "PSD · ERD/ERS · topomaps · accuracy", "core")
box(6.15, 2.55, 2.75, 0.95, "deidentify", "HMAC date shift · pseudonym", "core")
arrow((4.4, 3.95), (4.4, 3.5))

# --- access
box(
    9.15,
    5.35,
    3.6,
    0.95,
    "Catalog (SQLite · SQLAlchemy)",
    "subjects · sessions · runs · channels · status",
    "access",
)
box(9.15, 3.95, 3.6, 0.95, "Storage  (ABC)", "LocalStorage · S3Storage", "access")
box(
    9.15,
    2.55,
    3.6,
    0.95,
    "REST API (FastAPI)",
    "filter · status · band power · epochs · /docs",
    "access",
)
arrow((10.95, 5.35), (10.95, 4.9), style="<|-|>")
arrow((10.95, 3.95), (10.95, 3.5), style="<|-|>")
arrow((8.9, 4.425), (9.15, 4.425), style="<|-|>")

# --- realtime
label(0.2, 2.1, "REAL TIME")
box(0.2, 0.5, 2.5, 1.2, "AcquisitionDevice", "abstract interface", "rt")
box(
    3.1,
    0.5,
    2.5,
    1.2,
    "MockAmplifier",
    "chunks @160 Hz · ring buffer\nmarkers · fault injection",
    "rt",
)
box(6.0, 0.5, 2.5, 1.2, "OnlineProcessor", "sliding-window mu/beta\ngap-aware", "rt")
arrow((2.7, 1.1), (3.1, 1.1), style="<|-", color=EDGE["rt"], ls="--")
arrow((5.6, 1.1), (6.0, 1.1), color=EDGE["rt"])

# --- UI
label(9.15, 2.1, "INTERFACES")
box(9.15, 0.5, 1.75, 1.2, "Streamlit", "Explorer · Live", "ui")
box(11.0, 0.5, 1.75, 1.2, "CLI", "Typer", "ui")
arrow((8.5, 1.1), (9.15, 1.1), color=EDGE["rt"])
arrow((10.6, 2.55), (10.6, 1.7), color=EDGE["access"])
arrow((11.9, 1.7), (11.9, 2.55), color=EDGE["ui"], ls="--")

fig.tight_layout()
fig.savefig("docs/img/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved")
