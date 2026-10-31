"""Build docs/eeg-toolkit-demo.pptx (12 slides, 10-minute interview walkthrough)."""

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

IMG = Path("docs/img")
INK, MUTED, BG_CARD = RGBColor(0x22, 0x22, 0x22), RGBColor(0x6B, 0x6B, 0x6B), RGBColor(0xF1, 0xF4, 0xF7)
BLUE, ORANGE, GREEN = RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73)
FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
summary = json.load(open("outputs/group/summary.json"))
ACC = summary["decoding"]["mean_accuracy"]
ERD = summary["erders_mean_pct"]


def text(slide, x, y, w, h, s, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font=FONT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = color
    return tb


def bullets(slide, x, y, w, items, size=20, gap=0.22):
    tb = slide.shapes.add_textbox(x, y, w, Inches(0.6) * len(items))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap * 72)
        r0 = p.add_run()
        r0.text = "■  "
        r0.font.size, r0.font.name, r0.font.color.rgb = Pt(size * 0.55), FONT, BLUE
        r = p.add_run()
        r.text = it
        r.font.size, r.font.name, r.font.color.rgb = Pt(size), FONT, INK
    return tb


def card(slide, x, y, w, h, title, body="", accent=BLUE, body_size=15, mono=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.05
    shp.fill.solid()
    shp.fill.fore_color.rgb = BG_CARD
    shp.line.fill.background()
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + Inches(0.18), Inches(0.07), Inches(0.42))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    text(slide, x + Inches(0.28), y + Inches(0.18), w - Inches(0.5), Inches(0.45), title, 20, True,
         anchor=MSO_ANCHOR.MIDDLE)
    if body:
        text(slide, x + Inches(0.28), y + Inches(0.85), w - Inches(0.5), h - Inches(1.0), body,
             body_size, color=INK, font="Menlo" if mono else FONT)


def picture(slide, path, x, y, w=None, h=None):
    """Place an image preserving aspect ratio inside a w x h box (either may be None)."""
    iw, ih = Image.open(path).size
    if w and h:
        scale = min(w / iw, h / ih)
    else:
        scale = (w / iw) if w else (h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x, y, Emu(pw), Emu(ph))


def new(title, notes, n):
    s = prs.slides.add_slide(BLANK)
    text(s, Inches(0.6), Inches(0.42), Inches(12.1), Inches(0.8), title, 32, True, anchor=MSO_ANCHOR.MIDDLE)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.25), Inches(1.1), Inches(0.06))
    ln.fill.solid()
    ln.fill.fore_color.rgb = BLUE
    ln.line.fill.background()
    text(s, Inches(0.6), Inches(7.0), Inches(6), Inches(0.3), "eeg-toolkit", 11, color=MUTED)
    text(s, Inches(11.7), Inches(7.0), Inches(1.0), Inches(0.3), str(n), 11, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(BLANK)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.5), H)
band.fill.solid()
band.fill.fore_color.rgb = BLUE
band.line.fill.background()
text(s, Inches(1.2), Inches(2.3), Inches(11), Inches(1.2), "eeg-toolkit", 60, True)
text(s, Inches(1.2), Inches(3.55), Inches(11), Inches(0.8),
     "Managing, accessing and analyzing human EEG data", 28, color=MUTED)
text(s, Inches(1.2), Inches(5.3), Inches(11), Inches(0.5),
     "Motor-imagery demo on PhysioNet EEGBCI  ·  github.com/stone-coding/eeg-toolkit", 16, color=MUTED)
s.notes_slide.notes_text_frame.text = (
    "This is a portfolio project I built to show how I would support a neuroscience lab with "
    "research software. It covers analysis, data management, access, privacy, real-time "
    "acquisition and a GUI. I will walk through it in about ten minutes, with real results from "
    "ten subjects of a public motor-imagery dataset.")

# 2 ------------------------------------------------------------------ motivation
s = new("Why labs need reusable data tools", (
    "In many labs each study re-implements loading, cleaning and QC in a one-off script, so results are hard to reproduce. "
    "Raw files also carry hidden identifiers and inconsistent naming. Experiments that run live add hardware and failure "
    "concerns. The toolkit packages these concerns once, with tests, so researchers spend time on science."), 2)
bullets(s, Inches(0.6), Inches(1.9), Inches(7.2), [
    "One-off scripts per study: slow, unreproducible",
    "Ad hoc file layouts, names and hidden identifiers",
    "Results nobody can query months later",
    "Live experiments: hardware drops, disconnects",
], 24, 0.3)
card(s, Inches(8.3), Inches(1.9), Inches(4.4), Inches(1.9), "Goal", "Reusable, tested, documented\nbuilding blocks for the lab.", BLUE, 17)
card(s, Inches(8.3), Inches(4.1), Inches(4.4), Inches(1.9), "Principle", "Correctness and reproducibility\nbefore feature count.", GREEN, 17)

# 3 ------------------------------------------------------------------ dataset
s = new("Dataset: EEGBCI motor imagery", (
    "I used the public PhysioNet EEGBCI dataset: 64 channels at 160 hertz, with subjects imagining closing their left or right fist. "
    "I analysed subjects one to ten, three runs each, about 45 trials per person. The well-known effect to look for is a "
    "drop in mu and beta power over the opposite side of the motor cortex, called event-related desynchronization."), 3)
bullets(s, Inches(0.6), Inches(1.9), Inches(4.9), [
    "PhysioNet EEGBCI, 109 subjects; 10 used",
    "64 channels, 160 Hz, 10-10 montage",
    "Imagine left vs. right fist (~45 trials each)",
    "Expected: contralateral mu/beta ERD",
], 20, 0.28)
pic = picture(s, IMG / "raw_traces.png", Inches(5.8), Inches(1.9), Inches(6.9), Inches(4.6))
text(s, Inches(5.8), pic.top + pic.height + Inches(0.1), Inches(6.9), Inches(0.3), "Subject 1, C3/Cz/C4 and neighbours; dashed lines = imagery cues", 12, color=MUTED)

# 4 ------------------------------------------------------------------ architecture
s = new("Architecture", (
    "The core analysis package has no dependency on the catalog, API or GUI, which keeps it easy to test and reuse. "
    "BIDS is the system of record, and the catalog can always be rebuilt from it. The real-time path sits behind an abstract "
    "device interface so a real amplifier driver can replace the simulator."), 4)
picture(s, IMG / "architecture.png", Inches(0.9), Inches(1.55), Inches(11.5), Inches(5.3))

# 5 ------------------------------------------------------------------ data management
s = new("Data management: BIDS, batch, caching", (
    "Raw downloads are converted to a BIDS dataset with mne-bids, with de-identification applied at that boundary. "
    "The batch pipeline processes every subject, isolates failures so one bad recording cannot stop the run, and writes per-subject and group outputs. "
    "Each stage stores a hash of its config and input files, so rerunning only recomputes what changed."), 5)
cw = Inches(3.9)
card(s, Inches(0.6), Inches(1.9), cw, Inches(3.0), "BIDS dataset", "mne-bids, EDF\nevents · channels · electrodes\nde-identified on ingest", BLUE, 19)
card(s, Inches(4.7), Inches(1.9), cw, Inches(3.0), "Batch pipeline", "preprocess → epochs → analysis\nper-subject + group CSV/figures\nfailures isolated per subject", GREEN, 19)
card(s, Inches(8.8), Inches(1.9), cw, Inches(3.0), "Content-hash cache", "config + input files → hash\nchange bands: only analysis reruns\nsecond run: all stages skipped", ORANGE, 19)
text(s, Inches(0.6), Inches(5.5), Inches(12.1), Inches(0.5),
     "eegkit download  →  to-bids  →  run  →  report --subject N", 20, True, BLUE, font="Menlo")

# 6 ------------------------------------------------------------------ data access
s = new("Data access: catalog + API", (
    "A SQLite catalog, via SQLAlchemy, indexes subjects, sessions, runs, channels, sampling rate, file paths and processing status. "
    "Files are addressed by key through a storage interface with a local backend and an S3-compatible backend. "
    "A FastAPI service lets you filter recordings, read status and band power, and download the epochs file, with OpenAPI docs generated automatically."), 6)
card(s, Inches(0.6), Inches(1.9), Inches(3.9), Inches(3.9), "Catalog", "subjects → sessions\n  → runs → channels\nprocessing status\nartifacts (results)\nSQLite + SQLAlchemy", BLUE, 18)
card(s, Inches(4.7), Inches(1.9), Inches(3.4), Inches(3.9), "Storage ABC", "files addressed by key\n\nLocalStorage\nS3Storage (MinIO/AWS)\n\ntraversal-safe keys", GREEN, 18)
card(s, Inches(8.3), Inches(1.9), Inches(4.4), Inches(3.9), "REST API", "GET /recordings?status=ok\n    &min_channels=64\nGET /subjects/{id}/status\nGET /subjects/{id}/bandpower\nGET /subjects/{id}/epochs\n/docs (OpenAPI)", ORANGE, 15, True)

# 7 ------------------------------------------------------------------ privacy
s = new("Privacy: de-identify at the boundary", (
    "Recording headers can hold names, birthdays, record numbers, experimenter and device serials, and the measurement date. "
    "I strip those, replace the ID with a keyed pseudonym, and shift dates by a per-subject offset from an HMAC with a secret salt, so intervals within a subject are preserved. "
    "Tests plant identifiers and prove none remain in the object, in the saved file bytes, or in any file of the BIDS dataset. This is a demonstration, not a HIPAA certification."), 7)
card(s, Inches(0.6), Inches(1.9), cw, Inches(2.9), "Strip", "names, birthday, MRN\nexperimenter, device serial\nfree-text description", BLUE, 19)
card(s, Inches(4.7), Inches(1.9), cw, Inches(2.9), "Shift", "date offset 1–3 years\nHMAC(salt, subject label)\nwithin-subject intervals kept", GREEN, 19)
card(s, Inches(8.8), Inches(1.9), cw, Inches(2.9), "Verify", "identifier scanner\ntests: object, FIF bytes,\nevery BIDS file", ORANGE, 19)
text(s, Inches(0.6), Inches(5.4), Inches(12.1), Inches(0.5),
     "A demonstration of the approach — not a HIPAA certification. Docs: privacy.md", 16, color=MUTED)

# 8 ------------------------------------------------------------------ results
s = new("Analysis results", (
    f"The time-frequency map shows percent power change after the cue. Imagining a right-hand movement suppresses mu power over C3, the left motor cortex, by about {abs(ERD['right/C3/mu']):.0f} percent, versus {abs(ERD['right/C4/mu']):.0f} percent at C4. "
    f"The topographies show the same central suppression. Decoding left versus right with CSP and LDA gives {ACC:.2f} on average with a wide spread: three subjects reach 0.7 or better and several are at chance, which is typical for motor imagery and for about 45 trials per person."), 8)
picture(s, IMG / "results/erders.png", Inches(0.5), Inches(1.5), Inches(6.5), Inches(4.7))
picture(s, IMG / "results/topomaps.png", Inches(7.3), Inches(1.4), Inches(4.6), Inches(2.9))
picture(s, IMG / "results/decoding.png", Inches(7.3), Inches(4.4), Inches(5.0), Inches(2.1))
text(s, Inches(0.5), Inches(6.3), Inches(6.5), Inches(0.6),
     f"Right imagery: C3 mu {ERD['right/C3/mu']:.1f}% vs C4 {ERD['right/C4/mu']:.1f}%  (group mean, n=10)", 14, True, BLUE)
text(s, Inches(7.3), Inches(6.6), Inches(5.5), Inches(0.3),
     f"CSP+LDA, 5-fold CV: mean {ACC:.2f}; 3 of 10 subjects ≥ 0.70", 14, True, BLUE)

# 9 ------------------------------------------------------------------ real time
s = new("Real-time acquisition simulation", (
    "A mock amplifier replays a recording in chunks at the real sampling rate, using absolute deadlines so timing does not drift, and writes into a ring buffer with event markers. "
    "An online processor computes sliding-window mu and beta power at C3 and C4; measured latency is about two milliseconds. "
    "Each failure mode is handled and tested: dropped chunks reset the window, a disconnect stops cleanly, and overflow drops the oldest data and counts it. A device interface lets a real driver replace the mock."), 9)
bx = [("AcquisitionDevice\n(abstract)", 0.6), ("MockAmplifier\nchunks · markers", 3.5), ("OnlineProcessor\nmu/beta power", 6.4)]
for label, x in bx:
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.95), Inches(2.5), Inches(1.2))
    sh.fill.solid()
    sh.fill.fore_color.rgb = BG_CARD
    sh.line.color.rgb = ORANGE
    sh.line.width = Pt(1.5)
    tf = sh.text_frame
    tf.word_wrap = True
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name, r.font.bold = Pt(16 if i == 0 else 13), FONT, i == 0
        r.font.color.rgb = INK if i == 0 else MUTED
for x in (3.1, 6.0):
    text(s, Inches(x), Inches(2.35), Inches(0.4), Inches(0.4), "→", 24, True, ORANGE, PP_ALIGN.CENTER)
text(s, Inches(9.3), Inches(1.95), Inches(3.4), Inches(0.8), "~2 ms", 40, True, GREEN)
text(s, Inches(9.3), Inches(2.75), Inches(3.4), Inches(0.4), "chunk → band power", 15, color=MUTED)
rows = [("Dropped chunk", "seq gap detected → window reset"),
        ("Device disconnect", "buffered data delivered, clean stop"),
        ("Consumer too slow", "oldest chunk dropped and counted")]
for i, (a, b) in enumerate(rows):
    y = Inches(3.7 + i * 0.95)
    card(s, Inches(0.6), y, Inches(12.1), Inches(0.8), "", "", ORANGE)
    text(s, Inches(0.9), y + Inches(0.18), Inches(3.8), Inches(0.45), a, 19, True)
    text(s, Inches(4.9), y + Inches(0.2), Inches(7.6), Inches(0.45), b, 18, color=INK)

# 10 ----------------------------------------------------------------- GUI
s = new("GUI: Explorer and Live monitor", (
    "The Streamlit app has two pages. The Explorer lets you pick a subject and run from the catalog and view raw traces, spectra, time-frequency maps and topographies. "
    "The Live monitor starts and stops the mock amplifier and shows the scrolling signal with cue markers and live band-power bars, plus latency and drop counters. "
    "There is also a button to simulate a disconnect, which is useful for demonstrating the failure handling."), 10)
Image.open(IMG / "gui_explorer_erders.png").crop((330, 90, 1450, 1000)).save(IMG / "_gui_explorer_crop.png")
Image.open(IMG / "gui_live.png").crop((330, 100, 1450, 690)).save(IMG / "_gui_live_crop.png")
picture(s, IMG / "_gui_explorer_crop.png", Inches(0.6), Inches(1.6), Inches(4.7), Inches(3.9))
picture(s, IMG / "_gui_live_crop.png", Inches(5.6), Inches(1.6), Inches(7.2), Inches(3.9))
text(s, Inches(0.6), Inches(5.65), Inches(4.7), Inches(0.4), "Explorer: catalog-driven, 4 views", 14, True, BLUE)
text(s, Inches(5.6), Inches(5.65), Inches(7.2), Inches(0.4), "Live monitor: simulated amplifier, online power", 14, True, BLUE)

# 11 ----------------------------------------------------------------- quality
s = new("Engineering quality", (
    "The test suite is mostly synthetic signals built with RawArray, so it needs no downloads: known sines for band power, injected artifacts for rejection, and fault injection for the amplifier. "
    "A few tests that need the real dataset are marked slow and skipped in CI. CI runs lint, type-checking and the tests with coverage on Python 3.10 and 3.12, and the docs site and Docker images build from the same repo."), 11)
tiles = [("119", "tests"), ("98%", "coverage"), ("ruff + mypy", "pre-commit, CI"), ("Docker", "+ MkDocs site")]
for i, (big, small) in enumerate(tiles):
    x = Inches(0.6 + i * 3.1)
    card(s, x, Inches(1.9), Inches(2.9), Inches(2.3), "", "", [BLUE, GREEN, ORANGE, BLUE][i])
    text(s, x + Inches(0.3), Inches(2.3), Inches(2.4), Inches(0.9), big, 34 if len(big) < 8 else 24, True, INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + Inches(0.3), Inches(3.3), Inches(2.4), Inches(0.5), small, 16, color=MUTED)
bullets(s, Inches(0.6), Inches(4.7), Inches(12), [
    "Synthetic signals: known sines, injected artifacts, fault injection",
    "Real-data tests marked slow, skipped in CI",
    "GitHub Actions: lint, type-check, tests on 3.10 + 3.12",
], 20, 0.2)

# 12 ----------------------------------------------------------------- next steps
s = new("Next steps", (
    "First, a real device driver: implement the acquisition interface for an amplifier SDK or Lab Streaming Layer, handling timestamps and clock sync. "
    "Second, TMS-EEG: pulse and decay artifact handling, and gating the online processor around pulses. "
    "Third, lab storage integration: point the storage interface at the lab's object store or NAS, add authentication and audit logging, and link to the lab's metadata system."), 12)
card(s, Inches(0.6), Inches(1.9), cw, Inches(3.3), "Real device drivers", "implement AcquisitionDevice\n(SDK / Lab Streaming Layer)\ntimestamps, clock sync,\nimpedance", BLUE, 18)
card(s, Inches(4.7), Inches(1.9), cw, Inches(3.3), "TMS-EEG artifacts", "pulse + decay artifact removal\ninterpolation, ICA / SOUND\nonline gating around pulses", GREEN, 18)
card(s, Inches(8.8), Inches(1.9), cw, Inches(3.3), "Lab storage", "S3 / NAS via Storage ABC\nauth + audit log\nlink to lab metadata system", ORANGE, 18)

prs.save("docs/eeg-toolkit-demo.pptx")
print("saved", len(prs.slides), "slides")
