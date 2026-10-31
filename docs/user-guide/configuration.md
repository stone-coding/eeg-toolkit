# Configuration

All commands accept `-c/--config path.yaml`. Every key is optional; defaults live in
[`configs/default.yaml`](https://github.com/stone-coding/eeg-toolkit/blob/main/configs/default.yaml).
The file is validated with pydantic: unknown keys, reversed ranges, bad subject ids or run
numbers abort with a readable message and exit code 2.

| Section | Key | Default | Meaning |
|---|---|---|---|
| top | `subjects` | 1-10 | subject ids (1-109) |
| top | `runs` | `[4, 8, 12]` | left/right-fist imagery runs |
| `paths` | `raw_dir`, `bids_root`, `output_dir`, `cache_dir`, `catalog_db`, `storage_root` | `data/...`, `outputs/...` | locations |
| `privacy` | `deidentify` | `true` | strip identifiers / shift dates when converting to BIDS |
| `preprocess` | `l_freq`, `h_freq`, `notch_freqs` | 1, 40, [60] | filters (Hz) |
| `preprocess` | `flat_std_v`, `noisy_z`, `interpolate_bads` | 1e-7, 4.0, true | bad-channel detection |
| `epochs` | `tmin`, `tmax`, `baseline` | -1.5, 4.5, (-1.5, 0) | epoch window (s) |
| `epochs` | `reject_uv`, `reject_channels` | 300, sensorimotor ROI | peak-to-peak rejection |
| `analysis` | `bands`, `band_window` | mu 8-13, beta 13-30; 0.5-3.5 s | band power |
| `analysis` | `erders_*` | C3/C4, 6-30 Hz, baseline -1...0 s | ERD/ERS maps |
| `analysis.decoding` | `tmin`, `tmax`, `fmin`, `fmax`, `n_components`, `n_splits`, `seed` | 0.5-2.5 s, 8-30 Hz, 4, 5, 42 | CSP+LDA |

!!! note "Why the rejection uses a channel subset"
    EEGBCI is high-amplitude data: blinks give > 300 µV at frontal sites in almost every
    epoch. Rejection and decoding are therefore evaluated on 15 medial sensorimotor channels
    (`SENSORIMOTOR` in `eegkit.config`); the motor rhythms analysed are not affected by frontal
    blinks, and discarding every epoch with a blink would leave too few trials.
