"""Analysis: spectral band power, ERD/ERS time-frequency, and CSP+LDA decoding."""

from eegkit.analysis.decoding import DecodingResult, decode_csp_lda
from eegkit.analysis.erders import ErdersResult, band_erders, compute_erders
from eegkit.analysis.spectral import band_power, band_power_table, compute_psd_by_condition

__all__ = [
    "DecodingResult",
    "ErdersResult",
    "band_erders",
    "band_power",
    "band_power_table",
    "compute_erders",
    "compute_psd_by_condition",
    "decode_csp_lda",
]
