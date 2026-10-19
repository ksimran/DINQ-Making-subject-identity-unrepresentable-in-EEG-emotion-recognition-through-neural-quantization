"""Band decomposition: zero-phase MNE FIR filter bank (default) or wavelet packets
(ablation), applied to the continuous trial before windowing."""

from __future__ import annotations

import numpy as np


def bandpass_filterbank(
    eeg: np.ndarray,
    fs: float,
    bands: tuple[tuple[str, float, float], ...],
) -> np.ndarray:
    """(C, T) -> (B, C, T): zero-phase band-pass into each band via MNE.

    h_freq is clipped below Nyquist so a band like gamma 31-50 stays valid at
    low sampling rates.
    """
    import mne

    eeg = np.asarray(eeg, dtype=np.float64)
    nyq = fs / 2.0
    out = []
    for _, lo, hi in bands:
        hi_eff = min(hi, nyq - 1.0)
        xb = mne.filter.filter_data(
            eeg, sfreq=fs, l_freq=lo, h_freq=hi_eff,
            method="fir", phase="zero", verbose="ERROR",
        )
        out.append(xb)
    return np.stack(out, axis=0)  # (B, C, T)


def wavelet_packet_bands(
    eeg: np.ndarray,
    fs: float,
    bands: tuple[tuple[str, float, float], ...],
    wavelet: str = "db4",
    max_level: int | None = None,
) -> np.ndarray:
    """(C, T) -> (B, C, T): reconstruct each band from WPT nodes (granularity ablation).

    Each band is rebuilt from the wavelet-packet nodes whose frequency support
    overlaps the band's [lo, hi]. Returned band signals are reconstructed to the
    original length so downstream windowing/covariance is identical to the
    filter-bank path.
    """
    import pywt

    eeg = np.asarray(eeg, dtype=np.float64)
    C, T = eeg.shape
    if max_level is None:
        max_level = int(np.floor(np.log2(fs / 2.0 / 2.0)))  # ~2 Hz leaves
    max_level = max(1, max_level)

    out = np.zeros((len(bands), C, T), dtype=np.float64)
    nyq = fs / 2.0
    for ci in range(C):
        wp = pywt.WaveletPacket(data=eeg[ci], wavelet=wavelet, mode="symmetric",
                                maxlevel=max_level)
        nodes = wp.get_level(max_level, order="freq")
        n_nodes = len(nodes)
        node_bw = nyq / n_nodes
        for bi, (_, lo, hi) in enumerate(bands):
            rec = pywt.WaveletPacket(data=None, wavelet=wavelet, mode="symmetric",
                                     maxlevel=max_level)
            used = False
            for ni, node in enumerate(nodes):
                f_lo, f_hi = ni * node_bw, (ni + 1) * node_bw
                if f_hi > lo and f_lo < hi:  # overlap
                    rec[node.path] = node.data
                    used = True
            if used:
                sig = rec.reconstruct(update=False)[:T]
                out[bi, ci, : len(sig)] = sig
    return out


def decompose_bands(
    eeg: np.ndarray,
    fs: float,
    bands: tuple[tuple[str, float, float], ...],
    method: str = "filterbank",
    **kwargs,
) -> np.ndarray:
    """Dispatch band decomposition. Returns (B, C, T)."""
    if method == "filterbank":
        return bandpass_filterbank(eeg, fs, bands)
    if method == "wavelet_packet":
        return wavelet_packet_bands(eeg, fs, bands, **kwargs)
    raise ValueError(f"unknown band method {method!r}")
