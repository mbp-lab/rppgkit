"""Signal post-processing and heart-rate estimation for rPPG and PPG signals."""

import numpy as np
from scipy import signal as sps
from scipy import sparse
from scipy.sparse.linalg import spsolve


def detrend(x: np.ndarray, smoothing_lambda: float) -> np.ndarray:
    """Removes the slow trend with smoothness-priors detrending (Tarvainen et al., 2002).

    Computes x - (I + λ² DᵀD)⁻¹ x, with D the second-difference matrix, solved as a
    sparse system.

    Args:
        x: Signal, shape (T,).
        smoothing_lambda: λ; larger values give a smoother trend.

    Returns:
        The detrended signal, shape (T,).
    """
    n = len(x)
    second_diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    trend = spsolve(
        (
            sparse.identity(n) + smoothing_lambda**2 * second_diff.T @ second_diff
        ).tocsc(),
        x,
    )
    return x - trend


def bandpass(
    x: np.ndarray, fps: float, low_hz: float, high_hz: float, order: int
) -> np.ndarray:
    """Applies a zero-phase Butterworth band-pass filter (`filtfilt`).

    Args:
        x: Signal, shape (T,).
        fps: Sampling rate in Hz.
        low_hz: Lower cut-off frequency in Hz.
        high_hz: Upper cut-off frequency in Hz.
        order: Butterworth filter order.

    Returns:
        The filtered signal, float64, shape (T,).
    """
    b, a = sps.butter(order, [low_hz / fps * 2, high_hz / fps * 2], btype="bandpass")
    return sps.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def hr_periodogram_bpm(
    x: np.ndarray, fps: float, low_hz: float, high_hz: float, min_fft_length: int = 1
) -> float:
    """Estimates the heart rate from the highest periodogram peak in a band.

    The FFT length is the next power of two of len(x), at least `min_fft_length`
    (zero-padding for a finer frequency grid).

    Args:
        x: Signal, shape (T,).
        fps: Sampling rate in Hz.
        low_hz: Lower edge of the peak search band in Hz.
        high_hz: Upper edge of the peak search band in Hz.
        min_fft_length: Minimum FFT length.

    Returns:
        The heart rate in bpm, or NaN if the signal is not finite or has no power in
        the band (argmax would otherwise return the band's first bin, a
        plausible-looking heart rate).
    """
    if not np.all(np.isfinite(x)):
        return float("nan")
    freqs, power = sps.periodogram(
        x, fs=fps, nfft=_fft_length(len(x), min_fft_length), detrend=False
    )
    return _peak_bpm(freqs, power, low_hz, high_hz)


def hr_welch_bpm(
    x: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    segment_sec: float,
    min_fft_length: int = 1,
) -> float:
    """Estimates the heart rate from the highest Welch power spectrum peak in a band.

    Uses Hann segments with 50% overlap; the FFT length is the next power of two of
    the segment length, at least `min_fft_length`.

    Args:
        x: Signal, shape (T,).
        fps: Sampling rate in Hz.
        low_hz: Lower edge of the peak search band in Hz.
        high_hz: Upper edge of the peak search band in Hz.
        segment_sec: Segment length in seconds, at most len(x) samples.
        min_fft_length: Minimum FFT length.

    Returns:
        The heart rate in bpm, or NaN as in `hr_periodogram_bpm`.
    """
    if not np.all(np.isfinite(x)):
        return float("nan")
    nperseg = min(round(segment_sec * fps), len(x))
    freqs, power = sps.welch(
        x,
        fs=fps,
        nperseg=nperseg,
        nfft=_fft_length(nperseg, min_fft_length),
        detrend=False,
    )
    return _peak_bpm(freqs, power, low_hz, high_hz)


def _fft_length(n: int, min_fft_length: int) -> int:
    """Returns the next power of two of n, at least min_fft_length."""
    return max(1 << (n - 1).bit_length(), min_fft_length)


def _peak_bpm(
    freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float
) -> float:
    """Returns the frequency (bpm) of the highest power in the band; NaN if none."""
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.max(power[in_band]) > 0:
        return float("nan")
    return float(freqs[in_band][np.argmax(power[in_band])] * 60)
