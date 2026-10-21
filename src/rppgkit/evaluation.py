"""Stage 4: from predicted signals to per-video (or per-window) heart rates and metrics.

Prediction and ground truth go through the identical post-processing (detrend →
band-pass → spectral-peak heart rate), per whole video or per window; every step is an
EvaluationSettings field, so evaluation choices can be varied and compared. Ground truth is
the raw PPG from the ROI cache, trimmed to the prediction's length (a method returns
a signal aligned to frame 0, T' <= T).
A flat or non-finite signal has no heart rate (NaN); metrics use only the videos
where both sides have one, and count the failures of each side separately.
Knows nothing about methods, chunks or preprocessing.
"""

from dataclasses import dataclass

import numpy as np

from rppgkit.metrics import heart_rate_metrics
from rppgkit.recording import RecordingKey
from rppgkit.signal_processing import (
    bandpass,
    detrend,
    hr_periodogram_bpm,
    hr_welch_bpm,
)

HR_ESTIMATORS = ("periodogram", "welch")


@dataclass(frozen=True)
class EvaluationSettings:
    """Post-processing, heart-rate estimation and windowing settings.

    Applied to prediction and ground truth alike. Every setting comes from the config
    (configs/evaluation/); there are no defaults here, so a key can't hide one.

    Attributes:
        version: Bumped on any change to the heart rates; part of `experiment_key`.
        detrend_lambda: λ of the smoothness-priors detrending.
        low_hz: Lower cut-off of the band-pass filter in Hz.
        high_hz: Upper cut-off of the band-pass filter in Hz.
        filter_order: Butterworth order of the band-pass filter.
        search_low_hz: Lower edge of the heart-rate peak search in Hz; None uses
            `low_hz`.
        search_high_hz: Upper edge of the heart-rate peak search in Hz; None uses
            `high_hz`.
        hr_estimator: "periodogram" (whole window) or "welch" (averaged segments of
            `welch_segment_sec`).
        welch_segment_sec: Welch segment length in seconds.
        min_fft_length: Minimum FFT length for zero-padding; 1 means the next power
            of two of the signal length.
        window_sec: Window length in seconds, each window post-processed on its own;
            None gives one heart rate per whole video.
        window_stride_sec: Seconds between window starts; None uses `window_sec`
            (no overlap). Requires `window_sec`.

    Raises:
        ValueError: If `hr_estimator` is unknown, a length is not positive, or
            `window_stride_sec` is set without `window_sec`.
    """

    version = 1

    detrend_lambda: float
    low_hz: float
    high_hz: float
    filter_order: int
    search_low_hz: float | None
    search_high_hz: float | None
    hr_estimator: str
    welch_segment_sec: float
    min_fft_length: int
    window_sec: float | None
    window_stride_sec: float | None

    def __post_init__(self) -> None:
        """Rejects an unknown hr_estimator and non-positive or orphaned window settings."""
        if self.hr_estimator not in HR_ESTIMATORS:
            raise ValueError(f"hr_estimator must be one of {sorted(HR_ESTIMATORS)}")
        for name in ("window_sec", "window_stride_sec", "welch_segment_sec"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.window_stride_sec is not None and self.window_sec is None:
            raise ValueError("window_stride_sec needs window_sec")

    def heart_rate_bpm(self, x: np.ndarray, fps: float) -> float:
        """Detrends, band-pass filters and estimates the heart rate of one signal.

        Args:
            x: rPPG or PPG signal, shape (T,).
            fps: Sampling rate in Hz.

        Returns:
            The heart rate in bpm, or NaN if `x` is flat or not finite, or has no power
            in the search band.
        """
        x = np.asarray(x, dtype=np.float64)
        # Checked on the raw signal: detrending a constant leaves ~1e-13 residue,
        # which would still produce a peak.
        if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
            return float("nan")
        detrended = detrend(x, self.detrend_lambda)
        filtered = bandpass(
            detrended, fps, self.low_hz, self.high_hz, self.filter_order
        )
        low = self.low_hz if self.search_low_hz is None else self.search_low_hz
        high = self.high_hz if self.search_high_hz is None else self.search_high_hz
        # Other types of methods should be added here
        if self.hr_estimator == "welch":
            return hr_welch_bpm(
                filtered, fps, low, high, self.welch_segment_sec, self.min_fft_length
            )
        return hr_periodogram_bpm(filtered, fps, low, high, self.min_fft_length)

    def window_bounds(self, n_samples: int, fps: float) -> list[tuple[int, int]]:
        """Returns the sample ranges of the evaluation windows.

        The last window may be shorter; it is dropped if too short to filter, and no
        window starts after one reached the end.

        Args:
            n_samples: Length of the signal.
            fps: Sampling rate in Hz.

        Returns:
            `(start, stop)` pairs; `[(0, n_samples)]` if `window_sec` is None.
        """
        if self.window_sec is None:
            return [(0, n_samples)]
        length = round(self.window_sec * fps)
        stride_sec = (
            self.window_sec
            if self.window_stride_sec is None
            else self.window_stride_sec
        )
        stride = round(stride_sec * fps)
        # filtfilt needs more samples than its padlen: 3 × (2·order + 1) coefficients.
        min_samples = 3 * (2 * self.filter_order + 1) + 1
        windows = []
        for start in range(0, n_samples, stride):
            # min is there to get the last samples when they are smaller than the window_size (window_sec)
            stop = min(start + length, n_samples)
            if stop - start >= min_samples:
                windows.append((start, stop))
            if stop == n_samples:
                break
        return windows


def evaluate(
    predictions: dict[RecordingKey, np.ndarray],
    ground_truth: dict[RecordingKey, np.ndarray],
    fps: dict[RecordingKey, float],
    settings: EvaluationSettings,
) -> tuple[list[dict], dict[str, float]]:
    """Compares predicted and ground-truth heart rates per video or per window.

    Args:
        predictions: Predicted rPPG signal per recording, aligned to frame 0.
        ground_truth: Raw PPG per recording, at least as long as its prediction; only
            its first len(prediction) samples are used.
        fps: Frame rate per recording.
        settings: Post-processing, estimator and windowing.

    Returns:
        A tuple `(rows, metrics)`. `rows` has one dict per video, or per window if
        `settings.window_sec` is set, with `recording`, `window_start_sec`,
        `predicted_hr_bpm` and `true_hr_bpm` (NaN where there is no heart rate).
        `metrics` holds the `heart_rate_metrics` over rows with a heart rate on both
        sides, `n_windows` when windowed, `n_failed_predictions` and
        `n_failed_ground_truths`.

    Raises:
        ValueError: If a prediction is longer than its ground truth, or no row has a
            heart rate on both sides.
    """
    rows = []
    for key, prediction in predictions.items():
        reference = ground_truth[key]
        if len(prediction) > len(reference):
            raise ValueError(
                f"{key}: prediction has {len(prediction)} samples, "
                f"ground truth {len(reference)}"
            )
        for start, stop in settings.window_bounds(len(prediction), fps[key]):
            rows.append(
                {
                    "recording": str(key),
                    "window_start_sec": start / fps[key],
                    "predicted_hr_bpm": settings.heart_rate_bpm(
                        prediction[start:stop], fps[key]
                    ),
                    "true_hr_bpm": settings.heart_rate_bpm(
                        reference[start:stop], fps[key]
                    ),
                }
            )
    predicted_hr = np.array([row["predicted_hr_bpm"] for row in rows])
    true_hr = np.array([row["true_hr_bpm"] for row in rows])
    valid = np.isfinite(predicted_hr) & np.isfinite(true_hr)
    if not valid.any():
        raise ValueError(f"no video with a heart rate on both sides ({len(rows)} rows)")
    metrics = heart_rate_metrics(predicted_hr[valid], true_hr[valid])
    if settings.window_sec is not None:
        # heart_rate_metrics counted windows; n_videos stays the number of videos.
        metrics["n_windows"] = metrics["n_videos"]
        metrics["n_videos"] = len(
            {row["recording"] for row, is_valid in zip(rows, valid) if is_valid}
        )
    # A flat prediction is a method failure; a flat ground truth is a data problem.
    metrics["n_failed_predictions"] = int(np.sum(~np.isfinite(predicted_hr)))
    metrics["n_failed_ground_truths"] = int(np.sum(~np.isfinite(true_hr)))
    return rows, metrics
