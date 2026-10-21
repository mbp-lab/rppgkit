"""Heart-rate error metrics over per-video estimates (units in the names)."""

import numpy as np


def heart_rate_metrics(
    predicted_hr_bpm: np.ndarray, true_hr_bpm: np.ndarray
) -> dict[str, float]:
    """Computes heart-rate error metrics over per-video estimates.

    Args:
        predicted_hr_bpm: Predicted heart rate per video, bpm.
        true_hr_bpm: Ground-truth heart rate per video, bpm, same length. Only videos
            with a heart rate on both sides should be passed.

    Returns:
        `mae_bpm`, `rmse_bpm`, `mape_percent`, `pearson_r` (NaN for fewer than two
        videos) and `n_videos`.
    """
    predicted = np.asarray(predicted_hr_bpm, dtype=np.float64)
    true = np.asarray(true_hr_bpm, dtype=np.float64)
    error = predicted - true
    pearson = (
        float(np.corrcoef(predicted, true)[0, 1]) if len(true) > 1 else float("nan")
    )
    return {
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(np.abs(error) / true) * 100),
        "pearson_r": pearson,
        "n_videos": len(true),  # videos with a heart rate on both sides
    }
