"""POS: plane-orthogonal-to-skin rPPG.

Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017. Written from the
paper; no detrending or band-pass inside, so prediction and ground truth get identical
post-processing in evaluation.
"""

import math

import numpy as np

from rppgkit.cached_videos import CachedVideo
from rppgkit.methods.method import Method
from rppgkit.preprocessing.rgb_mean import RGBMean

# Projection onto the plane orthogonal to the skin tone (paper, Eq. 10).
PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos_pulse(rgb: np.ndarray, fps: float, window_sec: float) -> np.ndarray:
    """Computes the POS pulse signal from the mean face colour.

    Runs POS on every sliding window of `window_sec` (stride one frame) and
    overlap-adds the zero-mean window outputs.

    Args:
        rgb: Mean face colour per frame, shape (T, 3).
        fps: Frame rate in frames per second.
        window_sec: Window length in seconds (the paper's l = 1.6 s at 20 fps).

    Returns:
        The pulse signal, float64, shape (T,).
    """
    n_frames = len(rgb)
    length = math.ceil(window_sec * fps)
    pulse = np.zeros(n_frames)
    for start in range(n_frames - length + 1):
        window = rgb[start : start + length]
        normalized = window / window.mean(axis=0)  # temporal normalization per colour
        s1, s2 = PROJECTION @ normalized.T
        h = s1 + (s1.std() / (s2.std() + 1e-9)) * s2  # alpha tuning
        pulse[start : start + length] += h - h.mean()
    return pulse


class POS(Method):
    """POS on the mean face colour, computed on the fly from the ROI cache.

    Args:
        window_sec: Sliding-window length in seconds, passed to `pos_pulse`.
    """

    needs = ("frames",)
    version = 1

    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec

    def predict(self, video: CachedVideo) -> dict[str, np.ndarray]:
        """Predicts the rPPG signal of one whole video; needs no training or labels.

        Args:
            video: One recording with its face `frames`.

        Returns:
            `{"rppg": float64 (T,)}`.
        """
        rgb = RGBMean()(video.arrays["frames"], None)["rgb_mean"]
        return {"rppg": pos_pulse(rgb, video.fps, self.window_sec)}
