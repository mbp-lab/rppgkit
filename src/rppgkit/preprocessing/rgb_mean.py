"""Mean face colour per frame, for methods without training (e.g. POS)."""

import numpy as np

from rppgkit.preprocessing.step import PreprocessingStep


class RGBMean(PreprocessingStep):
    """Spatial mean of the R, G and B channels over all face pixels of each frame."""

    version = 1

    def __call__(self, frames: np.ndarray, ppg: np.ndarray) -> dict[str, np.ndarray]:
        """Averages each colour channel over all face pixels of each frame.

        Args:
            frames: Face video, `uint8` RGB, shape `(T, H, W, 3)`.
            ppg: Unused.

        Returns:
            `{"rgb_mean": rgb_mean}`, with `rgb_mean` of type `float32` and shape
            `(T, 3)`.
        """
        return {
            "rgb_mean": frames.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        }
