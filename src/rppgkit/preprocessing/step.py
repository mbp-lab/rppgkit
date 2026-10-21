"""Interface of a preprocessing step.

A step receives one complete ROI video and its raw PPG, `(frames, ppg)`, and returns
named `float32` arrays with one row per frame. Because it always receives the complete
video, any normalisation is per video, never per chunk. The output of a step is either
cached by `2_preprocess.py` (trainable methods) or computed during prediction
(methods without training). Label outputs (e.g. `ppg_standardized`) serve only as
training targets; evaluation uses the raw PPG from the ROI cache.
"""

from abc import ABC, abstractmethod

import numpy as np


class PreprocessingStep(ABC):
    """Abstract base class of a step that maps an ROI video and PPG to named arrays.

    Attributes:
        version: Incremented whenever the output changes. It is part of the cache
            key; `compute_cache_key` raises an error if a subclass does not set it.
    """

    version: int

    @abstractmethod
    def __call__(self, frames: np.ndarray, ppg: np.ndarray) -> dict[str, np.ndarray]:
        """Computes the arrays of the step for one complete video.

        Args:
            frames: ROI video, `uint8` RGB, shape `(T, H, W, 3)`.
            ppg: Raw PPG, `float32`, shape `(T,)`. Steps that use only the frames
                ignore it.

        Returns:
            Arrays by name, each `float32` with `T` rows and stored as `<name>.npy`. A
            name beginning with `"ppg"` denotes a label, which is removed for test
            recordings (`cached_videos.LABEL_PREFIX`); all other arrays must be
            computed from `frames` alone. Names must not coincide with those of the
            ROI cache arrays or of another step that a method reads.
        """
