"""Interface of a method: whole videos in, named outputs ({"rppg": ...}) out.

A method declares the arrays it needs from the caches and returns a signal aligned to
frame 0 with T' <= T samples; chunking and stitching stay inside the method, so
evaluation never needs to know about them. Trainable methods (step 6) add hooks named
like Lightning's (training_step, validation_step, configure_optimizers).
"""

from abc import ABC, abstractmethod

import numpy as np

from rppgkit.cached_videos import CachedVideo


class Method(ABC):
    """Predicts named per-frame outputs for one whole video.

    Attributes:
        needs: Names of the arrays read from the ROI and preprocessing caches.
        version: Bumped on any change to the output. Part of `experiment_key` in
            metrics.json; `compute_cache_key` raises if a subclass does not set it.
    """

    needs: tuple[str, ...]
    version: int

    @abstractmethod
    def predict(self, video: CachedVideo) -> dict[str, np.ndarray]:
        """Predicts the rPPG signal of one whole test video.

        Args:
            video: The arrays in `needs` for one recording, without labels.

        Returns:
            `{"rppg": float (T',)}`, aligned to frame 0, with T' <= T.
        """
