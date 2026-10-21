"""OpenCV Haar cascade face detector; the cascade XML ships in this folder."""

from pathlib import Path

import cv2
import numpy as np

from rppgkit.roi.face.cropper import Box

# Ships next to this file; used when no cascade_path is given.
DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"


class HaarFaceDetector:
    """OpenCV frontal-face Haar cascade; returns the largest face or None.

    Args:
        cascade_path: Cascade XML file; None uses the bundled
            haarcascade_frontalface_default.xml, and a relative path is resolved
            against this folder, so the cache key is the same on every machine.
        scale_factor: `scaleFactor` of `detectMultiScale`.
        min_neighbors: `minNeighbors` of `detectMultiScale`.

    Raises:
        FileNotFoundError: If the cascade cannot be loaded.
    """

    version = 1

    def __init__(
        self,
        cascade_path: str | Path | None,
        scale_factor: float,
        min_neighbors: int,
    ) -> None:
        # None = the bundled cascade; relative paths are relative to this folder.
        cascade_path = Path(__file__).parent / (cascade_path or DEFAULT_CASCADE)
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise FileNotFoundError(f"cannot load Haar cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def __call__(self, frame: np.ndarray) -> Box | None:
        """Detects the largest face in one frame.

        Args:
            frame: uint8 RGB frame, shape (H, W, 3).

        Returns:
            The largest face as (x, y, w, h) in pixels, or None if none is found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)
