"""Data contract between dataset readers and the rest of the pipeline.

A reader converts the files of one dataset into `VideoRecord` instances. No component
downstream of the reader depends on the dataset from which a record originates.
Readers only parse and resample; they never crop, resize, normalise or filter.

Sample `i` of `frames` and `ppg` is at time `i / fps` seconds. A recording is
identified by a `RecordingKey` (dataset, subject, recording) and may originate from a
video file, an image folder or a `.mat` file. String forms such as
`"ubfc_rppg/subject1/0"` appear only at boundaries (folder names, split files, logs)
and are converted with `str(key)` and `RecordingKey.parse`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class RecordingKey:
    """Identity of one recording, given by dataset, subject and recording.

    Each part is non-empty and contains no `"/"`, so the string form
    `"<dataset>/<subject>/<recording>"` is unambiguous and can be parsed.

    Attributes:
        dataset: Dataset name, e.g. `"ubfc_rppg"`.
        subject: Subject within the dataset, e.g. `"subject1"`.
        recording: Recording of the subject, e.g. `"0"` (UBFC-rPPG) or `"T1"` (a
            PURE task). Attributes such as task or lighting belong in
            `VideoRecord.meta`.
    """

    dataset: str
    subject: str
    recording: str

    def __post_init__(self) -> None:
        """Checks that each part is non-empty and contains no `"/"`.

        Raises:
            ValueError: If a part is empty or contains `"/"`.
        """
        for name in ("dataset", "subject", "recording"):
            part = getattr(self, name)
            if not part or "/" in part:
                raise ValueError(f"{name} must be non-empty without '/', got {part!r}")

    @property
    def subject_id(self) -> str:
        """Subject ID `"<dataset>/<subject>"`, by which splits group recordings."""
        return f"{self.dataset}/{self.subject}"

    def __str__(self) -> str:
        """Returns the string form `"<dataset>/<subject>/<recording>"`.

        The string form is used in folder names, split files, CSV files and logs.
        """
        return f"{self.dataset}/{self.subject}/{self.recording}"

    @classmethod
    def parse(cls, text: str) -> "RecordingKey":
        """Parses the string form produced by `str(key)`.

        This is the only conversion from the string form to a key.

        Args:
            text: A key of the form `"<dataset>/<subject>/<recording>"`.

        Returns:
            The `RecordingKey` with these three parts.

        Raises:
            ValueError: If `text` does not consist of exactly three parts separated
                by `"/"`, or if a part is empty.
        """
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"expected '<dataset>/<subject>/<recording>', got {text!r}"
            )
        return cls(*parts)


@dataclass(frozen=True, eq=False)
class VideoRecord:
    """One video with its ground-truth PPG resampled to one sample per frame.

    Sample `i` of `frames` and `ppg` is at time `i / fps` seconds.

    Attributes:
        key: Identity of the recording.
        frames: Video frames, `uint8` RGB, shape `(T, H, W, 3)`.
        ppg: Contact PPG, `float32`, shape `(T,)`, one sample per frame.
        fps: Frame rate, in frames per second; positive and finite.
        meta: Dataset-specific attributes, e.g. `meta["scenario"]`.
    """

    key: RecordingKey
    frames: np.ndarray
    ppg: np.ndarray
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks the dtypes and shapes, the equal length of frames and PPG, and fps.

        The key is validated on its own construction.

        Raises:
            ValueError: If `frames` is not `uint8` with shape `(T, H, W, 3)`, `ppg` is
                not `float32` with shape `(T,)`, their lengths differ, or `fps` is
                not positive and finite.
        """
        if (
            self.frames.dtype != np.uint8
            or self.frames.ndim != 4
            or self.frames.shape[-1] != 3
        ):
            raise ValueError(
                f"frames must be uint8 (T, H, W, 3), got {self.frames.dtype} {self.frames.shape}"
            )
        if self.ppg.dtype != np.float32 or self.ppg.ndim != 1:
            raise ValueError(
                f"ppg must be float32 (T,), got {self.ppg.dtype} {self.ppg.shape}"
            )
        if len(self.ppg) != len(self.frames):
            raise ValueError(
                f"ppg has {len(self.ppg)} samples but there are {len(self.frames)} frames"
            )
        if not (np.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps must be positive and finite, got {self.fps}")
