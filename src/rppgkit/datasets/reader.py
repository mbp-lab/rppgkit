"""Reader interface and shared helpers; readers only know file layouts.

Each dataset subclasses `DatasetReader` and implements `recording_keys` and `read`.
The helpers below cover steps that don't depend on the dataset (decoding a video
file, resampling PPG to one sample per frame); readers call the ones that fit their format.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from rppgkit.recording import RecordingKey, VideoRecord


class DatasetReader(ABC):
    """Turns one dataset's files into `VideoRecord`s.

    Subclasses set `version` as a class attribute, set `name` from the dataset config
    in `__init__`, and implement `list_recording_keys` and `read`.

    Attributes:
        name: `RecordingKey.dataset` of all recordings, and the cache folder name.
        version: Bumped on any change to what `read` returns; part of the cache key.
    """

    name: str
    version: int

    @abstractmethod
    def list_recording_keys(self) -> list[RecordingKey]:
        """Lists the recordings on disk without decoding anything.

        Returns:
            The keys of all recordings, sorted.
        """

    @abstractmethod
    def read(self, key: RecordingKey) -> VideoRecord:
        """Loads one recording and resamples its PPG to one sample per frame.

        Args:
            key: A key returned by `list_recording_keys`.

        Returns:
            The recording's frames, PPG, fps and meta.
        """


def read_video_file(path: Path) -> tuple[np.ndarray, float]:
    """Decodes a video file to RGB frames.

    Args:
        path: Path to a video file OpenCV can read.

    Returns:
        A tuple `(frames, fps)`: frames as uint8 RGB, shape (T, H, W, 3), and the
        frame rate stored in the file.

    Raises:
        OSError: If the file cannot be opened or no frame can be decoded.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        frame_read, frame = capture.read()
        if not frame_read:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise OSError(f"no frames decoded from: {path}")
    return np.stack(frames), fps


def resample_ppg(ppg: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resamples PPG to one sample per frame.

    Resamples by index, ignoring timestamps, so it assumes the PPG and the video start
    and end together; a duration mismatch is not detected.

    Args:
        ppg: PPG samples, shape (N,).
        n_frames: Number of video frames T.

    Returns:
        The resampled PPG, float32, shape (T,).
    """
    old_positions = np.linspace(0.0, 1.0, len(ppg))
    new_positions = np.linspace(0.0, 1.0, n_frames)
    return np.interp(new_positions, old_positions, ppg).astype(np.float32)
