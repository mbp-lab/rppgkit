"""Face cropping as an ROI method: face detection, box cropping and resizing.

The procedure is identical for all datasets and methods. The number of frames is
unchanged, so the face video remains aligned with the PPG. A failed detection reuses
the last detected box, or the first detected box for failures at the start. The crop
settings (box shape, scale, size, detection frequency) strongly affect the results and
are therefore all configuration options.

A face detector is any callable that receives one `uint8` RGB frame and returns the
face box as a `Box` `(x, y, w, h)` in pixels, where `(x, y)` is the top-left corner, or
`None`. Each detector has its own subpackage (e.g. `haar/`) with its code and weight
files.
"""

from collections.abc import Callable
from dataclasses import replace

import cv2
import numpy as np

from rppgkit.recording import VideoRecord

Box = tuple[int, int, int, int]  # x, y of the top-left corner, w, h, in pixels


class FaceCropper:
    """Crops the face from every frame of a video.

    The face is detected every `detect_every` frames. A box around it, either square or
    of the shape of the detection and scaled by `box_scale`, is cropped and resized to
    `output_size` × `output_size` pixels with `cv2.INTER_AREA`.

    Args:
        detector: Callable that receives one `uint8` RGB frame and returns the face
            box `(x, y, w, h)` in pixels, or `None` if no face is found.
        output_size: Side length of the square output frames, in pixels.
        box_scale: Factor applied to the detection box before cropping.
        square_box: If `True`, a square of side `box_scale` times the longer side of
            the box is cropped, with the same centre and shifted to fit within the
            frame. If `False`, the scaled detection box itself is cropped, clipped to
            the frame and stretched to the output size, without preserving the
            aspect ratio.
        detect_every: The detector runs on frames `0, N, 2N, ...`, and each box is
            used for the `N` frames from its detection onwards. If `N` is at least the
            video length, one box from the first frame is used for the whole video.

    Raises:
        ValueError: If `detect_every` is less than 1.
    """

    version = 1

    def __init__(
        self,
        detector: Callable[[np.ndarray], Box | None],
        output_size: int,
        box_scale: float,
        square_box: bool,
        detect_every: int,
    ) -> None:
        if detect_every < 1:
            raise ValueError(f"detect_every must be >= 1, got {detect_every}")
        self.detector = detector
        self.output_size = output_size
        self.box_scale = box_scale
        self.square_box = square_box
        self.detect_every = detect_every

    def __call__(self, record: VideoRecord) -> tuple[VideoRecord, np.ndarray, dict]:
        """Crops the face from one recording.

        A failed detection reuses the last detected box, or the first detected box for
        failures at the start.

        Args:
            record: The full-frame recording.

        Returns:
            A tuple `(face_record, boxes, stats)`. `face_record` is the recording with
            face frames of shape `(T, output_size, output_size, 3)` and unchanged PPG,
            key and metadata. `boxes` contains the crop box `(x, y, w, h)` of each
            frame, `int32` with shape `(T, 4)`. `stats` contains `n_frames`,
            `n_detections` (the number of frames on which the detector ran),
            `n_failed` and `failed_frames`.

        Raises:
            ValueError: If no face is detected in any of the frames examined.
        """
        n_frames = len(record.frames)
        detected_at = range(0, n_frames, self.detect_every)
        detections = [self.detector(record.frames[i]) for i in detected_at]
        failed = [i for i, box in zip(detected_at, detections) if box is None]
        if len(failed) == len(detections):
            raise ValueError(
                f"{record.key}: no face detected in any of the {len(failed)} frames tried"
            )

        height, width = record.frames.shape[1:3]
        detection_boxes = [
            self._crop_region(box, height, width)
            for box in _fill_failed_detections(detections)
        ]
        # Each box holds for detect_every frames, until the next detection.
        boxes = np.repeat(
            np.array(detection_boxes, dtype=np.int32), self.detect_every, axis=0
        )
        boxes = boxes[:n_frames]
        faces = np.stack(
            [
                cv2.resize(
                    frame[y : y + h, x : x + w],
                    (self.output_size, self.output_size),
                    interpolation=cv2.INTER_AREA,
                )
                for frame, (x, y, w, h) in zip(record.frames, boxes, strict=True)
            ]
        )
        stats = {
            "n_frames": n_frames,
            "n_detections": len(detections),  # frames the detector ran on
            "n_failed": len(failed),
            "failed_frames": failed,
        }
        return replace(record, frames=faces), boxes, stats

    def _crop_region(self, box: Box, height: int, width: int) -> Box:
        """Returns the crop box for one detection, square or scaled and clipped."""
        if self.square_box:
            return self._square_region(box, height, width)
        x, y, w, h = box
        center_x, center_y = x + w / 2, y + h / 2
        w, h = self.box_scale * w, self.box_scale * h
        x0 = max(0, round(center_x - w / 2))
        y0 = max(0, round(center_y - h / 2))
        x1 = min(width, round(center_x + w / 2))
        y1 = min(height, round(center_y + h / 2))
        return x0, y0, x1 - x0, y1 - y0

    def _square_region(self, box: Box, height: int, width: int) -> Box:
        """Returns a square of `box_scale` times the longer side, within the frame."""
        x, y, w, h = box
        side = min(round(self.box_scale * max(w, h)), height, width)
        x0 = int(np.clip(round(x + w / 2 - side / 2), 0, width - side))
        y0 = int(np.clip(round(y + h / 2 - side / 2), 0, height - side))
        return x0, y0, side, side


def _fill_failed_detections(detections: list[Box | None]) -> list[Box]:
    """Replaces `None` with the last detected box, or the first for leading failures."""
    first = next(box for box in detections if box is not None)
    filled, last = [], first
    for box in detections:
        last = box if box is not None else last
        filled.append(last)
    return filled
