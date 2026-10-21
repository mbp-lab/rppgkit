"""YOLO5Face behind rppgkit's detector interface.

The network and its pre/post-processing are the copied YOLO5Face files (yolo.py,
data_ops.py); this module only loads the bundled checkpoint and calls them in the
order of YOLO5Face's detection script: resize, letterbox, forward, NMS, rescale.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from rppgkit.roi.face.cropper import Box
from rppgkit.roi.face.yolo5face.data_ops import (
    check_img_size,
    letterbox,
    non_max_suppression_face,
    scale_coords,
)
from rppgkit.roi.face.yolo5face.yolo import Model

PACKAGE_DIR = Path(__file__).parent


class YOLO5FaceDetector:
    """YOLO5Face (small, WIDER FACE weights); returns the most confident face or None.

    Args:
        conf_threshold: Minimum confidence of a detection.
        iou_threshold: IoU threshold of non-maximum suppression.
        img_size: Side the frame is resized to (longer side) before inference.
        device: "auto" (first GPU if available, else CPU), "cpu" or e.g. "cuda:0".
    """

    version = 1

    def __init__(
        self,
        conf_threshold: float,
        iou_threshold: float,
        img_size: int,
        device: str,
    ) -> None:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        self.model = Model(cfg=str(PACKAGE_DIR / "model" / "yolo5face.yaml"), ch=3)
        weights = torch.load(
            PACKAGE_DIR / "ckpts" / "Y5sF_WFRGB.pth", map_location=self.device
        )
        self.model.load_state_dict(weights)
        self.model.to(self.device).float().eval()

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> Box | None:
        """Detects the most confident face in one frame.

        Args:
            frame: uint8 RGB frame, shape (H, W, 3).

        Returns:
            The face as (x, y, w, h) in frame pixels, or None if none is found.
        """
        h, w = frame.shape[:2]
        ratio = self.img_size / max(h, w)
        resized = frame
        if ratio != 1:
            interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(
                frame, (int(w * ratio), int(h * ratio)), interpolation=interpolation
            )
        size = check_img_size(self.img_size, s=self.model.stride.max())
        padded = letterbox(resized, new_shape=size)[0]

        image = torch.from_numpy(padded.transpose(2, 0, 1).copy()).to(self.device)
        image = image.float().unsqueeze(0) / 255.0
        prediction = self.model(image)[0]
        detections = non_max_suppression_face(
            prediction, self.conf_threshold, self.iou_threshold
        )[0]
        if len(detections) == 0:
            return None

        # NMS sorts by confidence: row 0 is the most confident face.
        box = scale_coords(image.shape[2:], detections[:1, :4], frame.shape).round()
        x1, y1, x2, y2 = (int(v) for v in box[0].tolist())
        return x1, y1, x2 - x1, y2 - y1
