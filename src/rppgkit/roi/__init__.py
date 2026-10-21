"""Extraction of the region of interest (ROI) from every frame of a video.

Each ROI method has its own subpackage, e.g. `face/`, which detects the face, crops a
box and resizes it. An ROI method is a callable that receives a `VideoRecord` and
returns `(roi_record, boxes, stats)`, as `FaceCropper` does. `1_extract_roi.py` writes
these outputs to the ROI cache, on which the subsequent stages rely as follows:

- `frames.npy`: `roi_record.frames`, `uint8` RGB, shape `(T, H, W, 3)`. `T` is the
  length of the input video, as no frame is dropped or resampled. `H × W` is fixed by
  the configuration and identical for all recordings, because methods batch chunks of
  different videos.
- `ppg.npy`: `roi_record.ppg`, the unchanged PPG of the reader, which serves as the
  ground truth in evaluation.
- `boxes.npy`: `boxes`, the crop box of each frame, with `T` rows.
- `meta.json`: the unchanged `fps` and `meta` of the reader, and `stats`, the
  diagnostics of the method. `1_extract_roi.py` logs `stats["n_detections"]` and
  `stats["n_failed"]`, so every method must provide both.

An additional array must have `T` rows and a name that is neither `frames` nor begins
with `ppg`, since such names denote labels, which are removed for test recordings. Its
name must also differ from the array names of all preprocessing steps, because both are
loaded together.

An ROI is a rectangular image patch. Every pixel of `frames` is used (e.g. `RGBMean`
averages all of them), so a method cannot mark pixels as lying outside the ROI.
"""
