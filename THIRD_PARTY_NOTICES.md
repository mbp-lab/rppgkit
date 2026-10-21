# Third-party notices

rppgkit is licensed under the GNU General Public License v3.0 only (see `LICENSE`).
It includes the third-party code and data below, each under its own license. All of
them can be distributed as part of a GPL-3.0 work.

## YOLO5Face face detector (GPL-3.0)

- Files: `src/rppgkit/roi/face/yolo5face/yolo.py`, `src/rppgkit/roi/face/yolo5face/data_ops.py`,
  `src/rppgkit/roi/face/yolo5face/model/yolo5face.yaml`
- Source: <https://github.com/deepcam-cn/yolov5-face> (Qi et al., "YOLO5Face: Why
  Reinventing a Face Detector", 2021), which builds on Ultralytics YOLOv5. Copied
  through rPPG-Toolbox (<https://github.com/ubicomplab/rPPG-Toolbox>), which
  redistributes it unchanged.
- License: GNU General Public License v3.0, the same text as `LICENSE`.
- Changes: listed in each file's docstring and marked `rppgkit:` in the code.

## YOLO5Face weights (GPL-3.0)

- File: `src/rppgkit/roi/face/yolo5face/ckpts/Y5sF_WFRGB.pth`
- Source: YOLO5Face-small checkpoint released with deepcam-cn/yolov5-face, trained on
  the WIDER FACE dataset (<http://shuoyang1213.me/WIDERFACE/>). Copied through
  rPPG-Toolbox.
- License: distributed with yolov5-face under GPL-3.0. Use of the WIDER FACE training data is
  subject to that dataset's own terms.

## PhysNet network (MIT)

- Code: the `PhysNetModel` class in `src/rppgkit/methods/physnet.py`
- Source: <https://github.com/ZitongYu/PhysNet> (Yu et al., BMVC 2019), copied through
  rPPG-Toolbox (`neural_methods/model/PhysNet.py`). The Toolbox's file header also
  states "Only for research purpose, and commercial use is not allowed."
- Changes: marked `rppgkit:` in the code.

```text
MIT License

Copyright (c) 2019 Zitong Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenCV Haar cascade (Intel License Agreement)

- File: `src/rppgkit/roi/face/haar/haarcascade_frontalface_default.xml`
- Source: OpenCV (<https://github.com/opencv/opencv>), created by Rainer Lienhart.
- License: Intel License Agreement for Open Source Computer Vision Library, a BSD-style
  license. The full notice is kept at the top of the file.

## Reimplemented from rPPG-Toolbox (no code copied)

The modules below are rppgkit's own code. They reproduce rPPG-Toolbox's formulas
(Liu et al., NeurIPS 2023) so that results are comparable:

- `src/rppgkit/signal_processing.py`: post-processing and heart-rate estimation
- `src/rppgkit/preprocessing/diff_normalize.py`, `src/rppgkit/preprocessing/standardize.py`

rPPG-Toolbox is licensed under the Responsible AI Source Code License (RAIL) v1.1.
rppgkit includes none of its own code.
