# Comparison with rPPG-Toolbox

rPPG-Toolbox ([Liu et al., 2023](references.md#liu2023),
<https://github.com/ubicomplab/rPPG-Toolbox>) is the reference implementation against
which we verify the defaults of rppgkit. The tables below list each correspondence and
each deliberate difference. The code and configuration files of rppgkit describe only
its own behaviour; the comparison is maintained on this page alone.

rppgkit contains none of the original code of rPPG-Toolbox (RAIL licence). Code that
rPPG-Toolbox redistributes from other projects under compatible licences is copied as
documented in `THIRD_PARTY_NOTICES.md`.

## Face cropping

| rppgkit | rPPG-Toolbox |
| --- | --- |
| YOLO5Face network (`roi/face/yolo5face/yolo.py`) | Verbatim copy of the YOLO5Face code that rPPG-Toolbox redistributes. Changes are marked `rppgkit:`. |
| YOLO5Face pre/post-processing (`roi/face/yolo5face/data_ops.py`) | Verbatim copy of `utils/data_ops.py`. |
| YOLO5Face weights (`Y5sF_WFRGB.pth`) | Same checkpoint. |
| Detected boxes | Same boxes as the rPPG-Toolbox detector. |
| `box_scale: 1.5` | rPPG-Toolbox default. |

## Preprocessing

| rppgkit | rPPG-Toolbox |
| --- | --- |
| `Standardize` | Same as `standardized_data` / `standardized_label`: one scalar mean and standard deviation per video. |
| `DiffNormalize` | Same as `diff_normalize_data` / `diff_normalize_label`. |
| `DiffNormalize.to_pulse` (integrates a prediction) | Same as the integration in the rPPG-Toolbox post-processing before detrending. |

## Evaluation

`src/rppgkit/signal_processing.py` reimplements the formulas of
`evaluation/post_process.py` in rPPG-Toolbox, and the defaults in
`configs/evaluation/default.yaml` reproduce its behaviour, so heart rates and metrics
are comparable. We verified the correspondence numerically.

| rppgkit | rPPG-Toolbox |
| --- | --- |
| `detrend_lambda: 100`, `low_hz: 0.75`, `high_hz: 2.5`, `filter_order: 1`, `hr_estimator: periodogram` | Same post-processing and heart-rate estimation. |
| Detrending solves the smoothness-priors system as a sparse linear system. | Solves the same system by inverting a dense T × T matrix. |

## PhysNet training

Support for PhysNet training is in development. The table describes the current
implementation.

| rppgkit | rPPG-Toolbox |
| --- | --- |
| `PhysNetModel` | `neural_methods/model/PhysNet.py` (`PhysNet_padding_Encoder_Decoder_MAX`) in rPPG-Toolbox, itself copied from the code of the original author. The layer names are retained, so its checkpoints load without modification. |
| `PhysNetModel` returns only the rPPG signal. | Also returns three feature maps. |
| `losses.NegativePearson`, vectorised over the batch | `neural_methods/loss/PhysNetNegPearsonLoss.py` iterates over the batch. The rPPG-Toolbox trainer additionally z-scores prediction and target over the whole batch beforehand; we omit this step because a single affine map applied to all samples leaves the Pearson r of each sample unchanged. |
| Input and target: per-video Standardized (`configs/method/physnet.yaml`) | DiffNormalized. To train as rPPG-Toolbox does, run with `+experiment=physnet_diffnormalized`. With Standardized targets, the output has the shape of a PPG signal and requires no integration before evaluation. |
| Adam, learning rate 9e-3, OneCycleLR peaking at that rate | Same. |
| `max_epochs: 30`, `batch_size: 4` (`configs/trainer/default.yaml`) | Same. |
| `chunk_length: 128`, `chunk_stride: 128` (no overlap) | Same. |
| Test videos: the last T mod `chunk_length` frames are dropped. | Same. |
| `augmentations: []`; augmentations can be added per experiment. | No augmentations. |
