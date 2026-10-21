# How to evaluate a method

This guide shows you how to evaluate a method that requires no training, such as POS
([Wang et al., 2017](../reference/references.md#wang2017)), and how to read the
results. It assumes that an ROI cache exists; see
[How to extract the ROI](extract-the-roi.md).

## Create a split file

Create each kind of split file once per dataset:

```bash
uv run python scripts/3_make_splits.py split=test_only   # one fold, all subjects in test
uv run python scripts/3_make_splits.py split=kfold       # five folds, each subject tested once
```

The files are written to `splits/<dataset>_<kind>.json`. An existing file is never
overwritten. To create a variant, set another path with `split.path=splits/<name>.json`,
and use the same path when you evaluate.

## Evaluate on all subjects

```bash
uv run python scripts/evaluate.py -m method=pos split=test_only
```

To run in your current shell instead of a SLURM job, omit `-m`.

## Evaluate on k-fold splits

To evaluate one fold:

```bash
uv run python scripts/evaluate.py -m method=pos split=kfold split.fold=0
```

To evaluate all five folds as separate jobs:

```bash
uv run python scripts/evaluate.py -m method=pos split=kfold split.fold=0,1,2,3,4
```

## Use a non-default ROI cache

Pass the same `roi` and `roi.*` settings that you used to build the cache:

```bash
uv run python scripts/evaluate.py -m method=pos roi=haar72
```

## Change the evaluation settings

The settings are defined in `configs/evaluation/default.yaml` and apply equally to the
prediction and the reference PPG.

To estimate one heart rate per 10 s window, with a new window every 5 s:

```bash
uv run python scripts/evaluate.py -m method=pos evaluation.window_sec=10 evaluation.window_stride_sec=5
```

To use Welch's method ([Welch, 1967](../reference/references.md#welch1967)) instead of
a single periodogram:

```bash
uv run python scripts/evaluate.py -m method=pos evaluation.hr_estimator=welch
```

## Read the results

Each run writes to its own folder: `outputs/<date>/<time>/`, or
`multirun/<date>/<time>/<n>/` with `-m`. The folder contains:

- `hr_per_video.csv`: the predicted and reference heart rate of each video or window.
- `metrics.json`: the metrics, and the cache keys, split file hash and settings that
  produced them.

Before you compare runs, check `n_failed_predictions` and `n_failed_ground_truths` in
`metrics.json`. A video without a heart rate shows `nan` in the CSV file and is excluded
from the metrics.

## Combine the folds of a k-fold evaluation

Group the `metrics.json` files by `experiment_key` and average their metrics. All folds
of one experiment have the same key. A change to the method, ROI cache, split file or
evaluation settings gives a different key.
