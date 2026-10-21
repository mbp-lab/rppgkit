# How to extract the ROI

This guide shows you how to build the ROI cache with a chosen detector and crop, how
to follow the job and how to rebuild the cache. The ROI cache contains the region of
interest, here the face, cropped from every frame.

## Build the cache with the default settings

On the login node, run:

```bash
uv run python scripts/1_extract_roi.py -m
```

This command uses YOLO5Face ([Qi et al., 2021](../reference/references.md#qi2021))
with a tight 36 × 36 crop (`configs/roi/yolo5face36_tight.yaml`) and submits one GPU
job. To run in your current shell instead, omit `-m`.

## Use another ROI configuration

Select a file from `configs/roi/` with `roi=` and the matching launcher with
`hydra/launcher=`:

| `roi=` | Detector | Crop | `hydra/launcher=` |
|---|---|---|---|
| `yolo5face36_tight` | YOLO5Face | detection box, 36 × 36 | `yolo5face` (default) |
| `yolo5face72` | YOLO5Face | square, 1.5 × the box, 72 × 72 | `yolo5face` (default) |
| `haar72` | OpenCV Haar cascade ([Viola and Jones, 2001](../reference/references.md#viola2001)) | square, 1.5 × the box, 72 × 72 | `haar72` |

For example, to use the Haar detector:

```bash
uv run python scripts/1_extract_roi.py -m roi=haar72 hydra/launcher=haar72
```

To use another dataset, add `dataset=<name>`.

## Change a crop setting

To test a setting, override it on the command line:

```bash
uv run python scripts/1_extract_roi.py -m roi.output_size=72 roi.box_scale=1.2
```

To use a setting in reported results, copy a file in `configs/roi/`, change the copy,
and select it with `roi=<new name>`.

Each combination of settings is written to its own cache folder; existing caches are
retained.

## Follow a SLURM job

The job writes its files to `multirun/<date>/<time>/.submitit/<jobid>/`:

- `<jobid>_0_log.out`: the log of the script.
- `<jobid>_0_log.err`: warnings and tracebacks.

To check the job, use the following commands:

```bash
squeue -u $USER                                                  # queued and running jobs
tail -f multirun/<date>/<time>/.submitit/<jobid>/<jobid>_0_log.out   # follow the log
sacct -j <jobid>                                                 # state and exit code
seff <jobid>                                                     # CPU and memory used
```

To change the requested resources, edit the launcher file in `configs/hydra/launcher/`.

## Resume an interrupted build

Run the same command again. The script builds only the videos that are not yet in the
cache and reports their number at the start of the log:

```
building 12/42 recordings (overwrite=False)
```

## Rebuild an existing cache

To rebuild every video, for example after a code fix that did not change the
configuration, add `overwrite=true`:

```bash
uv run python scripts/1_extract_roi.py -m overwrite=true
```

Then rebuild the preprocessing caches made from this ROI cache in the same way:

```bash
uv run python scripts/2_preprocess.py -m overwrite=true
```

## Use the cache in later steps

Pass the same `dataset`, `roi` and `roi.*` settings to `2_preprocess.py` and
`evaluate.py`. These scripts locate the cache from those settings and stop with
`roi cache not found` if they differ. See [Caches and keys](../how-it-works.md#caches-and-keys).

If a build was interrupted, these scripts stop before any processing with
`roi cache <folder> is missing <n>/<m> recordings`. To resolve this,
[resume the build](#resume-an-interrupted-build).
