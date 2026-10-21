# Your first POS evaluation

In this tutorial, you will crop the faces from the UBFC-rPPG videos
([Bobbia et al., 2019](../reference/references.md#bobbia2019)) and estimate each
subject's heart rate with POS ([Wang et al., 2017](../reference/references.md#wang2017)).
You will finish with a `metrics.json` file that reports
the heart-rate error over all 42 subjects.

## Prerequisites

- Access to the cluster and its `gpu` partition.
- [uv](https://docs.astral.sh/uv/) installed.
- The UBFC-rPPG dataset. Each subject folder, for example `subject1/`, contains
  `vid.avi` and `ground_truth.txt`.

Steps 3 and 5 run a script. Use the SLURM variant on the cluster login node, or the
Local variant in a shell on a GPU node. With SLURM, the log of the script is written to
`multirun/<date>/<time>/.submitit/<jobid>/<jobid>_0_log.out` instead of the terminal.

## Step 1: Install rppgkit

On the login node, clone the repository and install the dependencies:

```bash
git clone https://github.com/mbp-lab/rppgkit.git
cd rppgkit
uv sync --locked
```

## Step 2: Set the data paths

rppgkit reads the dataset from one folder and writes its caches to another.

1. In `configs/dataset/ubfc_rppg.yaml`, set `root` to the folder that contains the
   subject folders.
2. In `configs/paths.yaml`, set `cache_root` to a folder that you own.

!!! warning
    Both files contain the paths of the maintainer. If you do not change them, your
    caches are written to the folder of the lab.

## Step 3: Extract the ROI

The ROI cache contains, for each video, the cropped face in every frame (the region of
interest, ROI) and the PPG signal resampled to one value per frame. All subsequent
steps read from it.

=== "SLURM"

    The command blocks until the job has finished. Start it in a `tmux` session so that
    it continues if the connection is lost:

    ```bash
    tmux new -s rppgkit
    uv run python scripts/1_extract_roi.py -m
    ```

=== "Local"

    Open a shell on a GPU node and run the script:

    ```bash
    srun --partition=gpu --gres=gpu:a40:1 --pty bash
    uv run python scripts/1_extract_roi.py
    ```

The log first reports the number of recordings found and the number to be built,
followed by one line per video:

```
dataset ubfc_rppg: 42 recordings
roi cache: <cache_root>/ubfc_rppg/roi/<key>
building 42/42 recordings (overwrite=False)
[1/42] ubfc_rppg/subject1/0: face detected in <n>/<n> frames tried (100.0%)
```

When the command has finished, list the cache:

```bash
ls <cache_root>/ubfc_rppg/roi/
```

The output is a single folder with a 12-character name. The name is computed from the
ROI settings, so different settings produce a different folder.

## Step 4: Create the split file

A split file assigns each subject to training, validation or testing. POS requires no
training, so all subjects are assigned to testing:

```bash
uv run python scripts/3_make_splits.py split=test_only
```

The log shows:

```
dataset ubfc_rppg: 42 subjects
fold 0: 0 train, 0 val, 42 test
wrote <repository>/splits/ubfc_rppg_test_only.json
```

## Step 5: Run POS

=== "SLURM"

    ```bash
    uv run python scripts/evaluate.py -m method=pos split=test_only
    ```

=== "Local"

    ```bash
    uv run python scripts/evaluate.py method=pos split=test_only
    ```

The script reads the ROI cache built in Step 3 and computes a pulse signal for each
video. It then estimates one heart rate from the pulse signal and one from the
reference PPG. The last two lines of the log report the metrics and the output folder:

```
metrics: {'mae_bpm': 1.52, ...}
results in <output folder>
```

The output folder is `outputs/<date>/<time>/` for a local run and
`multirun/<date>/<time>/0/` for a SLURM run.

## Step 6: Check the results

Open `metrics.json` in the output folder. The `metrics` section contains:

| Key | Meaning | Expected |
|---|---|---|
| `mae_bpm` | mean absolute heart-rate error, in beats per minute | about 1.5 |
| `pearson_r` | correlation between predicted and reference heart rates | about 0.98 |
| `n_videos` | videos included in the metrics | 42 |
| `n_failed_predictions` | videos for which POS gave no heart rate | 0 |
| `n_failed_ground_truths` | videos for which the reference PPG gave no heart rate | 0 |

`hr_per_video.csv` in the same folder lists the predicted and reference heart rate of
each video.

## Summary

You built an ROI cache, created a split file, and evaluated POS on all UBFC-rPPG
subjects. The settings for each step came from the files in `configs/`, and
`metrics.json` records which cache, split and settings produced the result.

Next steps:

- [How to extract the ROI](../how-to/extract-the-roi.md) with another detector or crop.
- [How to evaluate a method](../how-to/evaluate-a-method.md) on k-fold splits or time windows.
- [How rppgkit works](../how-it-works.md), for a description of the stages and caches.
