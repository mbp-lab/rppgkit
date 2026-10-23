# How to track runs with MLflow

This guide shows you how to set up MLflow tracking, open the MLflow web interface on
the cluster and find, inspect and compare the logged runs.

Each time you run a script, MLflow records its settings, results and log files as a
run. With `-m`, each SLURM job is recorded as a separate run. The runs of each stage are
stored together in one MLflow experiment, for all datasets:

| Script | MLflow experiment |
|---|---|
| `1_extract_roi.py` | `roi_extraction` |
| `2_preprocess.py` | `preprocessing` |
| `3_make_splits.py` | `splits` |
| `evaluate.py` | `evaluation` |
| `train.py` | `training` |

The run folders under `outputs/` and `multirun/` are not changed; MLflow stores copies
of their files.

## Set your MLflow folder

Set `mlflow_root` in `configs/paths.yaml` to a folder of your own, next to `cache_root`:

```yaml
mlflow_root: /data/<your_folder>/mlflow
```

The folder holds the database `mlflow.db` and the logged files under `artifacts/`. It is
created on the first run. If the folder cannot be created or written, every script
stops before it starts its work.

## Run a script without tracking

To run a script without MLflow, add `hydra/callbacks=none`:

```bash
uv run python scripts/evaluate.py method=pos split=test_only hydra/callbacks=none
```

## Open the MLflow web interface

1. On the login node, in the rppgkit folder, start the interface on a free port, for
   example 5000:

    ```bash
    uv run mlflow ui --backend-store-uri sqlite:///<mlflow_root>/mlflow.db --port 5000
    ```

    With `mlflow_root: /data/<your_folder>/mlflow`, the URI is
    `sqlite:////data/<your_folder>/mlflow/mlflow.db`, with four slashes. If the port is
    in use by another user, choose another one.

2. On your laptop, forward the port to the login node:

    ```bash
    ssh -N -L 5000:localhost:5000 <user>@<login_node>
    ```

3. Open `http://localhost:5000` in a browser on your laptop.

The interface shows runs while their jobs are still running; reload the page to update
it. To stop the interface, press Ctrl+C in both terminals.

## Find runs

Select the MLflow experiment of the stage, for example `evaluation`. Each run is named
after the settings that distinguish it:

| MLflow experiment | Run name | Example |
|---|---|---|
| `roi_extraction` | ROI config and the first six characters of the cache key | `yolo5face36_tight · 2ec92a` |
| `preprocessing` | preprocessing steps and ROI config | `rgb_mean · yolo5face36_tight` |
| `splits` | split config | `kfold` |
| `evaluation`, `training` | method, preprocessing steps, ROI config, split config and fold | `pos · yolo5face36_tight · kfold fold 2` |

Settings changed on the command line, such as `evaluation.window_sec=10`, do not appear
in the name. They are listed in the tag `overrides` and in the parameters.

To select runs, enter a filter in the search field above the run table:

| To show | Filter |
|---|---|
| Runs on one dataset | `tags.dataset = "ubfc_rppg"` |
| Runs of one method | `params.method._target_ = "rppgkit.methods.pos.POS"` |
| All folds of one experiment | `tags.experiment_key = "<key>"` |
| Runs below an error threshold | `metrics.mae_bpm < 5` |

Combine filters with `and`.

## Inspect a run

Click a run name to open it. A run contains:

| Section | Content |
|---|---|
| Parameters | Every setting of the resolved config as a separate entry, for example `evaluation.window_sec` |
| Metrics | `mae_bpm`, `rmse_bpm`, `mape_percent`, `pearson_r`, `n_videos`, `n_failed_predictions`, `n_failed_ground_truths`; for `train`, also the loss and learning rate of each epoch |
| Datasets | Each cache that the run built or read, with its name (`roi`, `preprocessing.<step>`), its cache key and the label `built` or `read`; only the path is recorded, not the data |
| Tags | `script`, `dataset`, `run_dir`, `overrides`, `user`, `host`, `slurm_job_id`, `fold`, `experiment_key`, the cache keys, `split_sha256`, `source_sha256`, `uv_lock_sha256` and the library versions |
| Artifacts | `config.yaml`, `metrics.json`, `hr_per_video.csv`, the job log, `.hydra/`, the split file and the `config.json` of each cache |

To find the run that built a cache and every run that read it, select all MLflow
experiments and filter by `datasets.digest = "<cache key>"`.

The tag `run_dir` gives the run folder on the cluster. The tag `source_sha256` identifies
the code of rppgkit that produced the run; runs with the same value used the same code.

## Check failed runs

The status column shows `FINISHED` or `FAILED`. A failed run has the tag `error`, which
contains the exception, and the job log among its artifacts.

## Compare runs

Select the runs in the run table and click **Compare**. The comparison lists their
parameters and metrics side by side and highlights the parameters that differ.

To compare the training curves of `train` runs, open the chart view of the run table
and select a metric, for example `val/loss`. The x-axis is the epoch.

## Read the result of a k-fold evaluation

When one `-m` command runs several folds of the same settings, for example
`split=kfold split.fold=0,1,2,3,4`, the folds are recorded as child runs of one parent
run. The parent is named like its folds without the fold number, for example
`pos · yolo5face36_tight · kfold`. Expand it in the run table to see the folds.

When the last fold has ended, the parent receives:

- the mean of each metric over the finished folds, under the same name, for example
  `mae_bpm`; the counts `n_videos`, `n_failed_predictions` and `n_failed_ground_truths`
  are summed instead;
- the parameters that are the same in all folds;
- the tag `experiment_key`, which is the same for all folds;
- the status `FINISHED`, or `FAILED` if any fold failed.

A command that sweeps several settings, for example `roi=yolo5face36_tight,haar72
split.fold=0,1,2,3,4`, gives one parent for each ROI config. If the `-m` command is
cancelled before all folds have been submitted, the parent remains in the status
`RUNNING`.
