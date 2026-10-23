"""Logs every script run to MLflow, as a Hydra callback.

Selected in each top-level config (configs/hydra/callbacks/mlflow.yaml);
`hydra/callbacks=none` runs without it. One MLflow run per Hydra job, in one MLflow
experiment per stage (STAGE_EXPERIMENTS). The folds of one experiment in a sweep are
child runs of a parent run that holds their mean metrics. Each cache a run built or
read is logged as an MLflow dataset input (its path and key, not its data).

MLflow indexes the run folders: metrics.json, the caches' config.json and the
checkpoints stay the source of truth. The only module of rppgkit that knows Hydra.
"""

import getpass
import hashlib
import importlib.metadata
import json
import os
import socket
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from mlflow import MlflowClient
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.entities import Param
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from rppgkit.cache import StageCache, plain_config, preprocessing_caches

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

# Logged as version.<name> tags: what can change a result besides rppgkit's own code.
LIBRARIES = (
    "torch",
    "numpy",
    "scipy",
    "opencv-contrib-python-headless",
    "hydra-core",
    "mlflow",
)

# One MLflow experiment per stage, keyed by the script's config name.
STAGE_EXPERIMENTS = {
    "extract_roi": "roi_extraction",
    "preprocess": "preprocessing",
    "make_splits": "splits",
    "evaluate": "evaluation",
    "train": "training",
}

# The cache each building script writes; every other cache a run lists is read.
BUILDS = {"extract_roi": "roi", "preprocess": "preprocessing"}

# Results in the run folder; checkpoints stay on disk only (their path is in metrics.json).
RESULT_FILES = ("metrics.json", "hr_per_video.csv")


class MLflowTracking(Callback):
    """Logs each Hydra job of a script as one MLflow run.

    Args:
        tracking_uri: The MLflow store, e.g. sqlite:////<mlflow_root>/mlflow.db.
        artifact_root: Folder for the runs' files, one subfolder per experiment;
            created with its parents if missing.
    """

    def __init__(self, tracking_uri: str, artifact_root: str) -> None:
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root
        # Parent run id of each k-fold experiment of a sweep, by its overrides
        # without split.fold; set on the submitting node, pickled into each job.
        self.parent_runs: dict[frozenset[str], str] = {}

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Creates the store, the experiment and the parent runs, before a sweep's
        jobs start.

        Parallel jobs creating a new store or experiment at the same moment would
        race; this way they only find them. Jobs whose overrides differ only in
        split.fold share a parent run; the first of them names it, the last closes it.
        """
        experiment_id = self._experiment_id(
            STAGE_EXPERIMENTS[config.hydra.job.config_name]
        )
        # The sweep's jobs, expanded as Hydra's basic sweeper does (private Hydra API).
        jobs = BasicSweeper.split_arguments(
            OverridesParser.create().parse_overrides(list(config.hydra.overrides.task)),
            None,
        )[0]
        for overrides, n_folds in Counter(_without_fold(job) for job in jobs).items():
            if n_folds > 1:
                self.parent_runs[overrides] = (
                    MlflowClient()
                    .create_run(
                        experiment_id,
                        tags={
                            "n_folds": str(n_folds),
                            "overrides": " ".join(sorted(overrides)),
                        },
                    )
                    .info.run_id
                )

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Starts the job's run: the resolved config as params and config.yaml, and
        tags for the script, run folder, overrides, user, host, SLURM job, code and
        library versions.

        Raises:
            MlflowException: If the store can't be opened or written; the job then
                fails before doing any work.
        """
        experiment_id = self._experiment_id(
            STAGE_EXPERIMENTS[config.hydra.job.config_name]
        )
        job_config = plain_config(
            OmegaConf.masked_copy(config, [k for k in config if k != "hydra"])
        )
        tags = {
            "script": config.hydra.job.name,
            "dataset": config.dataset.name,
            "run_dir": config.hydra.runtime.output_dir,
            "overrides": " ".join(config.hydra.overrides.task),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "source_sha256": _source_sha256(),
            **{f"version.{name}": version for name, version in _versions().items()},
        }
        if "SLURM_JOB_ID" in os.environ:
            tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        if (REPO_ROOT / "uv.lock").is_file():
            tags["uv_lock_sha256"] = _sha256(REPO_ROOT / "uv.lock")
        if "split" in config and "fold" in config.split:
            tags["fold"] = str(config.split.fold)

        parent_run_id = self.parent_runs.get(_without_fold(config.hydra.overrides.task))
        if parent_run_id:
            MlflowClient().set_tag(
                parent_run_id, "mlflow.runName", _run_name(config, with_fold=False)
            )
        mlflow.start_run(
            experiment_id=experiment_id,
            run_name=_run_name(config),
            tags=tags,
            parent_run_id=parent_run_id,
        )
        try:
            mlflow.log_params(_flatten(job_config))
            mlflow.log_dict(job_config, "config.yaml")
        except Exception:
            mlflow.end_run("FAILED")
            raise

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Logs what the job built, read and wrote, then ends the run.

        Also runs when the job raised: whatever it wrote is still logged, and the run
        is marked FAILED with the error. If logging itself fails, the run is marked
        FAILED too, never left RUNNING. Then closes the parent run if this was its
        last fold.
        """
        try:
            self._log_outputs(config)
        except Exception:
            mlflow.end_run("FAILED")
            self._close_parent(config)
            raise

        # Status: FINISHED, or FAILED with the job's error.
        if job_return.status == JobStatus.COMPLETED:
            mlflow.end_run("FINISHED")
        else:
            # The exception itself; the public `return_value` would re-raise it.
            error = job_return._return_value
            mlflow.set_tag("error", f"{type(error).__name__}: {error}"[:5000])
            mlflow.end_run("FAILED")
        self._close_parent(config)

    def _close_parent(self, config: DictConfig) -> None:
        """Closes the job's parent run once all its folds have ended.

        Logs the mean of each metric over the finished folds (counts n_* summed), the
        params all folds share, and the status: FAILED if any fold failed. Two folds
        ending together may both close it, with the same values.
        """
        parent_run_id = self.parent_runs.get(_without_fold(config.hydra.overrides.task))
        if parent_run_id is None:
            return
        client = MlflowClient()
        parent = client.get_run(parent_run_id)
        folds = client.search_runs(
            [parent.info.experiment_id],
            f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
        )
        if len(folds) < int(parent.data.tags["n_folds"]) or any(
            fold.info.status == "RUNNING" for fold in folds
        ):
            return
        finished = [fold for fold in folds if fold.info.status == "FINISHED"]

        # Metrics: mean over the finished folds; counts summed.
        for name in set().union(*(fold.data.metrics for fold in finished)):
            values = [
                fold.data.metrics[name]
                for fold in finished
                if name in fold.data.metrics
            ]
            client.log_metric(
                parent_run_id,
                name,
                sum(values) if name.startswith("n_") else sum(values) / len(values),
            )

        # Params and experiment_key that all folds share.
        shared = set.intersection(*(set(fold.data.params.items()) for fold in folds))
        params = [Param(key, value) for key, value in shared]
        for start in range(0, len(params), 100):  # at most 100 params per batch
            client.log_batch(parent_run_id, params=params[start : start + 100])
        keys = {fold.data.tags.get("experiment_key") for fold in finished}
        if len(keys) == 1 and None not in keys:
            client.set_tag(parent_run_id, "experiment_key", keys.pop())

        client.set_terminated(
            parent_run_id, "FINISHED" if len(finished) == len(folds) else "FAILED"
        )

    def _log_outputs(self, config: DictConfig) -> None:
        """Logs the caches, split file, results and logs of the job's run folder."""
        run_dir = Path(config.hydra.runtime.output_dir)

        # Caches: the ones built (1_extract_roi, 2_preprocess) or read (evaluate, train).
        if "roi" in config:
            roi = StageCache.for_roi(config)
            caches = {"roi": roi}
            if "preprocessing" in config:
                caches |= {
                    f"preprocessing.{name}": cache
                    for name, cache in preprocessing_caches(config, roi).items()
                }
            for name, cache in caches.items():
                mlflow.set_tag(f"{name}_cache_key", cache.key)
                mlflow.log_input(
                    MetaDataset(
                        LocalArtifactDatasetSource(str(cache.dir)),
                        name=name,
                        digest=cache.key,
                    ),
                    context="built"
                    if name.split(".")[0] == BUILDS.get(config.hydra.job.config_name)
                    else "read",
                )
                if (cache.dir / "config.json").is_file():
                    mlflow.log_artifact(
                        str(cache.dir / "config.json"), f"caches/{name}"
                    )

        # Split: the file made (3_make_splits) or read (evaluate, train).
        if "split" in config:
            split_path = REPO_ROOT / config.split.path
            if split_path.is_file():
                mlflow.set_tag("split_sha256", _sha256(split_path))
                mlflow.log_artifact(str(split_path), "split")

        # Results: metrics and experiment_key (evaluate, train), files and logs.
        if (run_dir / "metrics.json").is_file():
            results = json.loads((run_dir / "metrics.json").read_text())
            mlflow.set_tag("experiment_key", results["experiment_key"])
            mlflow.log_metrics(
                {
                    name: float(value)
                    for name, value in results["metrics"].items()
                    if _is_number(value)
                }
            )
        for name in RESULT_FILES:
            if (run_dir / name).is_file():
                mlflow.log_artifact(str(run_dir / name))
        for log_path in run_dir.glob("*.log"):
            mlflow.log_artifact(str(log_path))
        if (run_dir / ".hydra").is_dir():
            mlflow.log_artifacts(str(run_dir / ".hydra"), ".hydra")

    def _experiment_id(self, experiment_name: str) -> str:
        """Returns the id of the stage's experiment; creates store and experiment."""
        artifact_dir = Path(self.artifact_root) / experiment_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        try:
            return mlflow.create_experiment(
                experiment_name, artifact_location=artifact_dir.resolve().as_uri()
            )
        except MlflowException:  # created by a parallel job in between
            return mlflow.get_experiment_by_name(experiment_name).experiment_id


def log_epoch_metrics(epoch_log: dict[str, Any]) -> None:
    """Logs one epoch of `Trainer.fit` to the active MLflow run, as it happens.

    Passed to `Trainer.fit` as `on_epoch_end`. Logs every number (train/*, val/*,
    learning_rate) at step `epoch`. Does nothing without an active run (tracking off);
    MLflow would otherwise start a run in a new store in the working directory.

    Args:
        epoch_log: One row of history.csv.
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_metrics(
        {
            name: float(value)
            for name, value in epoch_log.items()
            if name != "epoch" and _is_number(value)
        },
        step=epoch_log["epoch"],
    )


def _run_name(config: DictConfig, with_fold: bool = True) -> str:
    """Returns a run's name from its config choices, e.g. "pos · yolo5face36_tight · kfold fold 2".

    Args:
        config: The job's config, with its hydra node.
        with_fold: Whether to end the name with the fold; False for a parent run.
    """
    choices = config.hydra.runtime.choices
    steps = list(config.get("preprocessing", {}))
    match config.hydra.job.config_name:
        case "extract_roi":
            parts = [choices.roi, StageCache.for_roi(config).key[:6]]
        case "preprocess":
            parts = [*steps, choices.roi]
        case "make_splits":
            parts = [choices.split]
        case _:  # evaluate, train
            parts = [choices.method, *steps, choices.roi, choices.split]
            if with_fold:
                parts[-1] += f" fold {config.split.fold}"
    return " · ".join(parts)


def _without_fold(overrides: list[str]) -> frozenset[str]:
    """Returns a job's overrides without split.fold: the same for all folds of one
    experiment."""
    return frozenset(o for o in overrides if not o.startswith("split.fold="))


def _is_number(value: Any) -> bool:
    """Returns whether a value can be an MLflow metric (bools can't)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Returns nested dicts as {"a.b.c": value}, so each setting is one MLflow param.

    Non-strings (numbers, lists, None) are written as JSON.
    """
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = (
                value if isinstance(value, str) else json.dumps(value)
            )
    return flat


def _source_sha256() -> str:
    """Returns one sha256 over every .py file of the package (paths and contents).

    Identifies a run's code where there is no git, e.g. the cluster copy.
    """
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        digest.update(str(path.relative_to(PACKAGE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """Returns the sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str]:
    """Returns the installed version of each library in `LIBRARIES`."""
    return {name: importlib.metadata.version(name) for name in LIBRARIES}
