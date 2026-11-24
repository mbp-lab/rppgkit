"""Steps shared by evaluate.py and train.py.

Load a split fold, test a method on the fold's test recordings, and write the
results. The caches come from `rppgkit.cache`.

Takes config nodes or plain dicts, never Hydra's runtime state, so both scripts do
exactly the same thing and the test half of train.py is the same code as
scripts/evaluate.py.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path

from rppgkit.cache import StageCache, compute_cache_key, plain_config
from rppgkit.cached_videos import load_cached_videos
from rppgkit.evaluation import EvaluationSettings, evaluate
from rppgkit.methods.method import Method
from rppgkit.recording import RecordingKey
from rppgkit.splits import Split, read_split_fold

log = logging.getLogger(__name__)

# Settings that don't change a result. Unlike cache keys, batch_size does: it changes
# training, so it is part of experiment_key. predict_batch_size only changes memory.
EXPERIMENT_KEY_EXCLUDED = {"root", "device", "num_workers", "predict_batch_size"}


def load_fold(
    path: Path, fold: int, dataset_name: str, split_kind: str
) -> tuple[Split, str]:
    """Reads one fold of a split file and checks it can be used for this dataset.

    Args:
        path: The split file.
        fold: Index of the fold, from 0.
        dataset_name: The dataset of the experiment.
        split_kind: The split config name; named in the error if the file is missing.

    Returns:
        A tuple `(split, sha256)`: the fold, and the sha256 of the file's bytes.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If the fold has subjects from another dataset or no test subjects.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}; run scripts/3_make_splits.py split={split_kind}"
        )
    split = read_split_fold(path, fold)
    if other := split.dataset_names - {dataset_name}:
        raise ValueError(
            f"{path} has subjects from {sorted(other)}, but the dataset config is "
            f"{dataset_name}; use a split made for this dataset"
        )
    if not split.test:
        raise ValueError(f"{path} fold {fold} has no test subjects")
    return split, hashlib.sha256(path.read_bytes()).hexdigest()


def predict_and_evaluate(
    method: Method,
    recording_keys: list[RecordingKey],
    roi: StageCache,
    preprocessing: dict[str, StageCache],
    evaluation_settings: EvaluationSettings,
) -> tuple[list[dict], dict]:
    """Predicts every test recording without labels and evaluates the predictions.

    Ground truth is the raw PPG read straight from the ROI cache, never through the
    method. Logs a warning if any heart rate could not be estimated.

    Args:
        method: A method ready to predict (trained, if trainable).
        recording_keys: The test recordings.
        roi: The ROI cache; also provides the ground-truth PPG.
        preprocessing: The preprocessing caches by step name; may be empty.
        evaluation_settings: Post-processing, estimator and windowing.

    Returns:
        The `(rows, metrics)` of `evaluate`.
    """
    videos = load_cached_videos(
        recording_keys, roi, preprocessing, method.needs, with_labels=False
    )
    predictions = {}
    for i, video in enumerate(videos, 1):
        predictions[video.key] = method.predict(video)["rppg"]
        log.info("[%d/%d] %s: predicted", i, len(videos), video.key)

    # Ground truth: raw PPG straight from the ROI cache, never through the method.
    ground_truth = {key: roi.read_recording(key)[0]["ppg"] for key in recording_keys}
    fps = {video.key: video.fps for video in videos}
    rows, metrics = evaluate(predictions, ground_truth, fps, evaluation_settings)
    if metrics["n_failed_predictions"] or metrics["n_failed_ground_truths"]:
        log.warning(
            "no heart rate (flat or non-finite signal) for %d predictions and %d "
            "ground truths; metrics exclude them, see hr_per_video.csv",
            metrics["n_failed_predictions"],
            metrics["n_failed_ground_truths"],
        )
    return rows, metrics


def compute_experiment_key(experiment: dict) -> str:
    """Computes the key shared by all folds of one experiment.

    Args:
        experiment: Everything that defines the experiment except the fold (cache
            keys, split hash, method, trainer and evaluation configs, e.g.
            `cfg.method`).

    Returns:
        The cache key of `experiment`, ignoring `EXPERIMENT_KEY_EXCLUDED`.
    """
    return compute_cache_key(experiment, excluded=EXPERIMENT_KEY_EXCLUDED)


def write_results(out: Path, rows: list[dict], metrics: dict, provenance: dict) -> None:
    """Writes hr_per_video.csv and metrics.json to a run folder.

    Args:
        out: The run folder.
        rows: One row per test recording, or per window, from `evaluate`.
        metrics: The metrics from `evaluate`.
        provenance: Keys, hashes and configs (plain or OmegaConf) written next to
            the metrics.
    """
    with open(out / "hr_per_video.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "recording",
                "window_start_sec",
                "predicted_hr_bpm",
                "true_hr_bpm",
                "abs_error_bpm",
            ]
        )
        for r in rows:
            error = abs(r["predicted_hr_bpm"] - r["true_hr_bpm"])
            writer.writerow(
                [
                    r["recording"],
                    r["window_start_sec"],
                    r["predicted_hr_bpm"],
                    r["true_hr_bpm"],
                    error,
                ]
            )
    (out / "metrics.json").write_text(
        json.dumps(plain_config({"metrics": metrics, **provenance}), indent=2)
    )
    log.info("metrics: %s", metrics)
    log.info("results in %s", out)
