"""Evaluate a method on the test subjects of one split fold (no training).

Usage: uv run python scripts/evaluate.py [method=pos] [split=test_only|kfold] [split.fold=0]

For methods without training (POS, ...). Trainable methods use train.py: fit on the
fold's train/val subjects, then the same evaluation (rppgkit.experiment).

Reads the ROI cache (and the preprocessing caches if `preprocessing` is configured),
predicts every test video, and evaluates against the raw PPG from the ROI cache.
Writes hr_per_video.csv and metrics.json into the Hydra run folder.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from rppgkit.cache import StageCache, preprocessing_caches
from rppgkit.cached_videos import select_recordings
from rppgkit.experiment import (
    compute_experiment_key,
    load_fold,
    predict_and_evaluate,
    write_results,
)
from rppgkit.methods.trainable import TrainableMethod

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Predicts and evaluates the test recordings of one split fold.

    Args:
        cfg: The composed Hydra config (configs/evaluate.yaml), with command-line overrides.
    """
    # --- Dataset: the reader and every recording it finds on disk.
    dataset_reader = instantiate(cfg.dataset)
    recording_keys = dataset_reader.list_recording_keys()
    log.info("dataset %s: %d recordings", dataset_reader.name, len(recording_keys))

    # --- Split: the fold's test subjects and their recordings.
    split, split_sha256 = load_fold(
        REPO_ROOT / cfg.split.path,
        cfg.split.fold,
        dataset_reader.name,
        HydraConfig.get().runtime.choices["split"],
    )
    test_recording_keys = select_recordings(split.test, recording_keys)
    log.info(
        "fold %d: %d test subjects, %d recordings",
        cfg.split.fold,
        len(split.test),
        len(test_recording_keys),
    )

    # --- Caches: the ROI cache and every preprocessing cache, checked to hold every
    # test recording before anything is predicted.
    roi = StageCache.for_roi(cfg)
    preprocessing = preprocessing_caches(cfg, roi)
    for cache in [roi, *preprocessing.values()]:
        cache.check_has_recordings(test_recording_keys)
        log.info("%s cache: %s", cache.stage, cache.dir)

    # --- Method and evaluation settings.
    method = instantiate(cfg.method)
    if isinstance(method, TrainableMethod):
        # Would test random weights and still write plausible-looking metrics.
        raise TypeError(f"{cfg.method._target_} must be trained; use scripts/train.py")
    evaluation_settings = instantiate(cfg.evaluation)

    # --- Predict and evaluate: each test recording without labels, against the raw
    # PPG of the ROI cache.
    rows, metrics = predict_and_evaluate(
        method, test_recording_keys, roi, preprocessing, evaluation_settings
    )

    # --- Results: keys and configs that identify the run, written next to the metrics.
    # Same experiment_key for every fold of one experiment (everything but the fold),
    # so folds can be grouped and averaged; any other change gives a new key.
    experiment_config = {
        "roi_cache_key": roi.key,
        "preprocessing_cache_keys": {
            name: cache.key for name, cache in preprocessing.items()
        },
        "split_sha256": split_sha256,
        "method": cfg.method,
        "evaluation": cfg.evaluation,
    }
    provenance = {
        "experiment_key": compute_experiment_key(experiment_config),
        **experiment_config,
        "roi_cache": str(roi.dir),
        "preprocessing_caches": {
            name: str(cache.dir) for name, cache in preprocessing.items()
        },
        "split": {"path": cfg.split.path, "fold": cfg.split.fold},
    }
    log.info("experiment %s, fold %d", provenance["experiment_key"], cfg.split.fold)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    write_results(run_dir, rows, metrics, provenance)


if __name__ == "__main__":
    main()
