"""Make a subject-level split file for one dataset; run once, then commit the file.

Usage: uv run python scripts/3_make_splits.py [dataset=ubfc_rppg] [split=kfold|test_only]

Only lists the dataset's folders (no video is decoded). Writes split.path; an existing
file is never overwritten.
"""

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rppgkit.splits import select_subjects, unique_subject_ids, write_split_file

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="make_splits", version_base=None)
def main(cfg: DictConfig) -> None:
    """Makes the split file of one dataset; never overwrites an existing one.

    Args:
        cfg: The composed Hydra config (configs/make_splits.yaml), with command-line overrides.
    """
    # --- Dataset: the subjects of every recording on disk, restricted by the split
    # config's include/exclude.
    dataset_reader = instantiate(cfg.dataset)
    subjects = select_subjects(
        unique_subject_ids(dataset_reader.list_recording_keys()),
        cfg.split.include,
        cfg.split.exclude,
    )
    if not subjects:
        raise ValueError("no subjects left after include/exclude")
    log.info("dataset %s: %d subjects", dataset_reader.name, len(subjects))

    # --- Split: the folds the configured maker makes from those subjects.
    splits = instantiate(cfg.split.maker)(subjects)
    for fold, split in enumerate(splits):
        log.info(
            "fold %d: %d train, %d val, %d test",
            fold,
            len(split.train),
            len(split.val),
            len(split.test),
        )

    # --- Write: the folds and how they were made; never overwrites an existing file.
    split_path = REPO_ROOT / cfg.split.path
    write_split_file(split_path, splits, cfg.split)
    log.info("wrote %s", split_path)


if __name__ == "__main__":
    main()
