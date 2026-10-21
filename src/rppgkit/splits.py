"""Subject-level splits: which subjects are train, val and test, and the leakage check.

A split file lists every fold explicitly plus how it was made. Subject IDs carry their
dataset ("ubfc_rppg/subject1"), so one split can span datasets. Splits say nothing about
preprocessing: the same file works with any ROI or preprocessing config, so ablations
compare the same subjects. Subjects can be restricted first (select_subjects), then a
Splitter turns them into folds.
"""

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rppgkit.cache import plain_config
from rppgkit.recording import RecordingKey


@dataclass(frozen=True)
class Split:
    """Subject IDs of one fold; raises on creation if any subject is in two parts.

    Attributes:
        train: Subject IDs "<dataset>/<subject>" to train on.
        val: Subject IDs for validation; drawn from outside train and test.
        test: Subject IDs to test on.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        """Checks for leakage: no duplicates, no subject in two of train, val and test.

        Raises:
            ValueError: If a subject appears twice in one part or in two parts.
        """
        parts = {"train": self.train, "val": self.val, "test": self.test}
        for name, ids in parts.items():
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate subject in {name}")
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(parts[first]) & set(parts[second])
            if shared:
                raise ValueError(
                    f"subjects in both {first} and {second}: {sorted(shared)}"
                )

    @property
    def dataset_names(self) -> set[str]:
        """Dataset names of all subjects: the part of each ID before "/"."""
        return {subject.split("/")[0] for subject in self.train + self.val + self.test}


def unique_subject_ids(keys: list[RecordingKey]) -> list[str]:
    """Returns the sorted unique subject IDs "<dataset>/<subject>" of some recordings.

    Args:
        keys: Recording keys; a subject with several recordings is listed once.
    """
    return sorted({key.subject_id for key in keys})


def select_subjects(
    subject_ids: list[str], include: Sequence[str] | None, exclude: Sequence[str]
) -> list[str]:
    """Restricts subjects to `include`, minus `exclude`.

    Args:
        subject_ids: All subject IDs available.
        include: Subject IDs to keep, or None to keep all.
        exclude: Subject IDs to drop after `include` is applied.

    Returns:
        The kept subject IDs, in the order of `subject_ids`.

    Raises:
        ValueError: If `include` or `exclude` names an ID not in `subject_ids`
            (a guard against typos).
    """
    if unknown := (set(include or []) | set(exclude)) - set(subject_ids):
        raise ValueError(f"unknown subject IDs: {sorted(unknown)}")
    included = (
        subject_ids
        if include is None
        else [subject for subject in subject_ids if subject in include]
    )
    return [subject for subject in included if subject not in exclude]


class Splitter(ABC):
    """Turns subject IDs into folds; every fold's `Split` runs the leakage check."""

    @abstractmethod
    def __call__(self, subject_ids: list[str]) -> list[Split]:
        """Makes the folds.

        Args:
            subject_ids: The subjects to split, already restricted by
                `select_subjects`.

        Returns:
            One `Split` per fold.
        """


class KFold(Splitter):
    """K-fold over subjects: each subject is in the test part of exactly one fold.

    Subjects are sorted, shuffled with `seed`, and cut into `n_folds` consecutive
    blocks whose sizes differ by at most one (as sklearn's `KFold`). Each fold draws
    its own validation subjects from its non-test subjects with `seed + fold`.

    Args:
        n_folds: Number of folds, between 2 and the number of subjects.
        val_fraction: Fraction of each fold's non-test subjects used for validation.
        seed: Seed for the shuffle and the validation draws.
    """

    def __init__(self, n_folds: int, val_fraction: float, seed: int) -> None:
        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.seed = seed

    def __call__(self, subject_ids: list[str]) -> list[Split]:
        """Makes `n_folds` folds; the result does not depend on the input order.

        Raises:
            ValueError: If `subject_ids` has duplicates or `n_folds` is outside
                [2, number of subjects].
        """
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("duplicate subject IDs")
        if not 2 <= self.n_folds <= len(subject_ids):
            raise ValueError(
                f"n_folds must be in [2, {len(subject_ids)}], got {self.n_folds}"
            )
        subjects = sorted(subject_ids)  # same result whatever the input order
        random.Random(self.seed).shuffle(subjects)
        splits = []
        # Consecutive blocks of the shuffled list; sizes differ by at most one (as sklearn).
        for fold, test_block in enumerate(np.array_split(subjects, self.n_folds)):
            test = test_block.tolist()  # back to plain Python strings
            non_test_subjects = [subject for subject in subjects if subject not in test]
            # Own random val subjects per fold, reproducible from the seed.
            n_val = round(self.val_fraction * len(non_test_subjects))
            val = random.Random(self.seed + fold).sample(non_test_subjects, n_val)
            splits.append(
                Split(
                    train=tuple(
                        sorted(
                            subject
                            for subject in non_test_subjects
                            if subject not in val
                        )
                    ),
                    val=tuple(sorted(val)),
                    test=tuple(sorted(test)),
                )
            )
        return splits


class TestOnly(Splitter):
    """One fold with every subject in test, e.g. for methods without training."""

    def __call__(self, subject_ids: list[str]) -> list[Split]:
        """Makes one fold with all subjects, sorted, in test."""
        return [Split(train=(), val=(), test=tuple(sorted(subject_ids)))]


class CrossDataset(Splitter):
    """One fold that trains on some datasets and tests on others.

    Args:
        train_datasets: Datasets whose subjects make up train and val.
        test_datasets: Datasets whose subjects make up test.
        val_fraction: Fraction of the train datasets' subjects used for validation.
        seed: Seed for the validation draw.

    Raises:
        ValueError: If a dataset is on both sides.
    """

    def __init__(
        self,
        train_datasets: list[str],
        test_datasets: list[str],
        val_fraction: float,
        seed: int,
    ) -> None:
        if both := set(train_datasets) & set(test_datasets):
            raise ValueError(f"datasets on both sides: {sorted(both)}")
        self.train_datasets = list(train_datasets)
        self.test_datasets = list(test_datasets)
        self.val_fraction = val_fraction
        self.seed = seed

    def __call__(self, subject_ids: list[str]) -> list[Split]:
        """Makes the single cross-dataset fold.

        Raises:
            ValueError: If a named dataset has no subjects, or a subject comes from a
                dataset named on neither side.
        """
        subjects_by_dataset: dict[str, list[str]] = {}
        for subject in sorted(subject_ids):
            subjects_by_dataset.setdefault(subject.split("/")[0], []).append(subject)
        named_datasets = self.train_datasets + self.test_datasets
        if empty := [
            name for name in named_datasets if name not in subjects_by_dataset
        ]:
            raise ValueError(f"no subjects from datasets {empty}")
        if other := sorted(set(subjects_by_dataset) - set(named_datasets)):
            raise ValueError(f"subjects from datasets on neither side: {other}")
        train_val_subjects = [
            subject
            for dataset in self.train_datasets
            for subject in subjects_by_dataset[dataset]
        ]
        test = [
            subject
            for dataset in self.test_datasets
            for subject in subjects_by_dataset[dataset]
        ]
        n_val = round(self.val_fraction * len(train_val_subjects))
        val = random.Random(self.seed).sample(train_val_subjects, n_val)
        train = [subject for subject in train_val_subjects if subject not in val]
        return [Split(train=tuple(train), val=tuple(sorted(val)), test=tuple(test))]


def write_split_file(path: Path, splits: list[Split], split_config: Any) -> None:
    """Writes a split file: every fold's subjects and how the folds were made.

    The file holds `datasets` and `n_subjects` (from the folds), `include`, `exclude`
    and `maker` (from `split_config`), and `folds`. The maker's `_target_` names the
    Splitter class, its other keys are the Splitter's arguments.

    Args:
        path: The split file, e.g. splits/<dataset>_kfold.json.
        splits: All folds.
        split_config: The split config (e.g. `cfg.split`), with `include`, `exclude`
            and `maker`; its other keys (`path`, `fold`) are not written.

    Raises:
        FileExistsError: If `path` exists; split files are never overwritten.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; split files are committed, not replaced")
    subjects = {s for split in splits for s in split.train + split.val + split.test}
    record = {
        "datasets": sorted({subject.split("/")[0] for subject in subjects}),
        "n_subjects": len(subjects),
        "include": split_config["include"],
        "exclude": split_config["exclude"],
        "maker": split_config["maker"],
        "folds": [
            {"train": split.train, "val": split.val, "test": split.test}
            for split in splits
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain_config(record), indent=2) + "\n")


def read_split_fold(path: Path, fold: int) -> Split:
    """Reads one fold of a split file; the leakage check runs on creation.

    Args:
        path: A split file written by `write_split_file`.
        fold: Index of the fold, from 0.

    Returns:
        The fold's `Split`.

    Raises:
        IndexError: If the file has no fold `fold`.
        ValueError: If the fold fails the leakage check.
    """
    folds = json.loads(Path(path).read_text())["folds"]
    if not 0 <= fold < len(folds):
        raise IndexError(f"{path} has folds 0..{len(folds) - 1}, got {fold}")
    return Split(**{part: tuple(ids) for part, ids in folds[fold].items()})
