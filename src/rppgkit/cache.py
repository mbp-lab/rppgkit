"""Storage of pipeline stage outputs under configuration-derived keys.

Each cache is stored in `<cache_root>/<dataset>/<stage>/<key>/`. The folder contains
`config.json`, which records exactly what the key hashes, and one folder per recording
with `.npy` arrays and `meta.json`.

The ROI key is computed from the dataset and ROI configurations. A preprocessing key is
computed from the ROI key and the configuration of one step, so a new ROI cache always
yields new preprocessing caches. Keys exclude settings that do not affect the output
and include the `version` of every class named by a `_target_`, so that code changes
also invalidate caches. A recording folder appears only once it has been fully
written; an existing folder is therefore complete.

`StageCache.for_roi` and `StageCache.for_preprocessing` are the only places that
compute keys and folders. `1_extract_roi.py` and `2_preprocess.py` write through the
returned `StageCache`; all other scripts only read.
"""

import hashlib
import importlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rppgkit.recording import RecordingKey

# Settings that change where or how fast a cache is built, never its contents.
CACHE_KEY_EXCLUDED = {"root", "device", "batch_size", "num_workers"}

# `_target_` packages whose classes have no `version`: they are versioned by the
# package version pinned in uv.lock (e.g. torch.optim.Adam in a method config).
UNVERSIONED_PACKAGES = ("torch.",)

# The script that builds each stage's cache, named in "cache not found" errors.
STAGE_SCRIPTS = {"roi": "1_extract_roi.py", "preprocessing": "2_preprocess.py"}


def plain_config(config: Any) -> Any:
    """Converts a configuration to plain dicts and lists, resolving interpolations.

    Args:
        config: An OmegaConf node (e.g. `cfg.roi`), a plain value, or dicts and lists
            that contain both. A node resolves `${...}` against its complete
            configuration.

    Returns:
        The same structure as plain dicts, lists and values, serialisable to JSON.
    """
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return {k: plain_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [plain_config(v) for v in config]
    return config


def compute_cache_key(config: Any, excluded: set[str] = CACHE_KEY_EXCLUDED) -> str:
    """Hashes a configuration together with the versions of the classes it names.

    Args:
        config: A configuration as accepted by `plain_config`. It is hashed with its
            values resolved, so a value hashes identically regardless of how it was
            set.
        excluded: Keys removed at any depth before hashing because they do not affect
            the output (paths, device, batch size, workers).

    Returns:
        The first 12 hexadecimal digits of the SHA-256 hash of the configuration
        without the excluded keys, combined with `{_target_: version}` for every
        `_target_` it contains.

    Raises:
        TypeError: If a class named by a `_target_` has no integer `version`.
    """
    config = plain_config(config)
    payload = {
        "config": _without_excluded(config, excluded),
        "versions": _target_class_versions(config),
    }
    serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized_payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class StageCache:
    """The cache of one pipeline stage, `<cache_root>/<dataset>/<stage>/<key>/`.

    Instances are built from a run configuration with `for_roi` or `for_preprocessing`
    (or `preprocessing_caches` for all selected steps), never by hand, so that every
    script computes the same key from the same settings. Direct construction is
    reserved for tests that use a folder of their own.

    Example:
        ```python
        roi = StageCache.for_roi(cfg)
        preprocessing = preprocessing_caches(cfg, roi)
        for cache in [roi, *preprocessing.values()]:
            cache.check_has_recordings(recording_keys)
        arrays, meta = roi.read_recording(recording_keys[0])
        ```

    Attributes:
        stage: `"roi"` or `"preprocessing"`. Determines the build script named in
            error messages.
        key: Hash of `config` and the versions of the classes it names.
        dir: The cache folder.
        config: Exactly what the key hashes, as a plain, resolved configuration.
    """

    stage: str
    key: str
    dir: Path
    config: dict[str, Any]

    @classmethod
    def for_roi(cls, cfg: Any) -> "StageCache":
        """Returns the ROI cache of a run configuration.

        The cache need not have been built yet.

        Args:
            cfg: The run configuration, with `cache_root`, `dataset` (with `name`) and
                `roi`.

        Returns:
            The ROI cache, whose key is computed from the `dataset` and `roi`
            configurations.
        """
        config = plain_config({"dataset": cfg.dataset, "roi": cfg.roi})
        return cls._keyed(cfg, "roi", config)

    @classmethod
    def for_preprocessing(
        cls, cfg: Any, step_name: str, roi: "StageCache"
    ) -> "StageCache":
        """Returns the cache of one preprocessing step.

        The cache need not have been built yet.

        Args:
            cfg: The run configuration, with `cache_root`, `dataset` (with `name`) and
                `preprocessing` (steps by name).
            step_name: The name of the step in `cfg.preprocessing`.
            roi: The ROI cache that the step reads.

        Returns:
            The cache of the step, whose key is computed from the ROI key and the
            configuration of the step.
        """
        config = plain_config(
            {"roi_cache_key": roi.key, "preprocessing": cfg.preprocessing[step_name]}
        )
        return cls._keyed(cfg, "preprocessing", config)

    @classmethod
    def _keyed(cls, cfg: Any, stage: str, config: dict[str, Any]) -> "StageCache":
        """Returns the cache of one stage for the dataset of `cfg`, keyed by `config`.

        Args:
            cfg: The run configuration, with `cache_root` and `dataset` (with `name`).
            stage: `"roi"` or `"preprocessing"`.
            config: The plain, resolved configuration that the key hashes.

        Returns:
            The cache at `<cache_root>/<dataset>/<stage>/<key>/`.
        """
        key = compute_cache_key(config)
        return cls(
            stage, key, Path(cfg.cache_root) / cfg.dataset.name / stage / key, config
        )

    def check_has_recordings(self, recording_keys: list[RecordingKey]) -> None:
        """Checks that all given recordings are in the cache before any is read.

        Scripts call this method before their work begins, so that a missing
        recording stops them immediately rather than after hours of computation.

        Args:
            recording_keys: The recordings that the script will read.

        Raises:
            FileNotFoundError: If the cache folder or any of the recordings is
                missing. The message names the build script.
        """
        build_hint = (
            f"run scripts/{STAGE_SCRIPTS[self.stage]} with the same dataset, roi (and "
            f"preprocessing) settings"
        )
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"{self.stage} cache not found: {self.dir}; {build_hint}"
            )
        if missing := self.recordings_not_in_cache(recording_keys):
            raise FileNotFoundError(
                f"{self.stage} cache {self.dir} is missing {len(missing)}/"
                f"{len(recording_keys)} recordings (e.g. {missing[0]}); {build_hint}"
            )

    def recordings_not_in_cache(
        self, recording_keys: list[RecordingKey]
    ) -> list[RecordingKey]:
        """Returns the given recordings that have no complete folder in the cache.

        Args:
            recording_keys: The recordings to look for.

        Returns:
            The missing recordings in the order of `recording_keys`, or all of them if
            the cache folder does not exist yet.
        """
        return [key for key in recording_keys if not self.has_recording(key)]

    def has_recording(self, recording: RecordingKey) -> bool:
        """Returns whether the recording has a complete folder in the cache."""
        return self._recording_dir(recording).exists()

    def read_recording(
        self, recording: RecordingKey
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reads one recording from the cache.

        Args:
            recording: The recording to read.

        Returns:
            A tuple `(arrays, meta)` of all `.npy` arrays by name, memory-mapped
            read-only, and the parsed `meta.json`.

        Raises:
            FileNotFoundError: If the recording is not in the cache.
        """
        path = self._recording_dir(recording)
        if not path.exists():
            raise FileNotFoundError(
                f"{recording} missing in {self.dir}; build that cache with the same "
                f"config"
            )
        arrays = {
            npy_path.stem: np.load(npy_path, mmap_mode="r")
            for npy_path in sorted(path.glob("*.npy"))
        }
        meta = json.loads((path / "meta.json").read_text())
        return arrays, meta

    def write_recording(
        self,
        recording: RecordingKey,
        arrays: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> None:
        """Writes the arrays and metadata of one recording atomically.

        One `.npy` file per array and `meta.json` are written to a temporary folder,
        which is then renamed. An existing folder is replaced only after the new one
        has been fully written. A crash in between leaves the recording missing, to
        be rebuilt on the next run, but never partially written.

        Args:
            recording: The recording to write.
            arrays: Arrays by name, each saved as `<name>.npy`.
            meta: JSON-serialisable metadata, saved as `meta.json`.
        """
        path = self._recording_dir(recording)
        temp_dir = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        for name, array in arrays.items():
            np.save(temp_dir / f"{name}.npy", array)
        (temp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if path.exists():
            shutil.rmtree(path)
        temp_dir.rename(path)

    def write_config(self) -> None:
        """Writes `config.json` atomically and creates the cache folder if necessary.

        `config.json` contains the stage, the key, `config` and the version of every
        class that `config` names, which suffices to recompute the key and to
        identify what built the cache. Build scripts call this method before writing
        any recording.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": self.stage,
            "key": self.key,
            "config": self.config,
            "versions": _target_class_versions(self.config),
        }
        temp_path = self.dir / f"config.json.tmp-{os.getpid()}"
        temp_path.write_text(json.dumps(record, indent=2))
        temp_path.replace(self.dir / "config.json")

    def _recording_dir(self, recording: RecordingKey) -> Path:
        """Returns the folder of a recording, `<dir>/<subject>/<recording>`."""
        return self.dir / recording.subject / recording.recording


def preprocessing_caches(cfg: Any, roi: StageCache) -> dict[str, StageCache]:
    """Returns the cache of every selected preprocessing step.

    The caches need not have been built yet.

    Args:
        cfg: The run configuration, with `cache_root`, `dataset` (with `name`) and
            `preprocessing` (steps by name, empty for none).
        roi: The ROI cache that the steps read.

    Returns:
        One cache per step name, as returned by `StageCache.for_preprocessing`.
    """
    return {
        name: StageCache.for_preprocessing(cfg, name, roi) for name in cfg.preprocessing
    }


def _target_class_versions(config: Any) -> dict[str, int]:
    """Collects the `version` of every class named by a `_target_` in a configuration.

    Classes from `UNVERSIONED_PACKAGES` (e.g. `torch.optim.Adam`) are skipped.

    Args:
        config: A plain configuration, at any depth.

    Returns:
        `{_target_: version}` for every versioned `_target_` in `config`.

    Raises:
        TypeError: If a named class has no integer `version`.
    """
    if isinstance(config, list):
        return {
            k: v for item in config for k, v in _target_class_versions(item).items()
        }
    if not isinstance(config, dict):
        return {}
    versions = {
        k: v
        for item in config.values()
        for k, v in _target_class_versions(item).items()
    }
    target_name = config.get("_target_")
    if target_name and not target_name.startswith(UNVERSIONED_PACKAGES):
        module, _, name = target_name.rpartition(".")
        target = getattr(importlib.import_module(module), name)
        if not isinstance(getattr(target, "version", None), int):
            raise TypeError(
                f"{config['_target_']} has no integer `version`; add `version = 1` "
                f"to the class (it is part of the cache key)"
            )
        versions[target_name] = target.version
    return versions


def _without_excluded(value: Any, excluded: set[str]) -> Any:
    """Returns `value` with the excluded keys removed from every nested dict.

    Args:
        value: A plain configuration, at any depth.
        excluded: Keys to remove.

    Returns:
        A copy of `value` without the excluded keys.
    """
    if isinstance(value, dict):
        return {
            k: _without_excluded(v, excluded)
            for k, v in value.items()
            if k not in excluded
        }
    if isinstance(value, list):
        return [_without_excluded(v, excluded) for v in value]
    return value
