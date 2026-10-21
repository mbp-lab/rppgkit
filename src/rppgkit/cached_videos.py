"""Loading of the arrays that a method requires from the ROI and preprocessing caches.

A method receives only the arrays that it declares in `needs`. For test recordings,
every label array (whose name begins with `"ppg"`) is removed, so that methods never
receive test labels.
"""

from dataclasses import dataclass

import numpy as np

from rppgkit.cache import StageCache
from rppgkit.recording import RecordingKey

LABEL_PREFIX = "ppg"  # arrays starting with this are labels (raw or preprocessed PPG)


@dataclass(frozen=True, eq=False)
class CachedVideo:
    """The arrays of one recording as they are passed to a method.

    Attributes:
        key: Identity of the recording.
        fps: Frame rate, in frames per second.
        arrays: The arrays required by the method, by name, without label arrays for
            test recordings. They are read-only memory maps of the cache files; an
            array must be copied before it is modified in place.
    """

    key: RecordingKey
    fps: float
    arrays: dict[str, np.ndarray]


def select_recordings(
    subjects: tuple[str, ...], keys: list[RecordingKey]
) -> list[RecordingKey]:
    """Selects the recordings of the given subjects.

    Args:
        subjects: Subject IDs of the form `"<dataset>/<subject>"`, e.g. one part of a
            split.
        keys: All recordings on disk.

    Returns:
        The keys whose `subject_id` is in `subjects`, in the order of `keys`.

    Raises:
        ValueError: If a subject has no recording in `keys`.
    """
    wanted_subjects = set(subjects)
    selected_keys = [key for key in keys if key.subject_id in wanted_subjects]
    if missing := wanted_subjects - {key.subject_id for key in selected_keys}:
        raise ValueError(
            f"{len(missing)} split subjects have no recording on disk "
            f"(e.g. {sorted(missing)[0]}); check the dataset root and the split file"
        )
    return selected_keys


def _read_arrays(
    key: RecordingKey, roi: StageCache, preprocessing: dict[str, StageCache]
) -> tuple[dict, dict]:
    """Reads all arrays of one recording from the caches and the ROI cache metadata.

    Args:
        key: The recording to read.
        roi: The ROI cache.
        preprocessing: The preprocessing caches by step name.

    Returns:
        A tuple `(arrays, meta)` of all arrays by name and the `meta.json` of the ROI
        cache.

    Raises:
        ValueError: If an array name is in more than one cache.
    """
    arrays, roi_meta = roi.read_recording(key)
    for cache in preprocessing.values():
        step_arrays, _ = cache.read_recording(key)
        if clash := arrays.keys() & step_arrays.keys():
            raise ValueError(f"{key}: {sorted(clash)} in more than one cache")
        arrays |= step_arrays
    return arrays, roi_meta


def load_cached_videos(
    keys: list[RecordingKey],
    roi: StageCache,
    preprocessing: dict[str, StageCache],
    needs: tuple[str, ...],
    with_labels: bool,
) -> list[CachedVideo]:
    """Loads the arrays that a method requires for each recording.

    Args:
        keys: The recordings to load.
        roi: The ROI cache, which also provides the frame rate.
        preprocessing: The preprocessing caches by step name, empty if the method
            requires only arrays from the ROI cache.
        needs: Names of the arrays to keep, i.e. the `needs` of the method.
        with_labels: Whether to keep label arrays (names beginning with
            `LABEL_PREFIX`). If `False`, as for test recordings, required label
            arrays are omitted without an error.

    Returns:
        One `CachedVideo` per key, in the order of `keys`.

    Raises:
        FileNotFoundError: If a recording is missing from a cache.
        ValueError: If an array name appears in more than one cache.
        KeyError: If a needed array is in no cache.
    """
    videos = []
    for key in keys:
        arrays, meta = _read_arrays(key, roi, preprocessing)
        if missing := [name for name in needs if name not in arrays]:
            raise KeyError(f"{key}: needs {missing}, caches have {sorted(arrays)}")
        kept_names = [
            name for name in needs if with_labels or not name.startswith(LABEL_PREFIX)
        ]
        videos.append(
            CachedVideo(
                key=key,
                fps=meta["fps"],
                arrays={name: arrays[name] for name in kept_names},
            )
        )
    return videos
