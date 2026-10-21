"""Stage 2: preprocess the ROI videos of one dataset into preprocessing caches.

Usage: uv run python scripts/2_preprocess.py [dataset=ubfc_rppg] [roi=yolo5face36_tight]
       [preprocessing=[rgb_mean]]

One cache per selected step (preprocessing=[standardized,diff_normalized] builds two),
so a cache built once serves every method that reads it.

The ROI cache is found from the dataset + roi config, so use the same ones as for
1_extract_roi.py. Videos already in the cache are skipped, so a crashed job can be rerun;
overwrite=true rebuilds them anyway.
"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rppgkit.cache import StageCache, preprocessing_caches

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    """Builds the cache of every selected preprocessing step.

    Args:
        cfg: The composed Hydra config (configs/preprocess.yaml), with command-line overrides.
    """
    # --- Dataset: the reader and every recording it finds on disk.
    dataset_reader = instantiate(cfg.dataset)
    recording_keys = dataset_reader.list_recording_keys()
    log.info("dataset %s: %d recordings", dataset_reader.name, len(recording_keys))

    # --- ROI cache: the input of every step; must hold every recording.
    roi = StageCache.for_roi(cfg)
    roi.check_has_recordings(recording_keys)
    log.info("roi cache: %s", roi.dir)

    # --- Preprocess, one step at a time: the step, the cache it writes to and which
    # recordings are not in that cache yet, then each recording read, processed and
    # written.
    for step_name, cache in preprocessing_caches(cfg, roi).items():
        preprocessing_step = instantiate(cfg.preprocessing[step_name])
        cache.write_config()
        log.info("%s cache: %s", step_name, cache.dir)
        recordings_to_build = (
            recording_keys
            if cfg.overwrite
            else cache.recordings_not_in_cache(recording_keys)
        )
        log.info(
            "%s: building %d/%d recordings (overwrite=%s)",
            step_name,
            len(recordings_to_build),
            len(recording_keys),
            cfg.overwrite,
        )
        for i, recording_key in enumerate(recordings_to_build, 1):
            roi_arrays, roi_meta = roi.read_recording(recording_key)
            cache.write_recording(
                recording_key,
                preprocessing_step(roi_arrays["frames"], roi_arrays["ppg"]),
                {
                    "recording": str(recording_key),
                    "subject_id": recording_key.subject_id,
                    "fps": roi_meta["fps"],
                },
            )
            log.info(
                "%s [%d/%d] %s: done",
                step_name,
                i,
                len(recordings_to_build),
                recording_key,
            )


if __name__ == "__main__":
    main()
