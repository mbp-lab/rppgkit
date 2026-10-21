"""Stage 1: extract the region of interest (e.g. the face) of every recording.

Usage: uv run python scripts/1_extract_roi.py [dataset=ubfc_rppg] [roi=yolo5face36_tight] [roi.output_size=128]

Writes the ROI cache. Videos already in the cache are skipped, so a crashed job can
just be rerun; overwrite=true rebuilds them anyway.
"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rppgkit.cache import StageCache

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="extract_roi", version_base=None)
def main(cfg: DictConfig) -> None:
    """Extracts the ROI of every recording not yet in the ROI cache.

    Args:
        cfg: The composed Hydra config (configs/extract_roi.yaml), with command-line overrides.
    """
    # --- Dataset: the reader and every recording it finds on disk.
    dataset_reader = instantiate(cfg.dataset)
    recording_keys = dataset_reader.list_recording_keys()
    log.info("dataset %s: %d recordings", dataset_reader.name, len(recording_keys))

    # --- ROI: the extractor (e.g. the face detector) and the cache it writes to,
    # and which recordings are not in that cache yet.
    roi_extractor = instantiate(cfg.roi)
    cache = StageCache.for_roi(cfg)
    cache.write_config()
    log.info("roi cache: %s", cache.dir)
    recordings_to_build = (
        recording_keys
        if cfg.overwrite
        else cache.recordings_not_in_cache(recording_keys)
    )
    log.info(
        "building %d/%d recordings (overwrite=%s)",
        len(recordings_to_build),
        len(recording_keys),
        cfg.overwrite,
    )

    # --- Extract: read each recording, crop its ROI, write it to the cache.
    # TODO: This loop can be adapted to run in parallel
    for i, recording_key in enumerate(recordings_to_build, 1):
        roi_video, boxes, detection_stats = roi_extractor(
            dataset_reader.read(recording_key)
        )
        cache.write_recording(
            recording_key,
            {"frames": roi_video.frames, "ppg": roi_video.ppg, "boxes": boxes},
            {
                "recording": str(recording_key),
                "subject_id": recording_key.subject_id,
                "fps": roi_video.fps,
                "meta": roi_video.meta,
                "stats": detection_stats,
            },
        )
        n_detected = detection_stats["n_detections"] - detection_stats["n_failed"]
        log.info(
            "[%d/%d] %s: face detected in %d/%d frames tried (%.1f%%)",
            i,
            len(recordings_to_build),
            recording_key,
            n_detected,
            detection_stats["n_detections"],
            100 * n_detected / detection_stats["n_detections"],
        )


if __name__ == "__main__":
    main()
