"""Fixed-length training chunks cut from whole-video caches at load time.

The caches store whole videos, so one cache serves any chunk length and overlap. The
dataset only indexes (video, start); each item slices the memory-mapped arrays and
hands the chunk to the method's transform. DataLoader(shuffle=True) shuffles the index.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from rppgkit.cached_videos import CachedVideo


class ChunkDataset(Dataset):
    """All chunks of `length` frames starting every `stride` frames; the remainder of a
    video shorter than one more chunk is dropped."""

    def __init__(
        self,
        videos: list[CachedVideo],
        length: int,
        stride: int,
        transform: Callable[[dict[str, np.ndarray]], dict[str, torch.Tensor]],
    ) -> None:
        if length < 1 or stride < 1:
            raise ValueError(f"length and stride must be >= 1, got {length}, {stride}")
        self.videos = videos
        self.length = length
        self.transform = transform
        self.chunk_starts = [
            (video_index, start)
            for video_index, video in enumerate(videos)
            for start in range(0, _n_frames(video) - length + 1, stride)
        ]
        if not self.chunk_starts:
            raise ValueError(
                f"no chunk of {length} frames in {len(videos)} videos "
                f"(all shorter than one chunk, or no videos)"
            )

    def __len__(self) -> int:
        return len(self.chunk_starts)

    def __getitem__(self, chunk_index: int) -> dict[str, torch.Tensor]:
        video_index, start = self.chunk_starts[chunk_index]
        arrays = self.videos[video_index].arrays
        # Every loaded array (the method's `needs`, labels included for train/val) is
        # cut at the same frames; the method's transform picks input and target by name.
        return self.transform(
            {name: array[start : start + self.length] for name, array in arrays.items()}
        )


def _n_frames(video: CachedVideo) -> int:
    """Frame count; every array of one video has one row per frame."""
    lengths = {len(array) for array in video.arrays.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"{video.key}: arrays have different lengths {sorted(lengths)}"
        )
    return lengths.pop()
