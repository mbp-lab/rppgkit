"""Reader for the UBFC-rPPG dataset.

Layout: <root>/subject<N>/vid.avi and <root>/subject<N>/ground_truth.txt.
ground_truth.txt has three whitespace-separated lines: PPG, HR (bpm), timestamps (s).
One recording per subject, so its key is RecordingKey("ubfc_rppg", "subject<N>", "0").

Reference: Bobbia et al., "Unsupervised skin tissue segmentation for remote
photoplethysmography", Pattern Recognition Letters, 2019.
"""

import re
from pathlib import Path

import numpy as np

from rppgkit.datasets.reader import DatasetReader, read_video_file, resample_ppg
from rppgkit.recording import RecordingKey, VideoRecord


class UBFCrPPGReader(DatasetReader):
    """Reader for UBFC-rPPG; one `VideoRecord` per subject folder.

    Args:
        name: `RecordingKey.dataset` of all recordings, and the cache folder name.
        root: Folder that contains the subject<N> folders.

    Raises:
        FileNotFoundError: If `root` is not a folder.
    """

    version = 1

    def __init__(self, name: str, root: str | Path) -> None:
        self.name = name
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"UBFC-rPPG root not found: {self.root}")

    def list_recording_keys(self) -> list[RecordingKey]:
        """Lists the recordings on disk (one per subject) without decoding.

        Returns:
            One key per subject<N> folder that holds a vid.avi, sorted by N.

        Raises:
            FileNotFoundError: If no subject<N>/vid.avi is found under `root`.
        """
        subjects = [
            d.name
            for d in self.root.iterdir()
            if re.fullmatch(r"subject\d+", d.name) and (d / "vid.avi").is_file()
        ]
        if not subjects:
            raise FileNotFoundError(
                f"no subject<N>/vid.avi under {self.root}; root must be the folder "
                f"that contains the subject folders"
            )
        subjects.sort(key=lambda s: int(s.removeprefix("subject")))
        return [RecordingKey(self.name, s, "0") for s in subjects]

    def read(self, key: RecordingKey) -> VideoRecord:
        """Decodes one video and resamples its ground_truth.txt PPG to the frames.

        Args:
            key: A UBFC-rPPG key from `list_recording_keys`.

        Returns:
            The recording; `meta["n_ppg_samples"]` is the PPG length before resampling.

        Raises:
            ValueError: If `key` belongs to another dataset.
        """
        if key.dataset != self.name:
            raise ValueError(f"not a {self.name} recording: {key}")
        folder = self.root / key.subject
        frames, fps = read_video_file(folder / "vid.avi")
        ppg = _read_ground_truth(folder / "ground_truth.txt")
        return VideoRecord(
            key=key,
            frames=frames,
            ppg=resample_ppg(ppg, len(frames)),
            fps=fps,
            meta={"n_ppg_samples": len(ppg)},
        )


def _read_ground_truth(path: Path) -> np.ndarray:
    """Returns the PPG values: the first line of ground_truth.txt."""
    first_line = path.read_text().splitlines()[0]
    return np.array(first_line.split(), dtype=np.float64)
