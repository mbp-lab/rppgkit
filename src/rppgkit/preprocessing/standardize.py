"""Per-video z-score normalisation of the face frames and the PPG label.

One scalar mean and standard deviation are computed per video over all pixels and
channels. Normalisation per channel or per chunk would constitute a different step.
"""

import numpy as np

from rppgkit.preprocessing.step import PreprocessingStep


class Standardize(PreprocessingStep):
    """Z-scores frames and PPG separately, each over the complete video.

    Attributes:
        frames: Name of the frames output, a network input.
        label: Name of the PPG output, a training target.
    """

    frames = "frames_standardized"
    label = "ppg_standardized"
    version = 1

    def __call__(self, frames: np.ndarray, ppg: np.ndarray) -> dict[str, np.ndarray]:
        """Standardises the frames and the PPG of one video independently.

        Args:
            frames: Face video, `uint8` RGB, shape `(T, H, W, 3)`.
            ppg: Raw PPG, shape `(T,)`.

        Returns:
            `{"frames_standardized": frames, "ppg_standardized": ppg}`, both of type
            `float32`, with shapes `(T, H, W, 3)` and `(T,)`.
        """
        return {
            self.frames: self.standardize(frames),
            self.label: self.standardize(ppg),
        }

    @staticmethod
    def to_pulse(prediction: np.ndarray) -> np.ndarray:
        """Returns a prediction of `label` unchanged, as it already has pulse shape."""
        return prediction

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """Z-scores an array with one mean and standard deviation over all values.

        Args:
            data: A numeric array of any shape.

        Returns:
            `(data - mean) / std` as `float32`, with the shape of `data`. A constant
            array (standard deviation 0) yields zeros.
        """
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data.astype(np.float32)
