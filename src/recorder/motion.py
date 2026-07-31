from __future__ import annotations

import numpy as np


class MotionGate:
    """Detect coarse scene changes against a slowly updated background."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        delta: float = 12.0,
        min_area_fraction: float = 0.005,
        warmup_frames: int = 10,
        target_size: int = 80,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= delta <= 255.0:
            raise ValueError("delta must be between 0 and 255")
        if not 0.0 < min_area_fraction <= 1.0:
            raise ValueError("min_area_fraction must be between 0 and 1")
        if warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if target_size < 1:
            raise ValueError("target_size must be positive")
        self.alpha = alpha
        self.delta = delta
        self.min_area_fraction = min_area_fraction
        self.warmup_frames = warmup_frames
        self.target_size = target_size
        self._background: np.ndarray | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        self._background = None
        self._frames_seen = 0

    def has_motion(self, rgb_frame: np.ndarray) -> bool:
        frame = self._downsample_grayscale(rgb_frame)
        self._frames_seen += 1
        if self._background is None:
            self._background = frame.copy()
            return True

        difference = np.abs(frame - self._background)
        moving_fraction = float(np.mean(difference > self.delta))
        self._background *= 1.0 - self.alpha
        self._background += self.alpha * frame
        return (
            self._frames_seen <= self.warmup_frames
            or moving_fraction >= self.min_area_fraction
        )

    def _downsample_grayscale(self, rgb_frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(rgb_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError("motion frame must have shape HxWx3")
        rgb = frame.astype(np.float32, copy=False)
        grayscale = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )
        block_height = max(1, round(grayscale.shape[0] / self.target_size))
        block_width = max(1, round(grayscale.shape[1] / self.target_size))
        usable_height = grayscale.shape[0] - grayscale.shape[0] % block_height
        usable_width = grayscale.shape[1] - grayscale.shape[1] % block_width
        cropped = grayscale[:usable_height, :usable_width]
        return cropped.reshape(
            usable_height // block_height,
            block_height,
            usable_width // block_width,
            block_width,
        ).mean(axis=(1, 3))
