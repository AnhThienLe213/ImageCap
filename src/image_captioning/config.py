"""Configuration objects shared by the image-captioning pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Locations and split settings for Flickr8k."""

    root: Path
    captions_file: str = "captions.txt"
    images_dir: str = "Images"
    train_ratio: float = 0.85
    seed: int = 42

    @property
    def captions_path(self) -> Path:
        return self.root / self.captions_file

    @property
    def image_path(self) -> Path:
        return self.root / self.images_dir


@dataclass(frozen=True)
class CaptioningConfig:
    """Application-level configuration for both training pipelines."""

    dataset: DatasetConfig
    artifacts_dir: Path = Path("artifacts")
    max_length: int = 32
    batch_size: int = 32
    epochs: int = 5
    device: str = "auto"
    beam_width: int = 5
    early_exit_layers: Tuple[int, ...] = field(default_factory=lambda: (1, 3, 6, 9, 11))

    def resolved_device(self) -> str:
        """Resolve ``auto`` without importing a framework at module import time."""
        if self.device != "auto":
            return self.device

        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"