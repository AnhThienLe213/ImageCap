"""Flickr8k parsing and dataset persistence."""

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import random
from collections import defaultdict
from typing import Iterable, List


@dataclass(frozen=True)
class CaptionRecord:
    image: str
    captions: List[str]

    def as_training_item(self) -> dict:
        return {"image": self.image, "sentences": [{"raw": caption} for caption in self.captions]}


class Flickr8kDatasetBuilder:
    """Build deterministic train/validation records from Flickr8k captions."""

    def __init__(self, image_dir: Path, train_ratio: float = 0.85, seed: int = 42) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        self.image_dir = Path(image_dir)
        self.train_ratio = train_ratio
        self.seed = seed

    def read(self, captions_file: Path) -> List[CaptionRecord]:
        grouped = defaultdict(list)
        with Path(captions_file).open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    grouped[parts[0]].append(parts[2].strip())

        return [
            CaptionRecord(str(self.image_dir / image_name), captions[:5])
            for image_name, captions in grouped.items()
            if len(captions) >= 5
        ]

    def split(self, records: Iterable[CaptionRecord]) -> tuple[List[dict], List[dict]]:
        items = list(records)
        random.Random(self.seed).shuffle(items)
        split_at = int(len(items) * self.train_ratio)
        return (
            [record.as_training_item() for record in items[:split_at]],
            [record.as_training_item() for record in items[split_at:]],
        )

    def build(self, captions_file: Path) -> tuple[List[dict], List[dict]]:
        return self.split(self.read(captions_file))


class PickleDatasetRepository:
    """Persistence boundary for serialized train/validation datasets."""

    @staticmethod
    def save(records: Iterable[dict], path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(list(records), file)

    @staticmethod
    def load(path: Path) -> list[dict]:
        with Path(path).open("rb") as file:
            return pickle.load(file)