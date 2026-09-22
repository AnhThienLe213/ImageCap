"""Dataset adapters and persistence."""

from .flickr8k import CaptionRecord, Flickr8kDatasetBuilder, PickleDatasetRepository

__all__ = ["CaptionRecord", "Flickr8kDatasetBuilder", "PickleDatasetRepository"]