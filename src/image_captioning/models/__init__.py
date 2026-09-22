"""Model implementations used by the training pipelines."""

from .baseline import VGG16LSTMCaptioner
from .transformer import EarlyExitHead, TransformerCaptioner

__all__ = ["EarlyExitHead", "TransformerCaptioner", "VGG16LSTMCaptioner"]