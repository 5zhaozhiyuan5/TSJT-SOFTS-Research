"""Research adapter for the official MTPB backbone, 288 history -> 36 forecast."""
from .adapter import MTPBGradientAdapter, MTPBGradientConfig
__all__ = ["MTPBGradientAdapter", "MTPBGradientConfig"]
