"""TV-SGI with an SOFTS traffic forecaster and unchanged gradient modules.

Original TSJT data engine and TSB gradient filter are retained, while
SOFTS replaces the Graph WaveNet forward model.
"""
from .models.pgrad_model import TSJTPGrad
__all__ = ["TSJTPGrad"]
