"""Public package interface for CRFE."""

from ._crfe import CRFE
from .stopping import ParamParada, StoppingCriteria

__all__ = ["CRFE", "ParamParada", "StoppingCriteria"]
