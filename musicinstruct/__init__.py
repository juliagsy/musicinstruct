"""MIDI-Instruct: instruction-conditioned MIDI editing with dual metrics."""

from .evaluation import score_records
from .metrics import score_item
from .schema import BenchmarkItem, Plan, Prediction

__all__ = ["BenchmarkItem", "Plan", "Prediction", "score_item", "score_records"]
