"""Solve and recognize SHENZHEN I/O Solitaire deals."""

from .moves import CardMove, DragonClear, Move, Slot
from .solver import (
    DealUnsolvable,
    IllegalMove,
    SearchBudgetExhausted,
    SolveFailed,
    State,
    apply_move,
    normalize_automatic_moves,
    solve,
)

__all__ = [
    "CardMove",
    "DealUnsolvable",
    "DragonClear",
    "IllegalMove",
    "Move",
    "SearchBudgetExhausted",
    "Slot",
    "SolveFailed",
    "State",
    "apply_move",
    "normalize_automatic_moves",
    "solve",
]
