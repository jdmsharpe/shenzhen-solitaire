"""Machine-readable vocabulary for the moves a solution is made of.

A move is data, not prose.  ``str(move)`` renders the notation the solver
prints, while the fields stay available for replaying a solution, testing it,
or mapping it back onto screen coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

COLUMN = "column"
CELL = "cell"
FOUNDATION = "foundation"
FLOWER_FOUNDATION = "flower"

SlotKind = Literal["column", "cell", "foundation", "flower"]


@dataclass(frozen=True)
class Slot:
    """A place a card can occupy on the board.

    ``index`` identifies the tableau column, free cell, or suit foundation.
    The flower foundation is unique, so its index is always zero.
    """

    kind: SlotKind
    index: int = 0

    def __str__(self) -> str:
        match self.kind:
            case "column":
                return f"C{self.index + 1}"
            case "cell":
                return f"F{self.index + 1}"
            case "foundation":
                return "foundation"
            case _:
                return "flower foundation"


@dataclass(frozen=True)
class CardMove:
    """One or more cards dragged from one slot to another.

    ``cards`` is ordered bottom to top, so a multi-card tableau run keeps the
    order it is stacked in and ``cards[0]`` is the card that must land legally.
    """

    cards: tuple[str, ...]
    source: Slot
    destination: Slot

    def __str__(self) -> str:
        return f"{'/'.join(self.cards)} {self.source} -> {self.destination}"


@dataclass(frozen=True)
class DragonClear:
    """All four dragons of one colour collapsed into a free cell.

    The cell named by ``destination`` is covered by the resulting pile and
    holds ``BLOCKED_CELL`` afterwards.
    """

    dragon: str
    destination: Slot

    def __str__(self) -> str:
        return f"clear {self.dragon}"


Move = CardMove | DragonClear
