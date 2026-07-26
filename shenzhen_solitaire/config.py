"""Shared SHENZHEN I/O card vocabulary and domain helpers."""

from __future__ import annotations

SUITS = ("D", "B", "C")
DRAGONS = ("RD", "GD", "WD")
FLOWER = "FL"
BLOCKED_CELL = "XX"

MAX_NUMBER_RANK = 9
DRAGONS_PER_SET = 4
FREE_CELL_COUNT = 3
TABLEAU_COLUMN_COUNT = 8

NUMBER_CARD_LABELS = frozenset(
    f"{suit}{rank}" for suit in SUITS for rank in range(1, MAX_NUMBER_RANK + 1)
)


def is_number_card(card: str | None) -> bool:
    """Return whether ``card`` is a numbered suit card in the game."""

    return card in NUMBER_CARD_LABELS


def card_suit(card: str) -> str:
    """Return a number card's one-letter suit code."""

    return card[0]


def card_rank(card: str) -> int:
    """Return a number card's numeric rank."""

    return int(card[1:])


def foundation_index(card: str) -> int:
    """Return the foundation index associated with ``card``'s suit."""

    return SUITS.index(card_suit(card))
