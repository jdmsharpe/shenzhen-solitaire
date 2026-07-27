"""Shared SHENZHEN I/O card vocabulary and domain helpers."""

from __future__ import annotations

from collections import Counter
from typing import TypeGuard

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


def is_number_card(card: str | None) -> TypeGuard[str]:
    """Return whether ``card`` is a numbered suit card in the game.

    Returns a ``TypeGuard`` rather than a plain ``bool`` so that guarding on it
    also narrows away ``None``.  Every caller that asks this question goes on
    to call ``card_suit`` or ``card_rank``, and those need a ``str``.
    """

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


def board_errors(
    columns: tuple[tuple[str, ...], ...],
    cells: tuple[str | None, ...],
    foundations: tuple[int, ...],
    flower_done: bool,
    dragons_done: tuple[bool, ...],
) -> list[str]:
    """Return the reasons these values are not a legal 40-card position.

    Every card is either visible on the board or accounted for by a foundation
    rank, a cleared dragon pile, or the flower slot. Anything else means the
    position was misread or mistyped. Returns reasons rather than raising so
    each caller can report them in its own terms.
    """

    visible = Counter(card for column in columns for card in column)
    visible.update(card for card in cells if card not in (None, BLOCKED_CELL))
    errors: list[str] = []

    for suit_index, suit in enumerate(SUITS):
        for rank in range(1, MAX_NUMBER_RANK + 1):
            expected = int(rank > foundations[suit_index])
            actual = visible[f"{suit}{rank}"]
            if actual != expected:
                errors.append(
                    f"{suit}{rank}: expected {expected} visible, found {actual}"
                )

    for dragon_index, dragon in enumerate(DRAGONS):
        expected = 0 if dragons_done[dragon_index] else DRAGONS_PER_SET
        if visible[dragon] != expected:
            errors.append(
                f"{dragon}: expected {expected} visible, found {visible[dragon]}"
            )

    expected_flowers = 0 if flower_done else 1
    if visible[FLOWER] != expected_flowers:
        errors.append(
            f"{FLOWER}: expected {expected_flowers} visible, found {visible[FLOWER]}"
        )

    blocked = sum(card == BLOCKED_CELL for card in cells)
    if blocked != sum(dragons_done):
        errors.append(f"expected {sum(dragons_done)} blocked cells, found {blocked}")

    return errors


def summarize_errors(errors: list[str], limit: int = 8) -> str:
    """Join validation reasons into one line, capping how many are shown."""

    details = "; ".join(errors[:limit])
    if len(errors) > limit:
        details += f"; and {len(errors) - limit} more"
    return details
