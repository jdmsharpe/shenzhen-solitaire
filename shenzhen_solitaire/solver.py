#!/usr/bin/env python3
"""
Dependency-free SHENZHEN I/O Solitaire solver.

Card notation
-------------
D1..D9 : red dots
B1..B9 : green bamboo
C1..C9 : black characters
RD/GD/WD : red, green, and white dragons
FL : flower

Each tableau column is written from bottom to top, so the final item is
the exposed card.

Set ``START_COLUMNS`` to any valid deal before running the script.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count, pairwise
from pathlib import Path

from .config import (
    BLOCKED_CELL,
    DRAGONS,
    DRAGONS_PER_SET,
    FLOWER,
    FREE_CELL_COUNT,
    MAX_NUMBER_RANK,
    SUITS,
    card_rank,
    card_suit,
    foundation_index,
    is_number_card,
)


@dataclass(frozen=True)
class State:
    """Immutable snapshot of every card location and completed set."""

    # Each column is bottom -> top.
    columns: tuple[tuple[str, ...], ...]
    cells: tuple[str | None, str | None, str | None]
    # Foundation ranks in D, B, C order.
    foundations: tuple[int, int, int]
    flower_done: bool
    # Cleared dragon sets in R, G, W order.
    dragons_done: tuple[bool, bool, bool]


@dataclass(frozen=True)
class Edge:
    """A manual move and any automatic moves resulting from it."""

    state: State
    action: str
    automatic_actions: tuple[str, ...]


def can_stack(card: str, onto: str | None) -> bool:
    """Whether `card` may be placed on top of `onto` in the tableau."""
    if onto is None:
        return True
    if not is_number_card(card) or not is_number_card(onto):
        return False
    return card_rank(card) + 1 == card_rank(onto) and card_suit(card) != card_suit(onto)


def safe_for_automatic_foundation(state: State, card: str) -> bool:
    """
    Conservative FreeCell-style auto-play rule.

    Ranks 1 and 2 are always safe. A higher card is auto-played only when
    both other suits have already reached at least rank-1, so the card is
    no longer needed as a temporary tableau landing point.
    """
    rank = card_rank(card)
    if rank <= 2:
        return True

    own_suit = foundation_index(card)
    return all(
        foundation_rank >= rank - 1
        for index, foundation_rank in enumerate(state.foundations)
        if index != own_suit
    )


def put_on_foundation(
    state: State, source_kind: str, source_index: int
) -> tuple[State, str]:
    """Move the selected number card to its foundation."""

    columns = [list(column) for column in state.columns]
    cells = list(state.cells)
    foundations = list(state.foundations)

    if source_kind == "C":
        card = columns[source_index].pop()
    elif source_kind == "F":
        card = cells[source_index]
        cells[source_index] = None
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")

    if card is None or not is_number_card(card):
        raise ValueError("Only number cards can enter a foundation")

    foundation = foundation_index(card)
    expected_rank = foundations[foundation] + 1
    if card_rank(card) != expected_rank:
        raise ValueError(f"{card} cannot advance its foundation")

    foundations[foundation] += 1
    return (
        State(
            columns=tuple(tuple(column) for column in columns),
            cells=tuple(cells),  # type: ignore[arg-type]
            foundations=tuple(foundations),  # type: ignore[arg-type]
            flower_done=state.flower_done,
            dragons_done=state.dragons_done,
        ),
        card,
    )


def _move_flower_to_foundation(state: State) -> tuple[State, str | None]:
    if state.flower_done:
        return state, None

    for column_index, column in enumerate(state.columns):
        if column and column[-1] == FLOWER:
            columns = [list(item) for item in state.columns]
            columns[column_index].pop()
            return (
                State(
                    columns=tuple(tuple(item) for item in columns),
                    cells=state.cells,
                    foundations=state.foundations,
                    flower_done=True,
                    dragons_done=state.dragons_done,
                ),
                f"{FLOWER} C{column_index + 1} -> flower foundation",
            )

    for cell_index, card in enumerate(state.cells):
        if card == FLOWER:
            cells = list(state.cells)
            cells[cell_index] = None
            return (
                State(
                    columns=state.columns,
                    cells=tuple(cells),  # type: ignore[arg-type]
                    foundations=state.foundations,
                    flower_done=True,
                    dragons_done=state.dragons_done,
                ),
                f"{FLOWER} F{cell_index + 1} -> flower foundation",
            )

    return state, None


def _automatic_foundation_candidates(state: State) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []

    for column_index, column in enumerate(state.columns):
        if not column or not is_number_card(column[-1]):
            continue

        card = column[-1]
        foundation = foundation_index(card)
        if card_rank(card) == state.foundations[
            foundation
        ] + 1 and safe_for_automatic_foundation(state, card):
            candidates.append(("C", column_index, card))

    for cell_index, card in enumerate(state.cells):
        if not is_number_card(card):
            continue

        foundation = foundation_index(card)
        if card_rank(card) == state.foundations[
            foundation
        ] + 1 and safe_for_automatic_foundation(state, card):
            candidates.append(("F", cell_index, card))

    return candidates


def normalize_automatic_moves(state: State) -> tuple[State, tuple[str, ...]]:
    """
    Apply moves that the game can safely auto-play.

    This greatly reduces the search graph: many visually different states
    collapse to the same normalized state.
    """
    automatic: list[str] = []

    while True:
        # The flower has only one destination and can always leave immediately.
        state, flower_action = _move_flower_to_foundation(state)
        if flower_action is not None:
            automatic.append(flower_action)
            continue

        candidates = _automatic_foundation_candidates(state)
        if not candidates:
            return state, tuple(automatic)

        source_kind, source_index, card = candidates[0]
        state, _ = put_on_foundation(state, source_kind, source_index)
        automatic.append(f"{card} {source_kind}{source_index + 1} -> foundation")


def movable_run_starts(column: tuple[str, ...]) -> list[int]:
    """
    Return the start indices of every legal movable suffix.

    Runs are descending by one and adjacent cards must have different suits.
    """
    if not column:
        return []

    starts = [len(column) - 1]
    index = len(column) - 2

    while index >= 0 and can_stack(column[index + 1], column[index]):
        starts.append(index)
        index -= 1

    # Longest run first tends to expose buried cards faster.
    return list(reversed(starts))


def _dragon_clear_edges(state: State) -> Iterator[Edge]:
    # Clear a dragon set when all four matching dragons are exposed.
    for dragon_index, dragon in enumerate(DRAGONS):
        if state.dragons_done[dragon_index]:
            continue

        cell_sources = [
            index for index, card in enumerate(state.cells) if card == dragon
        ]
        column_sources = [
            index
            for index, column in enumerate(state.columns)
            if column and column[-1] == dragon
        ]

        if len(cell_sources) + len(column_sources) != DRAGONS_PER_SET:
            continue

        destination_cell = (
            cell_sources[0]
            if cell_sources
            else next(
                (index for index, card in enumerate(state.cells) if card is None),
                None,
            )
        )
        if destination_cell is None:
            continue

        columns = [list(column) for column in state.columns]
        cells = list(state.cells)

        for column_index in column_sources:
            columns[column_index].pop()
        for cell_index in cell_sources:
            cells[cell_index] = None

        cells[destination_cell] = BLOCKED_CELL
        dragons_done = list(state.dragons_done)
        dragons_done[dragon_index] = True

        next_state = State(
            columns=tuple(tuple(column) for column in columns),
            cells=tuple(cells),  # type: ignore[arg-type]
            foundations=state.foundations,
            flower_done=state.flower_done,
            dragons_done=tuple(dragons_done),  # type: ignore[arg-type]
        )
        next_state, automatic = normalize_automatic_moves(next_state)
        yield Edge(next_state, f"clear {dragon}", automatic)


def _foundation_edges(state: State) -> Iterator[Edge]:
    # Explicit tableau-to-foundation moves. Unsafe cards remain optional rather
    # than being forced by normalize_automatic_moves().
    for column_index, column in enumerate(state.columns):
        if not column or not is_number_card(column[-1]):
            continue

        card = column[-1]
        foundation = foundation_index(card)
        if card_rank(card) == state.foundations[foundation] + 1:
            next_state, _ = put_on_foundation(state, "C", column_index)
            next_state, automatic = normalize_automatic_moves(next_state)
            yield Edge(
                next_state,
                f"{card} C{column_index + 1} -> foundation",
                automatic,
            )

    # Free-cell-to-foundation moves.
    for cell_index, card in enumerate(state.cells):
        if not is_number_card(card):
            continue

        foundation = foundation_index(card)
        if card_rank(card) == state.foundations[foundation] + 1:
            next_state, _ = put_on_foundation(state, "F", cell_index)
            next_state, automatic = normalize_automatic_moves(next_state)
            yield Edge(
                next_state,
                f"{card} F{cell_index + 1} -> foundation",
                automatic,
            )


def _tableau_run_edges(state: State) -> Iterator[Edge]:
    # Tableau run to tableau.
    for source_index, source_column in enumerate(state.columns):
        if not source_column:
            continue

        for run_start in movable_run_starts(source_column):
            moving = source_column[run_start:]
            base_card = moving[0]

            for destination_index, destination_column in enumerate(state.columns):
                if destination_index == source_index:
                    continue

                if not destination_column:
                    # Moving an entire column to another empty column is merely
                    # a column renaming and creates a large symmetry loop.
                    if run_start == 0:
                        continue
                    legal = True
                else:
                    legal = can_stack(base_card, destination_column[-1])

                if not legal:
                    continue

                columns = [list(column) for column in state.columns]
                columns[source_index] = columns[source_index][:run_start]
                columns[destination_index].extend(moving)

                next_state = State(
                    columns=tuple(tuple(column) for column in columns),
                    cells=state.cells,
                    foundations=state.foundations,
                    flower_done=state.flower_done,
                    dragons_done=state.dragons_done,
                )
                next_state, automatic = normalize_automatic_moves(next_state)
                run_name = "/".join(moving)
                yield Edge(
                    next_state,
                    (f"{run_name} C{source_index + 1} -> C{destination_index + 1}"),
                    automatic,
                )


def _free_cell_to_tableau_edges(state: State) -> Iterator[Edge]:
    # Free cell to tableau.
    for cell_index, card in enumerate(state.cells):
        if card is None or card == BLOCKED_CELL:
            continue

        for destination_index, destination_column in enumerate(state.columns):
            if card in DRAGONS:
                legal = not destination_column
            else:
                legal = not destination_column or can_stack(
                    card, destination_column[-1]
                )

            if not legal:
                continue

            cells = list(state.cells)
            cells[cell_index] = None
            columns = [list(column) for column in state.columns]
            columns[destination_index].append(card)

            next_state = State(
                columns=tuple(tuple(column) for column in columns),
                cells=tuple(cells),  # type: ignore[arg-type]
                foundations=state.foundations,
                flower_done=state.flower_done,
                dragons_done=state.dragons_done,
            )
            next_state, automatic = normalize_automatic_moves(next_state)
            yield Edge(
                next_state,
                f"{card} F{cell_index + 1} -> C{destination_index + 1}",
                automatic,
            )


def _tableau_to_free_cell_edges(state: State) -> Iterator[Edge]:
    # Tableau top to a free cell. Empty free cells are strategically
    # interchangeable, so only the first one is generated.
    empty_cell = next(
        (index for index, card in enumerate(state.cells) if card is None),
        None,
    )
    if empty_cell is not None:
        for source_index, source_column in enumerate(state.columns):
            if not source_column:
                continue

            card = source_column[-1]
            columns = [list(column) for column in state.columns]
            columns[source_index].pop()
            cells = list(state.cells)
            cells[empty_cell] = card

            next_state = State(
                columns=tuple(tuple(column) for column in columns),
                cells=tuple(cells),  # type: ignore[arg-type]
                foundations=state.foundations,
                flower_done=state.flower_done,
                dragons_done=state.dragons_done,
            )
            next_state, automatic = normalize_automatic_moves(next_state)
            yield Edge(
                next_state,
                f"{card} C{source_index + 1} -> F{empty_cell + 1}",
                automatic,
            )


def neighbours(state: State) -> Iterator[Edge]:
    """Generate every strategically relevant legal one-click move."""
    yield from _dragon_clear_edges(state)
    yield from _foundation_edges(state)
    yield from _tableau_run_edges(state)
    yield from _free_cell_to_tableau_edges(state)
    yield from _tableau_to_free_cell_edges(state)


def heuristic(state: State) -> int:
    """
    A fast, intentionally non-admissible A* heuristic.

    It counts unfinished cards and adds a penalty for broken tableau
    adjacencies. The heuristic is aimed at finding a good solution quickly,
    not proving that the solution uses the absolute fewest clicks.
    """
    remaining = sum(MAX_NUMBER_RANK - rank for rank in state.foundations)

    if not state.flower_done:
        remaining += 1

    remaining += DRAGONS_PER_SET * sum(not done for done in state.dragons_done)

    disorder = 0
    for column in state.columns:
        for lower, upper in pairwise(column):
            if not can_stack(upper, lower):
                disorder += 1

    return remaining + disorder


def is_goal(state: State) -> bool:
    """Return whether all cards have reached their destinations."""

    return (
        state.foundations == (MAX_NUMBER_RANK,) * len(SUITS)
        and state.flower_done
        and all(state.dragons_done)
    )


def _reconstruct_solution(
    state: State,
    parent: dict[State, State | None],
    action: dict[State, tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
    result: list[tuple[str, tuple[str, ...]]] = []

    while (previous_state := parent[state]) is not None:
        result.append(action[state])
        state = previous_state

    result.reverse()
    return result


def solve(
    initial_state: State, max_states: int = 500_000
) -> tuple[list[tuple[str, tuple[str, ...]]], int]:
    """
    Run A* and return [(manual_action, automatic_actions), ...], state_count.
    """
    # Normalize every supplied setup, including deals with immediately playable
    # cards. The solver is therefore independent of when the state was captured.
    initial_state, _ = normalize_automatic_moves(initial_state)

    queue: list[tuple[int, int, int, State]] = []
    serial = count()
    heappush(queue, (heuristic(initial_state), 0, next(serial), initial_state))

    parent: dict[State, State | None] = {initial_state: None}
    action: dict[State, tuple[str, tuple[str, ...]]] = {}
    best_cost: dict[State, int] = {initial_state: 0}
    explored = 0

    while queue and explored < max_states:
        _, cost, _, state = heappop(queue)

        if cost != best_cost.get(state):
            continue

        explored += 1
        if is_goal(state):
            return _reconstruct_solution(state, parent, action), explored

        for edge in neighbours(state):
            next_cost = cost + 1
            if next_cost >= best_cost.get(edge.state, 10**9):
                continue

            best_cost[edge.state] = next_cost
            parent[edge.state] = state
            action[edge.state] = (edge.action, edge.automatic_actions)
            priority = next_cost + heuristic(edge.state)
            heappush(
                queue,
                (priority, next_cost, next(serial), edge.state),
            )

    raise RuntimeError(f"No solution found after exploring {explored:,} states")


# Starting board, bottom -> top in each column. Replace this with any valid deal.
START_COLUMNS = (
    ("B2", "D7", "B5", "RD", "B1"),
    ("WD", "D6", "B6", "B8", "C1"),
    ("C5", "D8", "D4", "B9", "D5"),
    ("GD", "C9", "D1", "B4", "B3"),
    ("C8", "WD", "RD", "RD", "C6"),
    ("C7", "GD", "RD", "WD", "C3"),
    ("FL", "GD", "WD", "D3", "GD"),
    ("C2", "D9", "D2", "B7", "C4"),
)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a SHENZHEN I/O Solitaire deal from columns or a screenshot."
    )
    parser.add_argument(
        "screenshot",
        nargs="?",
        type=Path,
        help="game screenshot to recognize; omit to use START_COLUMNS",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("shenzhen_reference.png"),
        help="labeled reference screenshot for the current game theme",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        help="write an image showing every recognized card and confidence score",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=500_000,
        help="maximum search states (default: 500000)",
    )
    return parser.parse_args(argv)


def _state_from_arguments(arguments: argparse.Namespace) -> State:
    if arguments.screenshot is None:
        return State(
            columns=START_COLUMNS,
            cells=(None,) * FREE_CELL_COUNT,
            foundations=(0, 0, 0),
            flower_done=False,
            dragons_done=(False, False, False),
        )

    try:
        from .vision import ScreenshotRecognitionError, extract_state
    except ModuleNotFoundError as error:
        if error.name not in {"cv2", "numpy"}:
            raise
        raise SystemExit(
            "Screenshot recognition requires OpenCV. Run:\n"
            "  uv run --extra ocr shenzhen-solitaire screenshot.png"
        ) from error

    try:
        extracted = extract_state(
            arguments.screenshot,
            arguments.reference,
            arguments.debug_image,
        )
    except ScreenshotRecognitionError as error:
        raise SystemExit(f"Screenshot recognition failed: {error}") from error

    print("Detected columns (bottom -> top):")
    for index, column in enumerate(extracted.columns, start=1):
        print(f"  C{index}: {' '.join(column) or '(empty)'}")
    print(
        f"Cells: {extracted.cells}; foundations (D/B/C): "
        f"{extracted.foundations}; flower: "
        f"{'done' if extracted.flower_done else 'in play'}\n"
    )
    return State(
        columns=extracted.columns,
        cells=extracted.cells,
        foundations=extracted.foundations,
        flower_done=extracted.flower_done,
        dragons_done=extracted.dragons_done,
    )


def main(argv: list[str] | None = None) -> None:
    """Recognize the requested board, solve it, and print its move sequence."""

    arguments = _parse_arguments(argv)
    initial_state = _state_from_arguments(arguments)
    solution, explored = solve(initial_state, max_states=arguments.max_states)

    print(
        f"Found a solution with {len(solution)} manual actions "
        f"after exploring {explored:,} states.\n"
    )

    for number, (manual, automatic) in enumerate(solution, start=1):
        print(f"{number:2}. {manual}")
        for move in automatic:
            print(f"      auto: {move}")


if __name__ == "__main__":
    main()
