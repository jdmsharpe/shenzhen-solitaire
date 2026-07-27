"""Focused behavioral tests for the SHENZHEN I/O solver."""

import unittest
from collections import Counter

from shenzhen_solitaire.config import (
    BLOCKED_CELL,
    DRAGONS,
    DRAGONS_PER_SET,
    FLOWER,
    FREE_CELL_COUNT,
    MAX_NUMBER_RANK,
    SUITS,
    TABLEAU_COLUMN_COUNT,
)
from shenzhen_solitaire.moves import CELL, COLUMN, CardMove, DragonClear, Slot
from shenzhen_solitaire.solver import (
    START_COLUMNS,
    DealUnsolvable,
    IllegalMove,
    SearchBudgetExhausted,
    State,
    apply_move,
    can_stack,
    is_goal,
    normalize_automatic_moves,
    solve,
)

FULL_DECK = Counter(
    [f"{suit}{rank}" for suit in SUITS for rank in range(1, MAX_NUMBER_RANK + 1)]
    + [dragon for dragon in DRAGONS for _ in range(DRAGONS_PER_SET)]
    + [FLOWER]
)

# The deal recognized by test_shenzhen_vision, reused here so the replay check
# runs against a second independent board.
SECOND_DEAL = (
    ("D5", "C4", "GD", "WD", "RD"),
    ("GD", "RD", "C6", "C2", "B3"),
    ("D8", "D1", "GD", "C5", "D7"),
    ("C1", "B7", "B8", "WD", "GD"),
    ("B6", "FL", "B9", "D6", "C7"),
    ("B1", "C3", "B5", "B2", "WD"),
    ("D4", "D3", "RD", "B4", "D2"),
    ("WD", "C8", "RD", "C9", "D9"),
)


def deal(columns: tuple[tuple[str, ...], ...]) -> State:
    """Build the opening state for a freshly dealt board."""

    return State(
        columns=columns,
        cells=(None,) * FREE_CELL_COUNT,
        foundations=(0,) * len(SUITS),
        flower_done=False,
        dragons_done=(False,) * len(DRAGONS),
    )


def census(state: State) -> Counter:
    """Every card the state accounts for, in play or already finished."""

    seen = Counter(card for column in state.columns for card in column)
    seen.update(card for card in state.cells if card not in (None, BLOCKED_CELL))
    for suit_index, suit in enumerate(SUITS):
        for rank in range(1, state.foundations[suit_index] + 1):
            seen[f"{suit}{rank}"] += 1
    if state.flower_done:
        seen[FLOWER] += 1
    for dragon_index, dragon in enumerate(DRAGONS):
        if state.dragons_done[dragon_index]:
            seen[dragon] += DRAGONS_PER_SET
    return seen


class SolverTest(unittest.TestCase):
    def test_tableau_stacks_descend_across_different_suits(self) -> None:
        self.assertTrue(can_stack("B4", "D5"))
        self.assertFalse(can_stack("D4", "D5"))
        self.assertFalse(can_stack("B5", "D5"))

    def test_completed_state_needs_no_moves(self) -> None:
        state = State(
            columns=((),) * TABLEAU_COLUMN_COUNT,
            cells=(None,) * FREE_CELL_COUNT,
            foundations=(MAX_NUMBER_RANK,) * len(SUITS),
            flower_done=True,
            dragons_done=(True,) * len(DRAGONS),
        )

        solution, explored = solve(state)

        self.assertEqual(solution, [])
        self.assertEqual(explored, 1)


class SolutionReplayTest(unittest.TestCase):
    """Replay each solution through apply_move to prove the moves are legal.

    apply_move re-derives the rules independently of the edge generators, so
    agreement between the two is a real check rather than a tautology.
    """

    def assert_solution_wins(self, columns: tuple[tuple[str, ...], ...]) -> None:
        solution, _ = solve(deal(columns))
        state, _ = normalize_automatic_moves(deal(columns))

        for number, (move, expected_automatic) in enumerate(solution, start=1):
            with self.subTest(move=number, action=str(move)):
                state = apply_move(state, move)
                state, automatic = normalize_automatic_moves(state)
                self.assertEqual(
                    census(state), FULL_DECK, "a card was lost or duplicated"
                )
                self.assertEqual(
                    automatic,
                    expected_automatic,
                    "replay disagreed about the follow-on automatic moves",
                )

        self.assertTrue(is_goal(state), "replay did not reach a won position")

    def test_bundled_deal_solution_replays_to_a_win(self) -> None:
        self.assert_solution_wins(START_COLUMNS)

    def test_second_deal_solution_replays_to_a_win(self) -> None:
        self.assert_solution_wins(SECOND_DEAL)


class ApplyMoveTest(unittest.TestCase):
    def test_stacking_equal_ranks_is_rejected(self) -> None:
        state = deal(
            (("D5",), ("B5",), (), (), (), (), (), ()),
        )

        with self.assertRaises(IllegalMove):
            apply_move(
                state,
                CardMove(
                    cards=("B5",),
                    source=Slot(COLUMN, 1),
                    destination=Slot(COLUMN, 0),
                ),
            )

    def test_moving_a_card_that_is_not_there_is_rejected(self) -> None:
        state = deal((("D5",), (), (), (), (), (), (), ()))

        with self.assertRaises(IllegalMove):
            apply_move(
                state,
                CardMove(
                    cards=("D4",),
                    source=Slot(COLUMN, 0),
                    destination=Slot(COLUMN, 1),
                ),
            )

    def test_clearing_dragons_needs_all_four_exposed(self) -> None:
        state = deal((("RD",), ("RD",), (), (), (), (), (), ()))

        with self.assertRaises(IllegalMove):
            apply_move(state, DragonClear(dragon="RD", destination=Slot(CELL, 0)))

    def test_clearing_dragons_blocks_the_destination_cell(self) -> None:
        state = deal((("RD",), ("RD",), ("RD",), ("RD",), (), (), (), ()))

        cleared = apply_move(state, DragonClear(dragon="RD", destination=Slot(CELL, 0)))

        self.assertEqual(cleared.cells[0], BLOCKED_CELL)
        self.assertEqual(cleared.dragons_done, (True, False, False))
        self.assertEqual(cleared.columns[:4], ((), (), (), ()))


class SearchOutcomeTest(unittest.TestCase):
    def test_a_deadlocked_board_is_reported_as_unsolvable(self) -> None:
        # No empty column, no empty cell, no stackable pair (all one suit) and
        # nothing playable to a foundation, so the search has nowhere to go.
        state = State(
            columns=tuple((f"D{rank}",) for rank in range(2, 10)),
            cells=DRAGONS,
            foundations=(0,) * len(SUITS),
            flower_done=True,
            dragons_done=(False,) * len(DRAGONS),
        )

        with self.assertRaises(DealUnsolvable) as caught:
            solve(state)

        self.assertEqual(caught.exception.explored, 1)

    def test_a_small_budget_is_reported_as_cut_short(self) -> None:
        with self.assertRaises(SearchBudgetExhausted) as caught:
            solve(deal(START_COLUMNS), max_states=10)

        self.assertEqual(caught.exception.explored, 10)
        # The furthest position is carried out so callers can report progress.
        self.assertIsInstance(caught.exception.best_state, State)


if __name__ == "__main__":
    unittest.main()
