"""Focused behavioral tests for the SHENZHEN I/O solver."""

import unittest

from shenzhen_solitaire.config import (
    DRAGONS,
    FREE_CELL_COUNT,
    MAX_NUMBER_RANK,
    SUITS,
    TABLEAU_COLUMN_COUNT,
)
from shenzhen_solitaire.solver import State, can_stack, solve


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


if __name__ == "__main__":
    unittest.main()
