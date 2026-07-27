"""Tests for the shared SHENZHEN I/O domain configuration."""

import unittest

from shenzhen_solitaire.config import (
    BLOCKED_CELL,
    DRAGONS,
    DRAGONS_PER_SET,
    FLOWER,
    MAX_NUMBER_RANK,
    SUITS,
    board_errors,
    card_rank,
    card_suit,
    foundation_index,
    is_number_card,
    summarize_errors,
)

FULL_DECK = (
    [f"{suit}{rank}" for suit in SUITS for rank in range(1, MAX_NUMBER_RANK + 1)]
    + [dragon for dragon in DRAGONS for _ in range(DRAGONS_PER_SET)]
    + [FLOWER]
)
DEALT_COLUMNS = tuple(tuple(FULL_DECK[i * 5 : i * 5 + 5]) for i in range(8))


class CardHelpersTest(unittest.TestCase):
    def test_number_cards_cover_each_suit_and_rank_boundary(self) -> None:
        for card in ("D1", "D9", "B1", "B9", "C1", "C9"):
            with self.subTest(card=card):
                self.assertTrue(is_number_card(card))

    def test_non_number_card_labels_are_rejected(self) -> None:
        for card in (None, "", "D", "D0", "D10", "X1", "RD", "FL", "XX"):
            with self.subTest(card=card):
                self.assertFalse(is_number_card(card))

    def test_number_card_components(self) -> None:
        self.assertEqual(card_suit("B7"), "B")
        self.assertEqual(card_rank("B7"), 7)
        self.assertEqual(foundation_index("D1"), 0)
        self.assertEqual(foundation_index("B1"), 1)
        self.assertEqual(foundation_index("C1"), 2)


class BoardValidationTest(unittest.TestCase):
    """A pure function over the five state fields, so no image is needed."""

    def errors_for(self, columns, cells=(None, None, None), **overrides):
        settings = {
            "foundations": (0, 0, 0),
            "flower_done": False,
            "dragons_done": (False, False, False),
        }
        settings.update(overrides)
        return board_errors(columns, cells, **settings)

    def test_a_freshly_dealt_board_has_no_errors(self) -> None:
        self.assertEqual(self.errors_for(DEALT_COLUMNS), [])

    def test_a_duplicated_card_is_reported(self) -> None:
        # Overwrite D2 with a second D1, which is what a transcription slip
        # looks like: one card twice and its neighbour gone.
        columns = (("D1", "D1") + DEALT_COLUMNS[0][2:],) + DEALT_COLUMNS[1:]

        errors = self.errors_for(columns)

        self.assertIn("D1: expected 1 visible, found 2", errors)
        self.assertIn("D2: expected 1 visible, found 0", errors)

    def test_a_card_on_its_foundation_must_not_also_be_visible(self) -> None:
        # D1..D3 are dealt face up, so claiming the D foundation reached 3
        # means three cards are in two places at once.
        errors = self.errors_for(DEALT_COLUMNS, foundations=(3, 0, 0))

        self.assertEqual(len(errors), 3)
        self.assertTrue(all("expected 0 visible, found 1" in e for e in errors), errors)

    def test_a_missing_card_is_reported(self) -> None:
        # The flower is the last card dealt, so dropping it leaves 39.
        columns = DEALT_COLUMNS[:-1] + (DEALT_COLUMNS[-1][:-1],)

        errors = self.errors_for(columns)

        self.assertEqual(errors, [f"{FLOWER}: expected 1 visible, found 0"])

    def test_blocked_cells_must_match_the_cleared_dragons(self) -> None:
        # Claim red is cleared without removing its four cards or blocking a
        # cell: both the dragon count and the cell count should complain.
        errors = self.errors_for(DEALT_COLUMNS, dragons_done=(True, False, False))

        self.assertTrue(any(error.startswith(DRAGONS[0]) for error in errors), errors)
        self.assertTrue(any("blocked cells" in error for error in errors), errors)

    def test_a_cleared_dragon_set_is_accepted(self) -> None:
        without_red = tuple(
            tuple(card for card in column if card != DRAGONS[0])
            for column in DEALT_COLUMNS
        )

        errors = self.errors_for(
            without_red,
            cells=(BLOCKED_CELL, None, None),
            dragons_done=(True, False, False),
        )

        self.assertEqual(errors, [])

    def test_summarize_caps_the_reported_reasons(self) -> None:
        summary = summarize_errors([f"reason {index}" for index in range(11)], limit=8)

        self.assertIn("reason 7", summary)
        self.assertNotIn("reason 8", summary)
        self.assertTrue(summary.endswith("and 3 more"))


if __name__ == "__main__":
    unittest.main()
