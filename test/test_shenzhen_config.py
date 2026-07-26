"""Tests for the shared SHENZHEN I/O domain configuration."""

import unittest

from shenzhen_solitaire.config import (
    card_rank,
    card_suit,
    foundation_index,
    is_number_card,
)


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


if __name__ == "__main__":
    unittest.main()
