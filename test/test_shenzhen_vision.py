"""Integration test for screenshot recognition when OCR dependencies are present."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

OCR_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("cv2", "numpy")
)


class VisionTest(unittest.TestCase):
    @unittest.skipUnless(OCR_AVAILABLE, "OpenCV and NumPy are not installed")
    def test_bundled_screenshot_uses_packaged_reference(self) -> None:
        from shenzhen_solitaire import solver, vision

        test_root = Path(__file__).parent
        arguments = solver._parse_arguments(
            [str(test_root / "fixtures" / "shenzhen_test.png")]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            state = solver._state_from_arguments(arguments)

        self.assertEqual(
            arguments.reference,
            Path(vision.__file__).with_name("shenzhen_reference.png"),
        )

        self.assertEqual(
            state,
            solver.State(
                columns=(
                    ("D5", "C4", "GD", "WD", "RD"),
                    ("GD", "RD", "C6", "C2", "B3"),
                    ("D8", "D1", "GD", "C5", "D7"),
                    ("C1", "B7", "B8", "WD", "GD"),
                    ("B6", "FL", "B9", "D6", "C7"),
                    ("B1", "C3", "B5", "B2", "WD"),
                    ("D4", "D3", "RD", "B4", "D2"),
                    ("WD", "C8", "RD", "C9", "D9"),
                ),
                cells=(None, None, None),
                foundations=(0, 0, 0),
                flower_done=False,
                dragons_done=(False, False, False),
            ),
        )


if __name__ == "__main__":
    unittest.main()
