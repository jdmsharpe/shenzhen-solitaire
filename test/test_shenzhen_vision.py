"""Integration test for screenshot recognition when OCR dependencies are present."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
import unittest.mock
from pathlib import Path

OCR_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("cv2", "numpy")
)

FIXTURE = Path(__file__).parent / "fixtures" / "shenzhen_test.png"

EXPECTED_COLUMNS = (
    ("D5", "C4", "GD", "WD", "RD"),
    ("GD", "RD", "C6", "C2", "B3"),
    ("D8", "D1", "GD", "C5", "D7"),
    ("C1", "B7", "B8", "WD", "GD"),
    ("B6", "FL", "B9", "D6", "C7"),
    ("B1", "C3", "B5", "B2", "WD"),
    ("D4", "D3", "RD", "B4", "D2"),
    ("WD", "C8", "RD", "C9", "D9"),
)


@unittest.skipUnless(OCR_AVAILABLE, "OpenCV and NumPy are not installed")
class VisionTest(unittest.TestCase):
    def test_bundled_screenshot_uses_packaged_reference(self) -> None:
        from shenzhen_solitaire import solver, vision

        arguments = solver._parse_arguments([str(FIXTURE)])
        with contextlib.redirect_stdout(io.StringIO()):
            state = solver._state_from_arguments(arguments)

        self.assertEqual(
            arguments.reference,
            Path(vision.__file__).with_name("shenzhen_reference.png"),
        )
        # Compared per field: a whole-State assertEqual elides both reprs, so a
        # one-card regression would not say which column moved.
        self.assertEqual(state.columns, EXPECTED_COLUMNS)
        self.assertEqual(state.cells, (None, None, None))
        self.assertEqual(state.foundations, (0, 0, 0))
        self.assertFalse(state.flower_done)
        self.assertEqual(state.dragons_done, (False, False, False))


@unittest.skipUnless(OCR_AVAILABLE, "OpenCV and NumPy are not installed")
class RecognitionRobustnessTest(unittest.TestCase):
    """The fixture is ground truth, so perturbing it is a free oracle."""

    def test_rescaled_screenshots_recognize_identically(self) -> None:
        import cv2

        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        original = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if original is None:
            self.fail(f"Could not read fixture: {FIXTURE}")

        # Spans the smallest card the size guard admits up to a 4K capture.
        # The narrow 0.85-1.1 band this replaced held while recognition was
        # already drifting: a real 0.75-scale screenshot read every card
        # correctly but at a margin of 0.0005, which the gate then refused.
        with tempfile.TemporaryDirectory() as directory:
            for scale in (0.42, 0.55, 0.85, 1.1, 1.6, 2.0):
                with self.subTest(scale=scale):
                    resized = cv2.resize(
                        original,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                    )
                    path = Path(directory) / f"scaled_{scale}.png"
                    cv2.imwrite(str(path), resized)

                    state = vision.extract_state(path, reference)

                    self.assertEqual(state.columns, EXPECTED_COLUMNS)

    def test_a_taller_felt_does_not_move_the_layout(self) -> None:
        """A wider window grows the felt downward without resizing the cards.

        This is the 16:9 regression. Deriving vertical spans from the felt's
        height made card_step shrink as the felt grew, so columns measured
        short, every component missed its expected position, and an entirely
        readable board came back as eight empty columns.
        """

        import cv2
        import numpy as np

        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        original = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if original is None:
            self.fail(f"Could not read fixture: {FIXTURE}")
        _, felt_y, _, felt_height = vision._find_geometry(original).felt

        with tempfile.TemporaryDirectory() as directory:
            for extra_rows in (150, 400, 700):
                with self.subTest(extra_rows=extra_rows):
                    # Repeating a row of bare felt lengthens the playfield
                    # without touching a card, so any change in the reading
                    # comes from the layout maths rather than the pixels.
                    seam = felt_y + felt_height - 3
                    stretched = np.vstack(
                        [
                            original[: seam + 1],
                            np.repeat(original[seam : seam + 1], extra_rows, axis=0),
                            original[seam + 1 :],
                        ]
                    )
                    path = Path(directory) / f"stretched_{extra_rows}.png"
                    cv2.imwrite(str(path), stretched)

                    state = vision.extract_state(path, reference)

                    self.assertEqual(state.columns, EXPECTED_COLUMNS)

    def test_every_fixture_card_clears_the_margin_with_headroom(self) -> None:
        from shenzhen_solitaire import vision

        readings = _fixture_readings()
        tightest = min(reading.margin for reading in readings)

        # extract_state already refuses anything below the threshold, so this
        # asserts headroom rather than correctness: it fails while the margin
        # is merely shrinking, instead of once recognition has already broken.
        self.assertGreater(tightest, 2 * vision._MIN_CLASSIFICATION_MARGIN)

    def test_debug_image_is_written(self) -> None:
        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        with tempfile.TemporaryDirectory() as directory:
            debug_path = Path(directory) / "debug.png"

            vision.extract_state(FIXTURE, reference, debug_path)

            self.assertTrue(debug_path.exists())
            self.assertGreater(debug_path.stat().st_size, 0)

    def test_debug_image_is_written_when_recognition_fails(self) -> None:
        """The failing read is the one worth looking at.

        Refusals raised while the board is still being read used to return
        before the image was written, so the run that most needed a picture
        was the one that produced none.
        """

        from shenzhen_solitaire import vision

        def fail_after_one_card(image, templates, geometry, components, readings):
            readings.append(
                vision._read_card(
                    image,
                    templates,
                    geometry.tableau_lefts[0],
                    geometry.tableau_y,
                    geometry.card_width,
                )
            )
            raise vision.ScreenshotRecognitionError("simulated mid-board refusal")

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        with tempfile.TemporaryDirectory() as directory:
            debug_path = Path(directory) / "debug.png"

            with (
                unittest.mock.patch.object(vision, "_read_board", fail_after_one_card),
                self.assertRaises(vision.ScreenshotRecognitionError),
            ):
                vision.extract_state(FIXTURE, reference, debug_path)

            self.assertTrue(debug_path.exists())
            self.assertGreater(debug_path.stat().st_size, 0)


@unittest.skipUnless(OCR_AVAILABLE, "OpenCV and NumPy are not installed")
class RecognitionErrorTest(unittest.TestCase):
    def test_an_image_without_a_playfield_is_rejected(self) -> None:
        import numpy as np

        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        with tempfile.TemporaryDirectory() as directory:
            import cv2

            path = Path(directory) / "blank.png"
            cv2.imwrite(str(path), np.zeros((600, 900, 3), dtype=np.uint8))

            with self.assertRaises(vision.ScreenshotRecognitionError) as caught:
                vision.extract_state(path, reference)

            self.assertIn("playfield", str(caught.exception))

    def test_a_missing_file_is_rejected(self) -> None:
        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        with self.assertRaises(vision.ScreenshotRecognitionError):
            vision.extract_state(Path("no-such-screenshot.png"), reference)

    def test_a_screenshot_too_small_to_read_is_refused(self) -> None:
        """Shrinking past the guard has to say so, not guess.

        Below it the fixture misreads erratically, and not every misread
        there trips the margin gate, so silence would mean a plausible but
        wrong deck rather than a visible failure.
        """

        import cv2

        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        original = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if original is None:
            self.fail(f"Could not read fixture: {FIXTURE}")

        with tempfile.TemporaryDirectory() as directory:
            tiny = cv2.resize(
                original, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA
            )
            path = Path(directory) / "tiny.png"
            cv2.imwrite(str(path), tiny)

            with self.assertRaises(vision.ScreenshotRecognitionError) as caught:
                vision.extract_state(path, reference)

            self.assertIn("higher resolution", str(caught.exception))

    def test_a_debug_path_without_an_image_extension_is_refused(self) -> None:
        """OpenCV picks its encoder from the extension and raises without one.

        Checked before any work happens, so the report arrives immediately
        and as a recognition error rather than an OpenCV traceback.
        """

        from shenzhen_solitaire import vision

        reference = Path(vision.__file__).with_name("shenzhen_reference.png")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(vision.ScreenshotRecognitionError) as caught:
                vision.extract_state(
                    FIXTURE, reference, Path(directory) / "Screenshot 2026"
                )

            self.assertIn("extension", str(caught.exception))


def _fixture_readings() -> list:
    """Read every tableau card of the bundled fixture, keeping the scores."""

    from shenzhen_solitaire import vision

    reference = Path(vision.__file__).with_name("shenzhen_reference.png")
    templates = vision._build_templates(vision._load_image(reference))
    image = vision._load_image(FIXTURE)
    geometry = vision._find_geometry(image)
    components = vision._card_components(image, geometry)

    readings = []
    for expected_left in geometry.tableau_lefts:
        component = vision._near_component(
            components, geometry, expected_left, geometry.tableau_y, top_row=False
        )
        assert component is not None
        count = (
            round((component.height - geometry.card_height) / geometry.card_step) + 1
        )
        for index in range(count):
            top = round(component.y + index * geometry.card_step)
            readings.append(
                vision._read_card(
                    image, templates, component.x, top, geometry.card_width
                )
            )
    return readings


if __name__ == "__main__":
    unittest.main()
