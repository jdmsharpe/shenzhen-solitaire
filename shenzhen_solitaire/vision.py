"""Screenshot recognition for the SHENZHEN I/O Solitaire solver.

The recognizer is calibrated from ``shenzhen_reference.png``.  It finds the
green playfield, scales all layout measurements from its width, and compares
the exposed upper-left corner of each card with templates taken from the
reference image.

Width, not the playfield's bounding box: the game sizes its cards from the
window width and lets the felt fill whatever height is left, so only the
horizontal extent is a fixed multiple of card size. That is what makes the
reading independent of both resolution and window shape.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    BLOCKED_CELL,
    DRAGONS,
    DRAGONS_PER_SET,
    FLOWER,
    FREE_CELL_COUNT,
    MAX_NUMBER_RANK,
    SUITS,
    TABLEAU_COLUMN_COUNT,
    board_errors,
    card_rank,
    card_suit,
    foundation_index,
    is_number_card,
    summarize_errors,
)


class ScreenshotRecognitionError(ValueError):
    """Raised when a screenshot cannot be converted into a valid game state."""


@dataclass(frozen=True)
class ExtractedState:
    """State values extracted from one screenshot.

    Mirrors the tuple fields of ``State``, arity included.
    """

    columns: tuple[tuple[str, ...], ...]
    cells: tuple[str | None, ...]
    foundations: tuple[int, ...]
    flower_done: bool
    dragons_done: tuple[bool, ...]


@dataclass(frozen=True)
class _Geometry:
    felt: tuple[int, int, int, int]
    card_width: float
    card_height: float
    tableau_y: int
    card_step: float
    top_y: int
    tableau_lefts: tuple[int, ...]
    flower_left: int


@dataclass(frozen=True)
class _Component:
    x: int
    y: int
    width: int
    height: int
    area: int


@dataclass(frozen=True)
class _Reading:
    label: str
    score: float
    second_label: str
    second_score: float
    bounds: tuple[int, int, int, int]

    @property
    def margin(self) -> float:
        """How much better the best match is than the runner-up."""

        return self.second_score - self.score

    @property
    def confident(self) -> bool:
        """Whether this reading is both card-like and unambiguous.

        The absolute score only rules out crops that are not cards at all;
        every pair of distinct classes scores below the threshold. Telling two
        cards apart is what the margin does.
        """

        return (
            self.score <= _MAX_CLASSIFICATION_SCORE
            and self.margin >= _MIN_CLASSIFICATION_MARGIN
        )


# The reference image contains every card face. These labels describe its
# tableau from bottom to top and are used only to build visual templates.

_REFERENCE_COLUMNS = (
    ("GD", "D5", "B1", "WD", "RD"),
    ("B9", "D6", "C1", "WD"),
    ("B5", "C7", "RD", "C6", "C5"),
    ("GD", "D7", "B2", "B6", "GD"),
    ("RD", "B4", "D8", "B7", "D3"),
    ("D2", "D9", "B8", "WD", "C2"),
    ("C8", "C9", "WD", "D4", "C4"),
    ("GD", "B3", "D1", "C3", "RD"),
)

# Horizontal measurements are fractions of the detected green playfield's
# width, which the game lays its board out from.
_FIRST_COLUMN_X = 0.0380
_COLUMN_PITCH = 0.1187
_CARD_WIDTH = 0.0942
_FLOWER_X = 0.4813

# Vertical measurements are multiples of the card width, *not* fractions of
# the playfield's height. The game sizes its cards from the window width and
# lets the felt grow downward to fill whatever height is left over, so the
# felt's aspect ratio is not fixed: it measures 1.585 on a 16:10 window and
# 1.821 on a 16:9 one, while card width stays within 0.3% of felt width on
# both. Height-relative spans therefore drift with the window's shape rather
# than its resolution. That is what made 16:9 screenshots read as eight empty
# columns: card_step came out 11% short, so every column measured the wrong
# height and no component landed where the layout predicted.
_CARD_HEIGHT = 1.9156
_CARD_STEP = 0.2559
_TOP_ROW_Y = 0.1464
_TABLEAU_Y = 2.3343

# How far a component may sit from where the layout predicts, as a multiple of
# card width. The two rows are 2.19 card widths apart, so this stays well
# clear of matching a tableau column against the top row.
_POSITION_TOLERANCE = 0.30

# The rank-and-suit corner a stacked card leaves exposed, as fractions of card
# width measured from the card's top-left. It is inset from the right edge and
# shorter than _CARD_STEP, so one card's corner never touches its neighbour's.
_CORNER_LEFT = 0.0435
_CORNER_RIGHT = 0.3600
_CORNER_TOP = 0.0190
_CORNER_BOTTOM = 0.2490

# A card corner is sampled into this many pixels regardless of the source
# resolution, so every comparison happens at one fixed size.
_FEATURE_SIZE = (64, 48)

# Below this a corner crop carries too few source pixels to separate ranks
# whose glyphs differ by one stroke. Downscaling the fixture reads perfectly
# at a 65.7 pixel card and wider; from there down to 62.5 it refuses more
# often than it reads, and at 62.5 it misreads a D5 as D8 convincingly enough
# to clear the margin gate. That last case is why the size is checked directly
# rather than left to be caught downstream. A card this wide needs a playfield
# around 700 pixels across, well under any real capture.
#
# The exact figure is a property of the templates, not of the layout, so it
# moves when the calibration image does: it was 64.0 for the reference this one
# replaced. Re-measure it against the fixture when swapping references.
_MINIMUM_CARD_WIDTH = 66.0

# A corner crop is roughly 51x37 source pixels scaled to 64x48, so one feature
# pixel is about 0.8 source pixels. A radius of 2 therefore only tolerated
# +/-1.6 source pixels, which is less than the rounding jitter that
# round(component.y + index * card_step) introduces on a rescaled screenshot.
_SHIFT_RADIUS = 4

# An absolute score this high only rules out crops that are not cards at all:
# every pair of distinct classes scores below it. Telling two cards apart is
# the margin's job, not this threshold's.
_MAX_CLASSIFICATION_SCORE = 0.18

# How far the best match must beat the runner-up. Across every resolution and
# felt aspect ratio the size guard admits, the tightest margin on a correct
# read is 0.0013, near the smallest accepted card; it is 0.0090 at the
# reference scale. This leaves headroom below that rather than tracking it
# closely, because a false rejection costs a usable screenshot. It is a
# diagnostic, not a guarantee: when it fires the report names both candidates
# instead of leaving a misread to surface later as a nonsense deck.
_MIN_CLASSIFICATION_MARGIN = 0.0005

# Debug labels are measured against the card they annotate, never the image.
# The floor is where OpenCV's stroke font stops being readable on a card face;
# the ceiling keeps a 4K capture's labels from dwarfing the cards.
_ANNOTATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MIN_ANNOTATION_SCALE = 0.34
_MAX_ANNOTATION_SCALE = 1.10
_ANNOTATION_FIT_PASSES = 4

# Extensions OpenCV is built to encode. Used to reject a debug path early
# rather than let it fail once recognition has already run.
_DEBUG_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ScreenshotRecognitionError(f"Could not read image: {path}")
    return image


def _find_geometry(image: np.ndarray) -> _Geometry:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array((45, 60, 10), dtype=np.uint8),
        np.array((100, 255, 180), dtype=np.uint8),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(green)
    image_height, image_width = image.shape[:2]
    minimum_area = image_height * image_width * 0.05
    candidates = [
        stat
        for stat in stats[1:]
        if stat[cv2.CC_STAT_AREA] >= minimum_area
        and stat[cv2.CC_STAT_WIDTH] >= image_width * 0.35
        and stat[cv2.CC_STAT_HEIGHT] >= image_height * 0.35
    ]
    if not candidates:
        raise ScreenshotRecognitionError(
            "Could not locate the green Solitaire playfield. "
            "Use the same game theme as the reference image."
        )

    stat = max(candidates, key=lambda item: item[cv2.CC_STAT_AREA])
    x = int(stat[cv2.CC_STAT_LEFT])
    y = int(stat[cv2.CC_STAT_TOP])
    width = int(stat[cv2.CC_STAT_WIDTH])
    height = int(stat[cv2.CC_STAT_HEIGHT])

    card_width = width * _CARD_WIDTH
    if card_width < _MINIMUM_CARD_WIDTH:
        raise ScreenshotRecognitionError(
            f"The playfield is only {width} pixels wide, which leaves cards "
            f"{card_width:.0f} pixels across; recognition needs at least "
            f"{_MINIMUM_CARD_WIDTH:.0f}. Capture the game at a higher resolution."
        )

    lefts = tuple(
        round(x + width * (_FIRST_COLUMN_X + index * _COLUMN_PITCH))
        for index in range(TABLEAU_COLUMN_COUNT)
    )
    return _Geometry(
        felt=(x, y, width, height),
        card_width=card_width,
        card_height=card_width * _CARD_HEIGHT,
        tableau_y=round(y + card_width * _TABLEAU_Y),
        card_step=card_width * _CARD_STEP,
        top_y=round(y + card_width * _TOP_ROW_Y),
        tableau_lefts=lefts,
        flower_left=round(x + width * _FLOWER_X),
    )


def _card_components(image: np.ndarray, geometry: _Geometry) -> list[_Component]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light_cards = cv2.inRange(
        hsv,
        np.array((0, 0, 120), dtype=np.uint8),
        np.array((179, 70, 255), dtype=np.uint8),
    )

    felt_x, felt_y, felt_width, felt_height = geometry.felt
    restricted = np.zeros_like(light_cards)
    restricted[
        felt_y : felt_y + felt_height,
        felt_x : felt_x + felt_width,
    ] = light_cards[
        felt_y : felt_y + felt_height,
        felt_x : felt_x + felt_width,
    ]

    _, _, stats, _ = cv2.connectedComponentsWithStats(restricted)
    minimum_width = geometry.card_width * 0.72
    maximum_width = geometry.card_width * 1.30
    minimum_height = geometry.card_height * 0.72
    minimum_area = geometry.card_width * geometry.card_height * 0.08
    result: list[_Component] = []
    for stat in stats[1:]:
        width = int(stat[cv2.CC_STAT_WIDTH])
        height = int(stat[cv2.CC_STAT_HEIGHT])
        area = int(stat[cv2.CC_STAT_AREA])
        if (
            minimum_width <= width <= maximum_width
            and height >= minimum_height
            and area >= minimum_area
        ):
            result.append(
                _Component(
                    x=int(stat[cv2.CC_STAT_LEFT]),
                    y=int(stat[cv2.CC_STAT_TOP]),
                    width=width,
                    height=height,
                    area=area,
                )
            )
    return result


def _near_component(
    components: list[_Component],
    geometry: _Geometry,
    expected_x: int,
    expected_y: int,
    *,
    top_row: bool,
) -> _Component | None:
    tolerance = geometry.card_width * _POSITION_TOLERANCE
    candidates = [
        component
        for component in components
        if abs(component.x - expected_x) <= tolerance
        and abs(component.y - expected_y) <= tolerance
        # A free cell or foundation shows a single card. Anything appreciably
        # taller is a tableau column, whatever its horizontal position.
        and (not top_row or component.height <= geometry.card_height * 1.35)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: component.area)


def _corner_bounds(
    image: np.ndarray, left: int, top: int, card_width: float
) -> tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]
    x1 = max(0, round(left + card_width * _CORNER_LEFT))
    x2 = min(image_width, round(left + card_width * _CORNER_RIGHT))
    y1 = max(0, round(top + card_width * _CORNER_TOP))
    y2 = min(image_height, round(top + card_width * _CORNER_BOTTOM))
    if x2 <= x1 or y2 <= y1:
        raise ScreenshotRecognitionError("A detected card lies outside the image")
    return x1, y1, x2, y2


def _resample(crop: np.ndarray) -> np.ndarray:
    """Scale a corner crop to the fixed feature size.

    The interpolation has to follow the direction of the resize. INTER_AREA
    averages the source pixels it covers when shrinking, but degenerates to
    nearest-neighbour when asked to grow. Using it in both directions sampled
    the reference, whose 51x37 corners are upscaled here, by a different
    algorithm than a 4K screenshot, whose corners are downscaled. Matching the
    direction roughly tripled the worst-case margin between the best and
    second-best card across a 0.45x to 2.0x scale sweep.
    """

    shrinking = crop.shape[1] >= _FEATURE_SIZE[0] and crop.shape[0] >= _FEATURE_SIZE[1]
    return cv2.resize(
        crop,
        _FEATURE_SIZE,
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )


def _feature(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ScreenshotRecognitionError("Could not crop a detected card corner")
    crop = _resample(crop)
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))

    ink = ((saturation > 45) | (value < 135)).astype(np.float32)
    red = (((hue < 20) | (hue > 170)) & (saturation > 45)).astype(np.float32)
    green = ((hue >= 20) & (hue <= 105) & (saturation > 45)).astype(np.float32)
    black = ((value < 135) & (saturation <= 100)).astype(np.float32)
    return np.stack((ink, red, green, black))


def _shift_feature(feature: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(feature)
    destination_y = slice(max(0, dy), min(feature.shape[1], feature.shape[1] + dy))
    source_y = slice(max(0, -dy), min(feature.shape[1], feature.shape[1] - dy))
    destination_x = slice(max(0, dx), min(feature.shape[2], feature.shape[2] + dx))
    source_x = slice(max(0, -dx), min(feature.shape[2], feature.shape[2] - dx))
    shifted[:, destination_y, destination_x] = feature[:, source_y, source_x]
    return shifted


def _shifted_variants(feature: np.ndarray) -> list[np.ndarray]:
    """Every alignment the matcher will try for one crop.

    Built once per card rather than once per template comparison. The
    candidate is identical across all 31 classes, so constructing its shifted
    copies inside the template loop repeats the same work for every class.
    """

    return [
        _shift_feature(feature, dx, dy)
        for dx in range(-_SHIFT_RADIUS, _SHIFT_RADIUS + 1)
        for dy in range(-_SHIFT_RADIUS, _SHIFT_RADIUS + 1)
    ]


def _aligned_distance(variants: list[np.ndarray], template: np.ndarray) -> float:
    """Distance from ``template`` to whichever alignment fits it best."""

    return min(float(np.mean(np.abs(variant - template))) for variant in variants)


def _build_templates(
    reference: np.ndarray,
) -> dict[str, tuple[np.ndarray, ...]]:
    geometry = _find_geometry(reference)
    components = _card_components(reference, geometry)
    templates: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    for expected_left, labels in zip(
        geometry.tableau_lefts, _REFERENCE_COLUMNS, strict=True
    ):
        component = _near_component(
            components,
            geometry,
            expected_left,
            geometry.tableau_y,
            top_row=False,
        )
        if component is None:
            raise ScreenshotRecognitionError(
                "The calibration image does not match its expected tableau"
            )
        detected_count = (
            round((component.height - geometry.card_height) / geometry.card_step) + 1
        )
        if detected_count != len(labels):
            raise ScreenshotRecognitionError(
                "The calibration image's card spacing could not be detected"
            )

        for index, label in enumerate(labels):
            top = round(component.y + index * geometry.card_step)
            bounds = _corner_bounds(reference, component.x, top, geometry.card_width)
            templates[label].append(_feature(reference, bounds))

    # The flower is the one class the tableau cannot supply, since it leaves for
    # its own slot before the player has a move.
    flower_component = _near_component(
        components,
        geometry,
        geometry.flower_left,
        geometry.top_y,
        top_row=True,
    )
    if flower_component is None:
        raise ScreenshotRecognitionError(
            f"The calibration image is missing the {FLOWER} template"
        )
    bounds = _corner_bounds(
        reference, flower_component.x, flower_component.y, geometry.card_width
    )
    templates[FLOWER].append(_feature(reference, bounds))

    expected_class_count = len(SUITS) * MAX_NUMBER_RANK + len(DRAGONS) + 1
    if len(templates) != expected_class_count:
        raise ScreenshotRecognitionError(
            f"Expected {expected_class_count} visual card classes; "
            f"found {len(templates)}"
        )
    return {label: tuple(items) for label, items in templates.items()}


def _classify(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    templates: dict[str, tuple[np.ndarray, ...]],
) -> _Reading:
    variants = _shifted_variants(_feature(image, bounds))
    scores = sorted(
        (
            min(_aligned_distance(variants, template) for template in examples),
            label,
        )
        for label, examples in templates.items()
    )
    best_score, label = scores[0]
    second_score, second_label = scores[1]
    return _Reading(label, best_score, second_label, second_score, bounds)


def _read_card(
    image: np.ndarray,
    templates: dict[str, tuple[np.ndarray, ...]],
    left: int,
    top: int,
    card_width: float,
) -> _Reading:
    return _classify(image, _corner_bounds(image, left, top, card_width), templates)


def _validate_state(state: ExtractedState) -> None:
    errors = board_errors(
        state.columns,
        state.cells,
        state.foundations,
        state.flower_done,
        state.dragons_done,
    )
    if errors:
        raise ScreenshotRecognitionError(
            f"Detected layout is not a valid deck: {summarize_errors(errors)}"
        )


def _check_debug_path(path: Path) -> None:
    """Reject a debug path that cannot be written, before doing the work.

    ``cv2.imwrite`` picks its encoder from the file extension and *raises*
    rather than returning False when it cannot find one, so an extensionless
    path used to surface as an OpenCV traceback after recognition had already
    finished. Checking first turns that into an immediate, readable message.

    Both checks also run early so that writing the image cannot fail during a
    recognition failure, where raising would replace the diagnosis the debug
    image was asked for in the first place.
    """

    if path.suffix.lower() not in _DEBUG_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(_DEBUG_IMAGE_SUFFIXES))
        raise ScreenshotRecognitionError(
            f"Debug image path needs an image file extension ({supported}); "
            f"got {path.name!r}"
        )
    if not path.parent.is_dir():
        raise ScreenshotRecognitionError(
            f"Debug image directory does not exist: {path.parent}"
        )


def _annotation_variants(reading: _Reading) -> tuple[str, ...]:
    """What one card's label could say, most informative first.

    Only the margin is drawn. The absolute score is near-binary.
    Refusals still quote both scores.
    """

    return (
        f"{reading.label} {reading.margin:.3f}",
        reading.label,
    )


def _annotation_box(
    reading: _Reading, card_width: float, padding: int
) -> tuple[int, int, int, int] | None:
    """The rows beside a card's corner that its label may occupy.

    Bounded by the card's own right edge and by the corner's own rows. That is
    what keeps labels apart: a corner is inset from that edge, and is shorter
    than the sliver a stacked card exposes, so these boxes are disjoint both
    across a row of columns and down a stack. None when the card is too narrow
    to hold anything beside its corner.
    """

    x1, y1, x2, y2 = reading.bounds
    left = x2 + padding
    right = round(x1 - card_width * _CORNER_LEFT + card_width)
    if right - left <= 2 * padding:
        return None
    return left, y1, right, y2


def _fit_annotation(
    reading: _Reading, width: int, height: int
) -> tuple[str, float, int]:
    """Fullest label that stays legible within ``width`` by ``height`` pixels.

    Scaling the text down to fit would make a dense column unreadable, so the
    size stops falling at _MIN_ANNOTATION_SCALE and the content gives way
    instead: a capture with room is annotated with the margin, one without it
    with the card name alone.
    """

    for text in _annotation_variants(reading):
        (unit_width, unit_height), _ = cv2.getTextSize(text, _ANNOTATION_FONT, 1.0, 1)
        scale = min(width / unit_width, height / unit_height, _MAX_ANNOTATION_SCALE)
        if scale < _MIN_ANNOTATION_SCALE:
            continue
        # Thicker strokes widen the glyphs, so the drawn width is measured
        # rather than extrapolated from the unit-scale one, and corrected
        # until it really fits. Without this the widest labels overhang the
        # card by a pixel or two into the column beside them, which is the one
        # thing this layout exists to prevent. Dividing by one more than the
        # measured width keeps every pass a strict shrink, so rounding cannot
        # stall it; it converges in two.
        thickness = max(1, round(scale * 2))
        for _ in range(_ANNOTATION_FIT_PASSES):
            (drawn, _), _ = cv2.getTextSize(text, _ANNOTATION_FONT, scale, thickness)
            if drawn <= width:
                break
            scale *= width / (drawn + 1)
        return text, scale, thickness
    return reading.label, _MIN_ANNOTATION_SCALE, 1


def _write_debug_image(
    image: np.ndarray,
    readings: list[_Reading],
    geometry: _Geometry,
    path: Path,
) -> None:
    """Draw every crop the matcher took, with what it read and how surely.

    Each label is confined to the card its crop came from, in the gap between
    the corner box and the card's right edge, and spans exactly the box's own
    rows. Since the corner is inset from that edge and is shorter than the
    sliver a stacked card exposes, no label can reach the column beside it or
    the card above it. Sizing from the card rather than from the image is what
    keeps that true at any resolution: an image-relative size drew a five-card
    column as five overlapping lines, each about twice the card's width.
    """

    output = image.copy()
    card_width = geometry.card_width
    thickness = max(1, round(card_width / 90))
    padding = max(1, round(card_width * 0.015))

    for reading in readings:
        x1, y1, x2, y2 = reading.bounds
        color = (60, 210, 60) if reading.confident else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        box = _annotation_box(reading, card_width, padding)
        if box is None:
            continue
        left, top, right, bottom = box

        text, scale, text_thickness = _fit_annotation(
            reading, right - left - 2 * padding, bottom - top
        )
        (text_width, text_height), _ = cv2.getTextSize(
            text, _ANNOTATION_FONT, scale, text_thickness
        )
        # Card faces are near-white and the felt is mid-green, so the label
        # carries its own backing rather than relying on either to contrast.
        cv2.rectangle(
            output,
            (left, top),
            (min(right, left + text_width + 2 * padding), bottom),
            (32, 32, 32),
            cv2.FILLED,
        )
        cv2.putText(
            output,
            text,
            (left + padding, (top + bottom + text_height) // 2),
            _ANNOTATION_FONT,
            scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(path), output):
        raise ScreenshotRecognitionError(f"Could not write debug image: {path}")


def _read_board(
    image: np.ndarray,
    templates: dict[str, tuple[np.ndarray, ...]],
    geometry: _Geometry,
    components: list[_Component],
    readings: list[_Reading],
) -> ExtractedState:
    """Read every slot on the board, recording each card in ``readings``.

    ``readings`` is filled in place rather than returned so that a caller
    unwinding from a failure part-way through still holds the cards recognized
    before it, which is what the debug image needs in order to show where the
    read went wrong.
    """

    columns: list[tuple[str, ...]] = []
    for expected_left in geometry.tableau_lefts:
        component = _near_component(
            components,
            geometry,
            expected_left,
            geometry.tableau_y,
            top_row=False,
        )
        if component is None:
            columns.append(())
            continue

        count = (
            round((component.height - geometry.card_height) / geometry.card_step) + 1
        )
        if not 1 <= count <= 13:
            raise ScreenshotRecognitionError(
                f"Detected an implausible {count}-card tableau column"
            )
        labels: list[str] = []
        for index in range(count):
            top = round(component.y + index * geometry.card_step)
            reading = _read_card(
                image, templates, component.x, top, geometry.card_width
            )
            readings.append(reading)
            labels.append(reading.label)
        columns.append(tuple(labels))

    cell_readings: dict[int, _Reading] = {}
    for cell_index, expected_left in enumerate(
        geometry.tableau_lefts[:FREE_CELL_COUNT]
    ):
        component = _near_component(
            components,
            geometry,
            expected_left,
            geometry.top_y,
            top_row=True,
        )
        if component is None:
            continue
        reading = _read_card(
            image, templates, component.x, component.y, geometry.card_width
        )
        readings.append(reading)
        cell_readings[cell_index] = reading

    foundations = [0, 0, 0]
    for expected_left in geometry.tableau_lefts[-len(SUITS) :]:
        component = _near_component(
            components,
            geometry,
            expected_left,
            geometry.top_y,
            top_row=True,
        )
        if component is None:
            continue
        reading = _read_card(
            image, templates, component.x, component.y, geometry.card_width
        )
        readings.append(reading)
        if not is_number_card(reading.label):
            raise ScreenshotRecognitionError(
                f"Foundation was recognized as {reading.label}, not a number card"
            )
        suit_index = foundation_index(reading.label)
        rank = card_rank(reading.label)
        if foundations[suit_index]:
            raise ScreenshotRecognitionError(
                f"Detected two foundations for suit {card_suit(reading.label)}"
            )
        foundations[suit_index] = rank

    flower_component = _near_component(
        components,
        geometry,
        geometry.flower_left,
        geometry.top_y,
        top_row=True,
    )
    flower_done = flower_component is not None
    if flower_component is not None:
        flower_reading = _read_card(
            image,
            templates,
            flower_component.x,
            flower_component.y,
            geometry.card_width,
        )
        readings.append(flower_reading)
        if flower_reading.label != FLOWER:
            raise ScreenshotRecognitionError(
                f"Flower foundation was recognized as {flower_reading.label}"
            )

    cells: list[str | None] = [None] * FREE_CELL_COUNT
    for index, reading in cell_readings.items():
        cells[index] = reading.label

    tableau_counter = Counter(card for column in columns for card in column)
    dragon_done: list[bool] = []
    blocked_without_component = 0
    for dragon in DRAGONS:
        tableau_count = tableau_counter[dragon]
        dragon_cells = [index for index, card in enumerate(cells) if card == dragon]
        visible_count = tableau_count + len(dragon_cells)
        if visible_count == DRAGONS_PER_SET:
            dragon_done.append(False)
        elif tableau_count == 0 and len(dragon_cells) == 1:
            # A cleared set is displayed as a single dragon pile in a free cell.
            cells[dragon_cells[0]] = BLOCKED_CELL
            dragon_done.append(True)
        elif visible_count == 0:
            # Some themes render a cleared set as an empty/covered cell.
            dragon_done.append(True)
            blocked_without_component += 1
        else:
            raise ScreenshotRecognitionError(
                f"Detected {visible_count} {dragon} cards; expected four loose "
                "cards or one cleared pile"
            )

    for index, card in enumerate(cells):
        if blocked_without_component == 0:
            break
        if card is None:
            cells[index] = BLOCKED_CELL
            blocked_without_component -= 1
    if blocked_without_component:
        raise ScreenshotRecognitionError(
            "Cleared dragon sets exceed the available free cells"
        )

    return ExtractedState(
        columns=tuple(columns),
        cells=tuple(cells),
        foundations=tuple(foundations),
        flower_done=flower_done,
        dragons_done=tuple(dragon_done),
    )


def _check_confidence(readings: list[_Reading]) -> None:
    """Refuse a board that was read, but not read convincingly."""

    not_cards = [
        reading for reading in readings if reading.score > _MAX_CLASSIFICATION_SCORE
    ]
    if not_cards:
        worst = max(not_cards, key=lambda reading: reading.score)
        raise ScreenshotRecognitionError(
            f"Low-confidence card match: {worst.label} scored {worst.score:.3f}. "
            "Inspect the debug image or use a closer reference screenshot."
        )

    ambiguous = [reading for reading in readings if not reading.confident]
    if ambiguous:
        worst = min(ambiguous, key=lambda reading: reading.margin)
        raise ScreenshotRecognitionError(
            f"Ambiguous card match: {worst.label} ({worst.score:.4f}) barely beat "
            f"{worst.second_label} ({worst.second_score:.4f}), a margin of "
            f"{worst.margin:.4f}. The crop sits between two cards; inspect the "
            "debug image or use a screenshot closer to the reference scale."
        )


def extract_state(
    screenshot_path: str | Path,
    reference_path: str | Path,
    debug_path: str | Path | None = None,
) -> ExtractedState:
    """Recognize a screenshot and return values suitable for ``State``."""

    screenshot_path = Path(screenshot_path)
    reference_path = Path(reference_path)
    if debug_path is not None:
        _check_debug_path(Path(debug_path))
    image = _load_image(screenshot_path)
    reference = _load_image(reference_path)
    templates = _build_templates(reference)
    geometry = _find_geometry(image)
    components = _card_components(image, geometry)

    readings: list[_Reading] = []
    try:
        state = _read_board(image, templates, geometry, components, readings)
        _check_confidence(readings)
        _validate_state(state)
    finally:
        # A debug image earns its keep precisely when recognition fails, so it
        # is written on the way out rather than only on the success path. The
        # two ways writing it can realistically fail -- an extension OpenCV
        # cannot encode, and a directory that is not there -- are both ruled
        # out by _check_debug_path before any work starts, so raising here
        # cannot replace the diagnosis this image exists to illustrate.
        if debug_path is not None:
            _write_debug_image(image, readings, geometry, Path(debug_path))
    return state
