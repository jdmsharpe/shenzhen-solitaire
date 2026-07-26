"""Screenshot recognition for the SHENZHEN I/O Solitaire solver.

The recognizer is calibrated from ``shenzhen_reference.png``.  It finds the
green playfield, scales all layout measurements relative to it, and compares
the exposed upper-left corner of each card with templates taken from the
reference image.
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
    card_rank,
    foundation_index,
    is_number_card,
)


class ScreenshotRecognitionError(ValueError):
    """Raised when a screenshot cannot be converted into a valid game state."""


@dataclass(frozen=True)
class ExtractedState:
    """State values extracted from one screenshot."""

    columns: tuple[tuple[str, ...], ...]
    cells: tuple[str | None, str | None, str | None]
    foundations: tuple[int, int, int]
    flower_done: bool
    dragons_done: tuple[bool, bool, bool]


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
    second_score: float
    bounds: tuple[int, int, int, int]


# The reference image contains every card face. These labels describe its
# tableau from bottom to top and are used only to build visual templates.
_REFERENCE_COLUMNS = (
    ("B5", "B1", "GD", "B4", "WD"),
    ("B6", "WD", "D9", "D7"),
    ("D2", "RD", "C9", "RD", "RD"),
    ("C7", "D3", "C3", "RD", "C6"),
    ("C8", "C2", "D4", "B2", "B9"),
    ("WD", "WD", "D8"),
    ("D6", "GD", "B7", "C4", "C5"),
    ("B8", "GD", "D5", "GD", "B3"),
)

# Measurements are ratios of the detected green playfield. They tolerate
# window movement, screenshot cropping, and uniform resolution changes.
_FIRST_COLUMN_X = 0.0380
_COLUMN_PITCH = 0.1187
_TABLEAU_Y = 0.3522
_CARD_STEP = 0.0389
_CARD_WIDTH = 0.0942
_CARD_HEIGHT = 0.2864
_TOP_ROW_Y = 0.0222
_FLOWER_X = 0.4813
_MAX_CLASSIFICATION_SCORE = 0.18


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
    lefts = tuple(
        round(x + width * (_FIRST_COLUMN_X + index * _COLUMN_PITCH))
        for index in range(TABLEAU_COLUMN_COUNT)
    )
    return _Geometry(
        felt=(x, y, width, height),
        card_width=width * _CARD_WIDTH,
        card_height=height * _CARD_HEIGHT,
        tableau_y=round(y + height * _TABLEAU_Y),
        card_step=height * _CARD_STEP,
        top_y=round(y + height * _TOP_ROW_Y),
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
    _, _, felt_width, felt_height = geometry.felt
    candidates = [
        component
        for component in components
        if abs(component.x - expected_x) <= felt_width * 0.025
        and abs(component.y - expected_y) <= felt_height * 0.030
        and (not top_row or component.height <= felt_height * 0.40)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: component.area)


def _corner_bounds(
    image: np.ndarray, left: int, top: int, card_width: float
) -> tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]
    x1 = max(0, round(left + card_width * 0.0435))
    x2 = min(image_width, round(left + card_width * 0.3600))
    y1 = max(0, round(top + card_width * 0.0190))
    y2 = min(image_height, round(top + card_width * 0.2490))
    if x2 <= x1 or y2 <= y1:
        raise ScreenshotRecognitionError("A detected card lies outside the image")
    return x1, y1, x2, y2


def _feature(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ScreenshotRecognitionError("Could not crop a detected card corner")
    crop = cv2.resize(crop, (64, 48), interpolation=cv2.INTER_AREA)
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


def _feature_distance(first: np.ndarray, second: np.ndarray) -> float:
    return min(
        float(np.mean(np.abs(_shift_feature(first, dx, dy) - second)))
        for dx in range(-2, 3)
        for dy in range(-2, 3)
    )


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

    top_templates = (
        (geometry.flower_left, FLOWER),
        (geometry.tableau_lefts[5], "C1"),
        (geometry.tableau_lefts[6], "D1"),
    )
    for expected_left, label in top_templates:
        component = _near_component(
            components,
            geometry,
            expected_left,
            geometry.top_y,
            top_row=True,
        )
        if component is None:
            raise ScreenshotRecognitionError(
                f"The calibration image is missing the {label} template"
            )
        bounds = _corner_bounds(
            reference, component.x, component.y, geometry.card_width
        )
        templates[label].append(_feature(reference, bounds))

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
    candidate = _feature(image, bounds)
    scores = sorted(
        (
            min(_feature_distance(candidate, template) for template in examples),
            label,
        )
        for label, examples in templates.items()
    )
    best_score, label = scores[0]
    return _Reading(label, best_score, scores[1][0], bounds)


def _read_card(
    image: np.ndarray,
    templates: dict[str, tuple[np.ndarray, ...]],
    left: int,
    top: int,
    card_width: float,
) -> _Reading:
    return _classify(image, _corner_bounds(image, left, top, card_width), templates)


def _validate_state(state: ExtractedState) -> None:
    visible = Counter(card for column in state.columns for card in column)
    visible.update(card for card in state.cells if card not in (None, BLOCKED_CELL))
    errors: list[str] = []

    for suit_index, suit in enumerate(SUITS):
        foundation_rank = state.foundations[suit_index]
        for rank in range(1, MAX_NUMBER_RANK + 1):
            expected = int(rank > foundation_rank)
            actual = visible[f"{suit}{rank}"]
            if actual != expected:
                errors.append(
                    f"{suit}{rank}: expected {expected} visible, detected {actual}"
                )

    for dragon_index, dragon in enumerate(DRAGONS):
        expected = 0 if state.dragons_done[dragon_index] else DRAGONS_PER_SET
        actual = visible[dragon]
        if actual != expected:
            errors.append(f"{dragon}: expected {expected} visible, detected {actual}")

    expected_flowers = 0 if state.flower_done else 1
    if visible[FLOWER] != expected_flowers:
        errors.append(
            f"{FLOWER}: expected {expected_flowers} visible, detected {visible[FLOWER]}"
        )

    blocked_cells = sum(card == BLOCKED_CELL for card in state.cells)
    if blocked_cells != sum(state.dragons_done):
        errors.append(
            f"expected {sum(state.dragons_done)} blocked cells, "
            f"detected {blocked_cells}"
        )

    if errors:
        details = "; ".join(errors[:8])
        if len(errors) > 8:
            details += f"; and {len(errors) - 8} more"
        raise ScreenshotRecognitionError(
            f"Detected layout is not a valid deck: {details}"
        )


def _write_debug_image(image: np.ndarray, readings: list[_Reading], path: Path) -> None:
    output = image.copy()
    scale = max(0.45, image.shape[1] / 2553)
    for reading in readings:
        x1, y1, x2, y2 = reading.bounds
        color = (
            (60, 210, 60) if reading.score <= _MAX_CLASSIFICATION_SCORE else (0, 0, 255)
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, max(1, round(scale * 2)))
        cv2.putText(
            output,
            f"{reading.label} {reading.score:.3f}",
            (x1, max(12, y1 - round(5 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            max(1, round(scale * 2)),
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), output):
        raise ScreenshotRecognitionError(f"Could not write debug image: {path}")


def extract_state(
    screenshot_path: str | Path,
    reference_path: str | Path,
    debug_path: str | Path | None = None,
) -> ExtractedState:
    """Recognize a screenshot and return values suitable for ``State``."""

    screenshot_path = Path(screenshot_path)
    reference_path = Path(reference_path)
    image = _load_image(screenshot_path)
    reference = _load_image(reference_path)
    templates = _build_templates(reference)
    geometry = _find_geometry(image)
    components = _card_components(image, geometry)
    readings: list[_Reading] = []

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
                f"Detected two foundations for suit {reading.label[0]}"
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

    state = ExtractedState(
        columns=tuple(columns),
        cells=tuple(cells),  # type: ignore[arg-type]
        foundations=tuple(foundations),  # type: ignore[arg-type]
        flower_done=flower_done,
        dragons_done=tuple(dragon_done),  # type: ignore[arg-type]
    )

    low_confidence = [
        reading for reading in readings if reading.score > _MAX_CLASSIFICATION_SCORE
    ]
    if debug_path is not None:
        _write_debug_image(image, readings, Path(debug_path))
    if low_confidence:
        worst = max(low_confidence, key=lambda reading: reading.score)
        raise ScreenshotRecognitionError(
            f"Low-confidence card match: {worst.label} scored {worst.score:.3f}. "
            "Inspect the debug image or use a closer reference screenshot."
        )

    _validate_state(state)
    return state
