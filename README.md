# SHENZHEN I/O Solitaire solver

A command-line solver for SHENZHEN I/O Solitaire. It can solve the bundled
example deal without third-party dependencies or recognize a game directly
from a screenshot using the optional OpenCV-based recognizer.

## Vision at a glance

The recognizer uses this bundled calibration screenshot to learn 31 visual
classes from a known deal. For each card, it extracts a normalized feature from
the rank-and-suit corner instead of comparing the entire card face.

![SHENZHEN I/O calibration deal used to build card templates](shenzhen_solitaire/shenzhen_reference.png)

When reading a screenshot, the recognizer locates the green playfield, scales
the expected layout from its width, and labels each detected card corner. Below
is that same calibration image read back, so every label can be checked against
the card underneath it. Each prediction carries its match-distance score and
the margin by which it beat the runner-up; green boxes cleared both thresholds.
Scoring its own reference puts every score at 0.000, which leaves the margins
on show: even an exact match beats its runner-up by only 0.014 to 0.050.

![OCR debug output with predicted cards and confidence scores](docs/images/shenzhen_debug.png)

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended), or `pip`
- The optional `ocr` dependencies when solving from screenshots

## Installation

Run the dependency-free solver directly from a checkout. The console command
and Python module entry point are equivalent:

```console
uv run shenzhen-solitaire
uv run python -m shenzhen_solitaire
```

Include the optional OCR dependencies when recognizing a screenshot:

```console
uv run --extra ocr shenzhen-solitaire path/to/screenshot.png
```

For the equivalent editable installation with `pip`, create a virtual
environment:

```console
python -m venv .venv
```

Activate it for the current shell:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```console
# Linux, macOS, or WSL
source .venv/bin/activate
```

Then install and run the project:

```console
python -m pip install -e ".[ocr]"
shenzhen-solitaire path/to/screenshot.png
```

Omit `[ocr]` when only the dependency-free solver is needed.

## Solving a screenshot

Capture the full green playfield with all tableau columns and the top row
visible. Window movement, cropping, resolution, and window shape are all
supported: every measurement is taken from the playfield's width, so a 16:9
window and a 16:10 one read the same board identically even though the felt
is a different shape in each. The game theme and card artwork do need to match
the reference image.

Resolution has a floor but no ceiling. Cards must come out at least 64 pixels
wide, which needs a playfield around 800 pixels across; below that the rank
glyphs stop separating reliably, so a smaller capture is reported as too small
rather than guessed at.

```console
uv run --extra ocr shenzhen-solitaire path/to/screenshot.png
```

The command prints the recognized columns first, followed by the manual moves
and any automatic moves triggered by each action. Moves are written in this
notation:

| Token | Meaning |
| --- | --- |
| `D1`–`D9` | red dots |
| `B1`–`B9` | green bamboo |
| `C1`–`C9` | black characters |
| `RD` `GD` `WD` | red, green, and white dragons |
| `FL` | flower |
| `T1`–`T8` | tableau columns, numbered from the left |
| `F1`–`F3` | free cells |

So `C4/B3 T8 -> T3` moves the character 4 and the bamboo 3 it carries from
column 8 onto column 3. Cards are named by suit and slots by location, and no
prefix means both, so a move never needs its grammar parsed to be read.

Limit the search when experimenting with difficult deals:

```console
uv run --extra ocr shenzhen-solitaire screenshot.png --max-states 1000000
```

When no solution comes back, the two possible reasons are reported differently,
because only one of them is worth retrying. A search that runs out of budget
reports how far it got and suggests a larger `--max-states`; a search that
exhausts every reachable position reports the deal as unsolvable and suggests
nothing, since a larger budget would return the same answer.

Memory is worth watching. `--max-states` bounds how many positions are
*expanded*, but every position *discovered* is retained, and that is between
one and roughly eight times as many depending on how the deal branches. At
about 1.45 KB per retained position, a million-state budget can mean anywhere
from 1.5 GB to well over 8 GB. The retry hint estimates from what the run
actually retained rather than from the budget.

## Reference image

The package includes `shenzhen_reference.png`, which is used automatically.
It contains every card face in a known deal so the recognizer can build visual
templates for the current game theme.

Supply a different calibration screenshot with `--reference`:

```console
uv run --extra ocr shenzhen-solitaire screenshot.png \
  --reference path/to/shenzhen_reference.png
```

A replacement reference must show the same known calibration deal represented
by `_REFERENCE_COLUMNS` in `shenzhen_solitaire/vision.py`. If the deal differs,
update that label map to match its tableau from bottom to top.

## Debug images

Use `--debug-image` to save an annotated copy of the screenshot:

```console
uv run --extra ocr shenzhen-solitaire screenshot.png \
  --debug-image deals/screenshot_debug.png
```

Each detected card is labeled beside its corner with the predicted card, its
match score, and the margin by which it beat the runner-up. The margin is the
number that matters: every pair of distinct cards scores below the absolute
threshold, so a low score means "this is a card" rather than "this is the
right card". Green boxes cleared both checks; red boxes are either
unrecognizable or too close to call.

Labels are sized from the card they annotate rather than from the image, so
each one stays inside its own card. On a capture too small to fit all three
values legibly, a label drops the margin, then the score, instead of shrinking
until it is unreadable or running into the column alongside.

The debug image is written even when recognition fails, so the runs most worth
inspecting are the ones it covers. Give the path a real image extension, since
OpenCV picks its encoder from it. `deals/` is intended as a local scratch
directory; its generated screenshots and debug images are ignored by Git.

## Development

Run the tests and code-quality checks with:

```console
uv run --extra ocr python -m unittest discover -s test -v
uv run ruff check .
uv run ruff format --check .
uv run --extra ocr mypy
```

The OCR integration test skips automatically when the optional dependencies
are not installed. `mypy` takes its settings and file list from
`pyproject.toml`, and passes either way; including the extra is what lets it
check `vision.py` against the real OpenCV and NumPy signatures.

Solutions are returned as `Move` objects rather than strings, so a solution can
be replayed and checked rather than only printed:

```python
from shenzhen_solitaire import apply_move, normalize_automatic_moves, solve

solution, explored = solve(state)

# solve() works from the normalized position, so a replay has to start there
# too, and re-normalize after each manual move to absorb the auto-play.
state, _ = normalize_automatic_moves(state)
for move, automatic in solution:
    state = apply_move(state, move)  # raises IllegalMove if the move is not legal
    state, _ = normalize_automatic_moves(state)
    print(move)  # renders as "C4/B3 T8 -> T3"
```

The test suite uses exactly this to verify that every solution it produces is
legal and actually wins, since `apply_move` checks the rules independently of
the code that generates candidate moves.

## License

MIT. See [LICENSE](LICENSE).
