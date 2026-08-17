"""persistence.py — per-camera cross-clip persistence map.

A static human-like apparatus (mannequin, poster, standee) occupies the *same
screen location in every clip, across all times and lightings*. A real person
does not. This module accumulates, per NVR channel and across sweeps, how often
each screen cell was covered by a FIXTURE-LIKE person track — so a location that
fires in a large fraction of the clips that had any such track can be confirmed
as a fixture.

WHAT THE STORED VALUE MEANS (state version 2)
---------------------------------------------
For a cell ``(r, c)``::

    persistence(r, c) = hits[r,c] / fixture_clips

where a clip increments ``fixture_clips`` if and only if that clip contributed at
least one fixture-like track (``add_clip`` with a non-empty box list), and
``hits[r,c]`` counts such clips whose fixture-like tracks covered that cell.
So the value is the conditional probability

    P(cell hosted a fixture-like track | the clip had a fixture-like track)

The conditioning is the point. State version 1 divided by *every* mined clip and
its numerator accepted *any* briefly-stationary track, which made the statistic
useless in both directions:

  * Denominator inflation — most clips contain no stationary track at all, so
    they could not carry evidence either way, yet they diluted every cell.
  * Numerator dilution — transient standing people scatter across many cells,
    each hit once or twice, so no cell ever concentrated.

Measured on 144 clips/channel of real footage under version 1: the highest cell
value on any of 13 channels was 0.056, against a threshold of 0.6. The signal had
never once fired. Version 2 conditions both numerator and denominator on the same
fixture-like predicate (defined in ``flicker_miner._static_fp_tids``: stationary
AND appearance-static under ZNCC AND at least ``persist_min_track_steps`` steps
long), so a genuine fixture — present at one cell in essentially every clip that
has a fixture at all — converges toward 1.0, while a habitual standing spot stays
near 1/N.

Version 2 also queries with ``max`` over the query box's cells rather than
``mean``. A person-sized query box spans roughly 10 cells at the default 64x36
grid, while a fixture's concentrated footprint may occupy only a few of them, so
averaging diluted a strong local signal by the ratio of the two areas. ``max`` is
the most sensitive statistic available and is the correct choice for a
recall-oriented candidate generator.

KNOWN LIMITATION (deliberate, not an oversight): the map has no time decay. A
fixture that is physically removed keeps its accumulated hits, and its value
falls only asymptotically as ``fixture_clips`` grows. A decayed or sliding-window
estimator would fix this, at the cost of making the stored value no longer a
plain ratio of counts. Since a stale fixture claim costs only a review-time
relabel click, the simpler exact-count semantics are kept.

State is a small JSON per channel::

    <staging>/<nvr>/persistence/ch<NN>.json
    { version, cols, rows, fixture_clips, hits: {"r,c": clip_count} }

A stored state whose ``version`` or grid shape differs from the running config is
moved aside and rebuilt from scratch, because neither the ratio's meaning nor the
cell indexing survives such a change.
"""
import json
import logging
import os

logger = logging.getLogger("mine_dataset.persistence")

STATE_VERSION = 2


class ChannelPersistence:
    def __init__(self, cols, rows, fixture_clips=0, hits=None):
        self.cols = cols
        self.rows = rows
        self.fixture_clips = fixture_clips
        self.hits = hits or {}   # "r,c" -> number of fixture-carrying clips hitting it

    def _cells(self, box, W, H):
        x1, y1, x2, y2 = box
        c0 = max(0, min(self.cols - 1, int(x1 / W * self.cols)))
        c1 = max(0, min(self.cols - 1, int(x2 / W * self.cols)))
        r0 = max(0, min(self.rows - 1, int(y1 / H * self.rows)))
        r1 = max(0, min(self.rows - 1, int(y2 / H * self.rows)))
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def _at(self, r, c):
        if self.fixture_clips == 0:
            return 0.0
        return self.hits.get(f"{r},{c}", 0) / self.fixture_clips

    def is_persistent(self, box, W, H, thresh, min_clips):
        """True if any cell of this box's footprint hosted a fixture-like track in
        at least ``thresh`` of the clips that had one (needs ``min_clips`` of such
        history first). ``max`` over cells, not ``mean`` — see module docstring."""
        if self.fixture_clips < min_clips:
            return False
        cells = self._cells(box, W, H)
        if not cells:
            return False
        return max(self._at(r, c) for (r, c) in cells) >= thresh

    def add_clip(self, boxes, W, H):
        """Fold one clip's FIXTURE-LIKE boxes into the map (each cell counted at
        most once per clip). A clip with no such box carries no evidence and is
        NOT counted in the denominator — that conditioning is what makes the
        stored ratio meaningful. Returns True if the clip was counted."""
        if not boxes:
            return False
        marked = set()
        for box in boxes:
            marked.update(self._cells(box, W, H))
        for (r, c) in marked:
            k = f"{r},{c}"
            self.hits[k] = self.hits.get(k, 0) + 1
        self.fixture_clips += 1
        return True

    def top_fraction(self):
        """Highest cell value currently in the map, for threshold calibration:
        ``persist_thresh`` is only meaningful relative to this. 0.0 if empty."""
        if not self.fixture_clips or not self.hits:
            return 0.0
        return max(self.hits.values()) / self.fixture_clips

    def to_dict(self):
        return {"version": STATE_VERSION, "cols": self.cols, "rows": self.rows,
                "fixture_clips": self.fixture_clips, "hits": self.hits}


def _move_aside(path):
    """Rename an incompatible state file to the first free ``<path>.v<N>.bak``."""
    n = 1
    while os.path.exists(f"{path}.v{n}.bak"):
        n += 1
    backup = f"{path}.v{n}.bak"
    os.replace(path, backup)
    return backup


def load(path, cols, rows):
    """Load a channel's map, or return a fresh one.

    A stored state is discarded (moved aside, not deleted) when its ``version`` or
    its grid shape does not match the running config. Both changes invalidate the
    accumulated counts: a version change redefines what the ratio means, and a
    grid change redefines which pixels a ``"r,c"`` key refers to. Silently reusing
    either would produce numbers that look valid and are not — the previous
    implementation took ``cols``/``rows`` from the file, so editing
    ``persist_grid_cols`` in the config had no observable effect on an existing
    map."""
    if not os.path.exists(path):
        return ChannelPersistence(cols, rows)
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, OSError) as e:
        logger.warning(f"      persistence: unreadable state {path} ({e}); starting fresh")
        return ChannelPersistence(cols, rows)

    version = d.get("version", 1)
    if version != STATE_VERSION or d.get("cols") != cols or d.get("rows") != rows:
        backup = _move_aside(path)
        logger.warning(
            f"      persistence: {os.path.basename(path)} is version {version} "
            f"grid {d.get('cols')}x{d.get('rows')}, config wants version "
            f"{STATE_VERSION} grid {cols}x{rows}; moved to "
            f"{os.path.basename(backup)} and rebuilding")
        return ChannelPersistence(cols, rows)

    return ChannelPersistence(cols, rows, d.get("fixture_clips", 0), d.get("hits", {}))


def save(path, cp):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp.to_dict(), f)
