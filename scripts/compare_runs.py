"""Say what changed between two audit runs, field by field and site by site.

CLAUDE.md asks that a change be verified by running it against the live sites
and reporting the before/after numbers. Doing that by eye over ten score files
is slow and misses the small moves, which are the ones that matter: a field
going from 100% to 96% is a regression hiding inside a run that otherwise looks
better.

Give it two log directories — the one a run wrote and a copy kept from before:

    .venv/bin/python scripts/compare_runs.py logs-before-session logs

Rows are sorted by how much they moved, worst first, because a drop is the
thing worth looking at and a rise explains itself.

**Two runs rarely audit the same pages.** The script picks whichever tour URLs
are live and unseen, so a field can move a long way because one run met tours
priced in dollars and the next did not. A drop here is a question, not a
verdict: check it against a single page read both ways before believing it.
Measured while writing this — bisanseir's foreign prices read 100% in one run
and 34% in the next, and the same page read the same either way.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# "   departure_date        100%   100%   ####" — the field, the share of pages
# it was filled on, and the share of rows.
_SCORE = re.compile(r"^\s{3}(\S+)\s+(\d+)%\s+(\d+)%")


def read_scores(folder: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Every site's per-field fill rate, as ``{site: {field: (pages, rows)}}``.

    Example:
        >>> read_scores(Path("/nowhere"))
        {}
    """
    scores: dict[str, dict[str, tuple[int, int]]] = {}
    if not folder.is_dir():
        return scores
    for site in sorted(folder.iterdir()):
        path = site / "score.txt"
        if not path.is_file():
            continue
        fields: dict[str, tuple[int, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _SCORE.match(line)
            if match:
                fields[match.group(1)] = (int(match.group(2)), int(match.group(3)))
        scores[site.name] = fields
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--min", type=int, default=1,
                        help="ignore moves smaller than this many points")
    args = parser.parse_args()

    before = read_scores(Path(args.before))
    after = read_scores(Path(args.after))
    if not before or not after:
        print("one of the two directories holds no score.txt")
        return 1

    moves: list[tuple[int, str, str, str]] = []
    for site in sorted(set(before) | set(after)):
        was, now = before.get(site, {}), after.get(site, {})
        for field in sorted(set(was) | set(now)):
            old = was.get(field)
            new = now.get(field)
            if old is None:
                moves.append((100, site, field, f"new field, now {new[1]}% of rows"))
            elif new is None:
                moves.append((-100, site, field, f"gone, was {old[1]}% of rows"))
            elif abs(new[1] - old[1]) >= args.min:
                moves.append((new[1] - old[1], site, field,
                              f"{old[1]}% -> {new[1]}% of rows"))

    if not moves:
        print("nothing moved")
        return 0

    moves.sort(key=lambda m: (m[0], m[1], m[2]))
    print(f"{len(moves)} fields moved, worst first\n")
    for delta, site, field, what in moves:
        sign = "-" if delta < 0 else "+"
        print(f"   {sign}{abs(delta):3d}  {site:16} {field:30} {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
