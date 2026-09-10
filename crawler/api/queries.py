"""Answering the two questions the client's routes ask of the published files.

One of these is where the routes are most likely to fail quietly. The two
structures write a departure date differently — ``14050619`` in a package and
``1405-06-19`` in a tour — so a filter that compared them as written would
match nothing, return an empty list, and report success. Every date is brought
to the packed form before it is compared, whichever way it arrived.
"""

from __future__ import annotations

import re
from typing import Optional

from crawler.extraction.readers.prices import normalize_digits

_A_DATE = re.compile(r"^(1[34]\d{2})\D?(\d{1,2})\D?(\d{1,2})$")


def as_packed(text: Optional[str]) -> Optional[str]:
    """One Jalali date in the eight-digit form both files can be compared in.

    Example:
        >>> as_packed("1405-06-19"), as_packed("14050619"), as_packed("1405/6/9")
        ('14050619', '14050619', '14050609')
        >>> as_packed("۱۴۰۵-۰۶-۱۹")
        '14050619'
        >>> as_packed("soon") is None
        True
    """
    if not text:
        return None
    found = _A_DATE.match(normalize_digits(text).strip())
    if not found:
        return None
    year, month, day = (int(part) for part in found.groups())
    return f"{year:04d}{month:02d}{day:02d}"


def packages_for(
    objects: list[dict],
    tour_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """The packages of one tour, narrowed to a window of departure dates.

    Both ends are inclusive, and either may be left out. A package the crawl
    could not date is kept only when no window was asked for: it cannot be
    shown to fall inside one.

    Example:
        >>> rows = [{"tour_code": "1", "start_date": "14050619"},
        ...         {"tour_code": "1", "start_date": "14050625"},
        ...         {"tour_code": "2", "start_date": "14050619"}]
        >>> [r["start_date"] for r in packages_for(rows, "1")]
        ['14050619', '14050625']
        >>> [r["start_date"] for r in packages_for(rows, "1", "1405-06-20")]
        ['14050625']
        >>> packages_for(rows, "9")
        []
    """
    since, until = as_packed(start_date), as_packed(end_date)
    out: list[dict] = []
    for row in objects:
        if row.get("tour_code") != tour_code:
            continue
        when = as_packed(row.get("start_date"))
        if when is None:
            if since is None and until is None:
                out.append(row)
            continue
        if since is not None and when < since:
            continue
        if until is not None and when > until:
            continue
        out.append(row)
    return out


def only_changed(records: list[dict], keys: list) -> list[dict]:
    """The records among these that the last run found new or different.

    The keys come from the run's own memory, so this asks the same question the
    delivery asked, rather than working it out a second way.

    Example:
        >>> rows = [{"api_id": "a"}, {"api_id": "b"}]
        >>> only_changed(rows, ["b"])
        [{'api_id': 'b'}]
        >>> only_changed(rows, [])
        []
    """
    from crawler.automation.memory import record_key

    wanted = set(keys)
    return [row for row in records if record_key(row) in wanted]
