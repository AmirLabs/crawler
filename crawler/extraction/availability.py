"""Whether what a page offers can still be bought.

A crawl that keeps every row it can parse fills the output with departures that
already left. The client's users search for a trip to take, so a date in the
past is not a cheaper option — it is noise that makes the cheapest row wrong.

The obvious way to find these pages is to read the notice the sites print, and
that turns out to be wrong. Measured over 60 saved pages:

* ``تکمیل ظرفیت`` appears thirteen times on one nahalgasht page, once per row.
  It marks a full departure, never the page.
* ``مدت زمان اعتبار نتایج جستجو به اتمام رسید`` on tahagasht is about a search
  session expiring. It says nothing about the tour.
* ``این تور منقضی شده است`` on alefbatour and avisatravel *does* describe the
  page — and those pages still list future departures of the same tour, which
  are perfectly sellable. Skipping them would throw away good data.

So the notice is not the test. The test is the dates themselves: a page is
worth keeping when at least one of its offers can still be booked. That holds
whatever a site chooses to print, and it keeps the live departures on a page
whose headline departure has passed.

Rows with no date at all are kept. bisanseir sells by room type and prints no
departure date, so "no date" means "this page does not say", not "expired".
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from crawler.extraction.models import ExtractedRow

# Notices worth recording when they appear, so a page that yields nothing can
# be told apart from one that failed to parse. These explain a result; they
# never decide it — see the module docstring for why.
CLOSED_NOTICES = (
    "برگزار نمی",      # nahalgasht: the tour is not running
    "منقضی شده",       # alefbatour, avisatravel: this departure has passed
    "لغو شده",
)


def _departure(row: ExtractedRow) -> Optional[date]:
    """The day a row leaves, if it states one plainly.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> row = ExtractedRow(fields={"departure_date": ExtractedField(
        ...     {"iso": "2026-09-14", "raw": "14 شهریور", "time": None},
        ...     METHOD_HEURISTIC_HTML)})
        >>> _departure(row)
        datetime.date(2026, 9, 14)
    """
    field = row.fields.get("departure_date")
    if field is None or not isinstance(field.value, dict):
        return None
    iso = field.value.get("iso")
    if not isinstance(iso, str) or not iso:
        return None
    try:
        return date.fromisoformat(iso)
    except ValueError:
        return None


def is_bookable(row: ExtractedRow, today: Optional[date] = None) -> bool:
    """Whether this row's departure has not already gone.

    A row that names no date is bookable: not every site prints one, and
    treating silence as "expired" would empty bisanseir entirely.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> def row(iso):
        ...     return ExtractedRow(fields={"departure_date": ExtractedField(
        ...         {"iso": iso, "raw": iso, "time": None}, METHOD_HEURISTIC_HTML)})
        >>> is_bookable(row("2026-09-14"), today=date(2026, 8, 20))
        True
        >>> is_bookable(row("2026-08-19"), today=date(2026, 8, 20))
        False
        >>> is_bookable(ExtractedRow(), today=date(2026, 8, 20))
        True
    """
    when = _departure(row)
    if when is None:
        return True
    return when >= (today or date.today())


def split_by_availability(
    rows: list[ExtractedRow], today: Optional[date] = None
) -> tuple[list[ExtractedRow], list[ExtractedRow]]:
    """Separate the rows still on sale from the ones that have gone.

    Both halves are returned rather than one filtered list, so a caller can say
    *why* a page ended up empty: "we read nothing" and "we read four departures
    and all four have left" are different problems.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> def row(iso):
        ...     return ExtractedRow(fields={"departure_date": ExtractedField(
        ...         {"iso": iso, "raw": iso, "time": None}, METHOD_HEURISTIC_HTML)})
        >>> live, gone = split_by_availability(
        ...     [row("2026-08-19"), row("2026-09-14")], today=date(2026, 8, 20))
        >>> len(live), len(gone)
        (1, 1)
    """
    live: list[ExtractedRow] = []
    gone: list[ExtractedRow] = []
    for row in rows:
        (live if is_bookable(row, today) else gone).append(row)
    return live, gone


def find_closed_notice(text: str) -> Optional[str]:
    """The site's own "this is over" notice, when it prints one.

    Recorded on the page for explanation only. It is not what decides whether
    a page is kept, because on two of the target sites it appears above a list
    of departures that are still open.

    Example:
        >>> find_closed_notice("تور 5 روز تفلیس تور برگزار نمی‌شود!")
        'برگزار نمی'
        >>> find_closed_notice("تور کیش تکمیل ظرفیت") is None
        True
    """
    return next((notice for notice in CLOSED_NOTICES if notice in text), None)
