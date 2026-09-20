"""The departure calendar a page publishes as data instead of as text.

Most sites write their other departures into links, which
:mod:`crawler.extraction.date_links` reads. One shape writes them into a JSON
object embedded in the page, keyed by date, and paints the strip in the
browser. Nothing of it reaches the markup: the date is a key, so it is not in
the text a reader scans and not in the paths :func:`flatten_json` produces
either.

nahalgasht.com is where this was measured. Its ``?date=`` query is answered by
the browser, not the server — the same page comes back byte for byte for every
date — so there is nothing to fetch and the calendar is the only statement of
when the tour leaves. Two of forty of its dateless pages carried one, holding
15 departures between them where the page itself says only "شهریور ماه".

Only the date and what it starts from are read. The calendar also holds a full
price matrix, and reading that would mean deciding that the third number in a
list is the child-without-bed rate — a rule about one site's column order,
which is what this project does not write. The room each column belongs to is
taken from the page's own table instead, and a calendar whose columns do not
line up with it is read for its dates alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from crawler.extraction.readers.dates import dates_in_keys
from crawler.extraction.readers.prices import CURRENCY_WORDS
from crawler.schema.output_models import ADULT_ROOMS, CURRENCY_RIAL, CURRENCY_TOMAN

# What one entry of a calendar calls the two halves of a price. A price can be
# quoted as a foreign amount plus a local one — 235 euro plus 40,000,000 toman.
# Which half is which is settled by the currency each names, never by the order
# they appear in: "first" is the euro on the page this was measured against.
_BEDS = "bed_prices"
_HALVES = (("first", "first_price_currency"), ("second", "second_price_currency"))
_LOCAL_CURRENCIES = frozenset({CURRENCY_RIAL, CURRENCY_TOMAN})


@dataclass(frozen=True)
class Departure:
    """One date a calendar offers, and the cheapest adult price on it.

    Amounts stay in the currency the page quoted them in; converting is step
    4's job. Zero means the calendar named no price for that half, which is a
    real answer here: a tour can be sold entirely in euros.

    Example:
        >>> Departure(jalali="1405-06-30", local=45_000_000.0, local_currency="IRT")
        Departure(jalali='1405-06-30', local=45000000.0, local_currency='IRT', foreign=0.0, foreign_currency=None)
    """

    jalali: str
    local: float = 0.0
    local_currency: Optional[str] = None
    foreign: float = 0.0
    foreign_currency: Optional[str] = None


def package_rooms(rooms: list[str]) -> list[str]:
    """The rooms of one package, off the front of a page's whole row list.

    Every package repeats the same columns, so the order ends where it starts
    over. Taking it from the page rather than naming the columns here is what
    keeps this free of any one site's layout.

    Example:
        >>> package_rooms(["double", "single", "child_no_bed",
        ...                "double", "single", "child_no_bed"])
        ['double', 'single', 'child_no_bed']
        >>> package_rooms([])
        []
    """
    out: list[str] = []
    for room in rooms:
        if room in out:
            break
        out.append(room)
    return out


def _entries(blob: object) -> dict:
    """The date-keyed part of a blob, or nothing where it keys something else."""
    if not isinstance(blob, dict) or not dates_in_keys(blob):
        return {}
    return blob


def _price_of(entry: dict, column: int) -> tuple[float, Optional[str], float, Optional[str]]:
    """One column's two halves, sorted into the local one and the foreign one.

    Example:
        >>> entry = {"first_price_currency": "یورو", "second_price_currency": "تومان",
        ...          "bed_prices": [{"first": 255, "second": 45000000}]}
        >>> _price_of(entry, 0)
        (45000000.0, 'IRT', 255.0, 'EUR')
        >>> _price_of(entry, 9)
        (0.0, None, 0.0, None)
    """
    beds = entry.get(_BEDS) or []
    if column >= len(beds) or not isinstance(beds[column], dict):
        return 0.0, None, 0.0, None
    local = foreign = 0.0
    local_currency = foreign_currency = None
    for amount_key, currency_key in _HALVES:
        amount = beds[column].get(amount_key) or 0
        if not isinstance(amount, (int, float)) or not amount:
            continue
        currency = CURRENCY_WORDS.get(str(entry.get(currency_key) or "").strip())
        if currency in _LOCAL_CURRENCIES:
            local, local_currency = float(amount), currency
        elif currency:
            foreign, foreign_currency = float(amount), currency
    return local, local_currency, foreign, foreign_currency


def read_calendar(
    blobs: list[dict],
    room_order: list[str],
    adult_rooms: frozenset[str] = ADULT_ROOMS,
) -> list[Departure]:
    """Every departure the page's embedded calendar states, cheapest first.

    ``room_order`` is the rooms of one package as the page's own table lists
    them, in that order, so column two can be called a single room without
    anything here knowing which site wrote it. A calendar with more columns
    than the table explains is read for its dates and left without a price:
    the cheapest of four unnamed numbers is the child rate, and sending that
    as what a tour starts from is worse than sending nothing.

    Example:
        >>> blob = {"2026-09-21": [{"first_price_currency": "یورو",
        ...                         "second_price_currency": "تومان",
        ...                         "bed_prices": [{"first": 255, "second": 45000000},
        ...                                        {"first": 430, "second": 45000000}]}]}
        >>> read_calendar([blob], ["double", "single"])
        [Departure(jalali='1405-06-30', local=45000000.0, local_currency='IRT', foreign=255.0, foreign_currency='EUR')]
        >>> read_calendar([blob], [])
        [Departure(jalali='1405-06-30', local=0.0, local_currency=None, foreign=0.0, foreign_currency=None)]
        >>> read_calendar([{"@context": "https://schema.org"}], [])
        []
    """
    wanted = [index for index, room in enumerate(room_order) if room in adult_rooms]
    found: dict[str, Departure] = {}
    for blob in blobs:
        entries = _entries(blob)
        for key, packages in entries.items():
            dates = dates_in_keys({key: None})
            if not dates or not isinstance(packages, list):
                continue
            best: Optional[tuple] = None
            for package in packages:
                if not isinstance(package, dict):
                    continue
                for column in wanted:
                    local, local_currency, foreign, foreign_currency = _price_of(
                        package, column
                    )
                    if not local and not foreign:
                        continue
                    # Ranked the way the listing is: a local price first, then
                    # the amount, so a euro-only date still ranks against itself.
                    rank = (0 if local else 1, local or foreign)
                    if best is None or rank < best[0]:
                        best = (rank, local, local_currency, foreign, foreign_currency)
            found[dates[0]] = (
                Departure(dates[0], *best[1:]) if best else Departure(dates[0])
            )
    return [found[date] for date in sorted(found)]
