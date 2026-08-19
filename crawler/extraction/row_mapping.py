"""Turning a repeating block of markup into rows with named fields.

:mod:`crawler.extraction.structure.html_rows` finds *where* a page repeats;
:mod:`~crawler.extraction.readers.prices`, :mod:`~crawler.extraction.readers.labels`
and :mod:`~crawler.extraction.readers.dates` read the individual values. This
module is where the three meet: it walks one
:class:`~crawler.extraction.structure.html_rows.RowContainer` and produces one
:class:`~crawler.extraction.models.ExtractedRow` per member, with the fields
named as :mod:`crawler.extraction.target_fields` defines them.

Two readings need a decision rather than a lookup, and both are kept small and
in one place so they can be reviewed:

* **Which date is the departure.** avisatravel.com labels its two dates
  ``رفت`` and ``برگشت``; alefbatour.com prints them bare and relies on order.
  Labels win where they exist, order decides otherwise.
* **Which price the row is about.** A row usually quotes one price, and pages
  that quote a local and a foreign half join them with a ``+``. Rows that
  quote several unrelated prices — nahalgasht.com puts a 2-night and a 3-night
  price on one card — keep only the first, because nothing in the row says
  which of the two the rest of its fields describe.

Fields nothing could read are listed in ``unresolved_fields`` rather than
filled, so :attr:`~crawler.extraction.models.ExtractedPage.fill_rate` reports
what really happened.
"""

from __future__ import annotations

import re
from typing import Optional

from crawler.extraction.structure.hotel_block import HotelBlock, read_hotel_block
from crawler.extraction.structure.html_rows import RowContainer
from crawler.extraction.models import (
    METHOD_HEURISTIC_HTML,
    ExtractedField,
    ExtractedRow,
    record_attempt,
)
from crawler.extraction.target_fields import ROW_FIELDS, unresolved
from crawler.extraction.readers.dates import DateText, find_dates, to_iso
from crawler.extraction.readers.labels import (
    detect_board_type,
    detect_nights,
    detect_room_type,
    detect_star_rating,
)
from crawler.extraction.readers.prices import PriceText, find_prices, pair_dual_currency

# Words that mark the return leg. Checked before the departure words because a
# row labels its second date and leaves the first one implied.
_RETURN_LABELS = ("برگشت", "بازگشت", "return")

# What this module reads, and from where. Recorded against any of these that
# comes back empty, so a later stage can tell a field that was searched for and
# missing from one no rule has ever looked for — ``hotel_name`` is the second
# kind today, and knowing that is what stops the LLM stage being asked to
# repeat work the deterministic rules already do.
_ROW_TEXT_SCOPE = f"{METHOD_HEURISTIC_HTML}:row_text"
_HOTEL_BLOCK_SCOPE = f"{METHOD_HEURISTIC_HTML}:hotel_block"

# What the block enclosing the rows can supply, and the field each one fills.
# These are read once per block and shared by its rows, because the rows are
# the room choices for that one hotel.
_BLOCK_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "hotel_name"),
    ("star_rating", "hotel_star_rating"),
    ("board_type", "board_type"),
    ("nights", "nights"),
)
_ATTEMPTED_FIELDS: tuple[str, ...] = (
    "room_type",
    "hotel_star_rating",
    "board_type",
    "nights",
    "departure_date",
    "return_date",
    "price",
)


def _find_return_marker(text: str) -> Optional[int]:
    """Where the return leg is announced in ``text``, if it is.

    Example:
        >>> _find_return_marker("رفت 07 شهریور برگشت 16 شهریور")
        14
        >>> _find_return_marker("01 شهریور 08 شهریور") is None
        True
    """
    positions = [text.find(label) for label in _RETURN_LABELS]
    found = [p for p in positions if p >= 0]
    return min(found) if found else None


def _assign_dates(
    text: str, dates: list[DateText]
) -> tuple[Optional[DateText], Optional[DateText]]:
    """Decide which of a row's dates is the departure and which the return.

    A row with one date has only a departure: a tour that states a single date
    states when it leaves, never when it comes back.

    Example:
        >>> found = find_dates("رفت 07 شهریور برگشت 16 شهریور")
        >>> out, back = _assign_dates("رفت 07 شهریور برگشت 16 شهریور", found)
        >>> out.day, back.day
        (7, 16)
    """
    if not dates:
        return None, None
    if len(dates) == 1:
        return dates[0], None

    marker = _find_return_marker(text)
    if marker is not None:
        before = [d for d in dates if d.start < marker]
        after = [d for d in dates if d.start >= marker]
        if before and after:
            return before[0], after[0]

    return dates[0], dates[1]


def _price_value(local: PriceText, foreign: Optional[PriceText]) -> dict[str, object]:
    """Flatten one logical price into the shape step 4 builds a ``Price`` from.

    Both halves are kept: dropping the foreign one turns
    ``92,000,000 تومان + 1,090 یورو`` into a materially different offer.

    Example:
        >>> text = "دو تخته 92,000,000 تومان + 1,090 یورو"
        >>> local, foreign = pair_dual_currency(text, find_prices(text))[0]
        >>> _price_value(local, foreign)
        {'amount': 92000000.0, 'currency': 'IRT', 'foreign_amount': 1090.0,
         'foreign_currency': 'EUR', 'raw': '92,000,000 تومان + 1,090 یورو'}
    """
    raw = local.raw if foreign is None else f"{local.raw} + {foreign.raw}"
    return {
        "amount": local.amount,
        "currency": local.currency,
        "foreign_amount": foreign.amount if foreign else None,
        "foreign_currency": foreign.currency if foreign else None,
        "raw": raw,
    }


def _date_value(parsed: DateText) -> dict[str, object]:
    """Flatten one date into the three things :class:`TourDate` keeps.

    The time is carried here rather than dropped: avisatravel.com and
    alefbatour.com print a departure time beside the date, and the schema has a
    place for it. ``iso`` stays empty when the page gave no day, so a partial
    date is passed on as partial instead of being completed by guesswork.

    Example:
        >>> _date_value(find_dates("07 شهریور ساعت : 22:30")[0])
        {'iso': '2026-08-29', 'raw': '07 شهریور ساعت : 22:30', 'time': '22:30'}
    """
    return {"iso": to_iso(parsed), "raw": parsed.raw, "time": parsed.time}


def map_row_text(
    text: str, index: int = 0, source_path: Optional[str] = None
) -> ExtractedRow:
    """Read every field this module knows how to read out of one row's text.

    ``source_path`` and ``index`` are carried through so a wrong value can be
    traced back to the block it came from without re-fetching the page.

    Example:
        >>> row = map_row_text("دو تخته 92,000,000 تومان + 1,090 یورو")
        >>> sorted(row.fields)
        ['price', 'room_type']
        >>> row.fields["room_type"].value
        'double'
        >>> row.attempts["departure_date"]
        ['heuristic_html:row_text']
        >>> "hotel_name" in row.attempts   # nothing looks for it yet
        False
    """
    text = " ".join(text.split())
    fields: dict[str, ExtractedField] = {}

    def record(name: str, value: object, source: Optional[str] = None) -> None:
        if value is None:
            return
        fields[name] = ExtractedField(
            value=value,
            method=METHOD_HEURISTIC_HTML,
            source_key=source or source_path,
        )

    # --- labels ------------------------------------------------------------ #
    record("room_type", detect_room_type(text))
    record("hotel_star_rating", detect_star_rating(text))
    record("board_type", detect_board_type(text))
    record("nights", detect_nights(text))

    # --- dates ------------------------------------------------------------- #
    departure, return_ = _assign_dates(text, find_dates(text))
    if departure is not None:
        record("departure_date", _date_value(departure), departure.raw)
    if return_ is not None:
        record("return_date", _date_value(return_), return_.raw)

    # --- price ------------------------------------------------------------- #
    pairs = pair_dual_currency(text, find_prices(text))
    if pairs:
        local, foreign = pairs[0]
        record("price", _price_value(local, foreign), local.raw)

    attempts: dict[str, list[str]] = {}
    for name in _ATTEMPTED_FIELDS:
        if name not in fields:
            record_attempt(attempts, name, _ROW_TEXT_SCOPE)

    return ExtractedRow(
        fields=fields,
        unresolved_fields=unresolved(fields, ROW_FIELDS),
        attempts=attempts,
        source_path=source_path,
        index=index,
    )


def _apply_hotel_block(row: ExtractedRow, block: HotelBlock) -> None:
    """Add the block's facts to a row, without overriding what the row said.

    The row is the more specific source: if a row states its own board type,
    that beats the one printed above the table. Whatever the block supplies is
    marked with its own scope, so a value shared by every row is never mistaken
    for one read from the row itself.

    Example:
        >>> row = map_row_text("دو تخته 82,000,000 تومان")
        >>> _apply_hotel_block(row, HotelBlock(name="The Train Resort", star_rating=3))
        >>> row.fields["hotel_name"].value, row.fields["hotel_name"].source_key
        ('The Train Resort', 'heuristic_html:hotel_block')
    """
    for attribute, field_name in _BLOCK_FIELDS:
        value = getattr(block, attribute)
        if value is None or field_name in row.fields:
            continue
        row.fields[field_name] = ExtractedField(
            value=value,
            method=METHOD_HEURISTIC_HTML,
            source_key=_HOTEL_BLOCK_SCOPE,
        )
        row.attempts.pop(field_name, None)

    for _, field_name in _BLOCK_FIELDS:
        if field_name not in row.fields:
            record_attempt(row.attempts, field_name, _HOTEL_BLOCK_SCOPE)

    row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)


def map_rows(container: RowContainer) -> list[ExtractedRow]:
    """Read every member of ``container`` into a row.

    Facts printed above the rows — the hotel's name, rating and board — are
    read once from the enclosing block and applied to all of them, since that
    is what the page means by putting them there.

    Members that yield no field at all are dropped: an empty row would count
    against the page's fill rate while saying nothing about it.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> from crawler.extraction.structure.html_rows import find_row_containers
        >>> tree = HTMLParser(
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div>'
        ... )
        >>> rows = map_rows(find_row_containers(tree)[0])
        >>> [r.fields["room_type"].value for r in rows]
        ['double', 'single']
    """
    block = read_hotel_block(container)

    rows: list[ExtractedRow] = []
    for index, text in enumerate(container.texts()):
        row = map_row_text(text, index=index, source_path=container.path)
        if not row.fields:
            continue
        _apply_hotel_block(row, block)
        rows.append(row)
    return rows


def _fingerprint(row: ExtractedRow) -> tuple[tuple[str, str], ...]:
    """What makes two rows the same offer, ignoring where they were found.

    Example:
        >>> _fingerprint(map_row_text("دو تخته 92,000,000 تومان"))
        (('price', "{'amount': 92000000.0, 'currency': 'IRT', 'foreign_amount': None,
          'foreign_currency': None, 'raw': '92,000,000 تومان'}"), ('room_type', 'double'))
    """
    return tuple(sorted((name, str(f.value)) for name, f in row.fields.items()))


def map_blocks(containers: list[RowContainer]) -> list[ExtractedRow]:
    """Read every block of a page into one list of rows.

    Each block is read on its own so that its enclosing hotel — the name, the
    rating, the board type printed above its rows — reaches the rows it belongs
    to. Reading them together would give all sixty-four of bisanseir's rows the
    first hotel's name.

    Rows repeated between blocks are kept once. alefbatour.com prints its
    departure list twice, in the page and again inside a booking dialog, and the
    second copy is shorter, so only comparing the rows themselves catches it.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> from crawler.extraction.structure.html_rows import (
        ...     find_row_containers, offer_blocks)
        >>> tree = HTMLParser(
        ...     '<div class="hotel"><h3>هتل آرگ ۴ ستاره</h3><div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div></div></div>'
        ...     '<div class="hotel"><h3>هتل هما ۵ ستاره</h3><div class="beds">'
        ...     '<div class="bed">دو تخته 82,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 89,000,000 تومان</div></div></div>'
        ... )
        >>> rows = map_blocks(offer_blocks(find_row_containers(tree)))
        >>> [(r.index, r.fields["hotel_name"].value) for r in rows]
        [(0, 'هتل آرگ ۴ ستاره'), (1, 'هتل آرگ ۴ ستاره'),
         (2, 'هتل هما ۵ ستاره'), (3, 'هتل هما ۵ ستاره')]
    """
    rows: list[ExtractedRow] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for container in containers:
        for row in map_rows(container):
            fingerprint = _fingerprint(row)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            # Numbered across the page rather than within its block, so an index
            # points at one row of the output and not at one of seventeen.
            row.index = len(rows)
            rows.append(row)

    return rows
