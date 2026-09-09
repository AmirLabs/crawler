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
from copy import deepcopy
from datetime import date
from typing import Optional

from selectolax.parser import HTMLParser

from crawler.extraction.structure.hotel_block import (
    _HOTEL_WORDS,
    _clean,
    _looks_like_a_name,
    HotelBlock,
    name_only,
    read_hotel_block,
)
from crawler.extraction.structure.html_rows import RowContainer
from crawler.extraction.models import (
    METHOD_HEURISTIC_HTML,
    ExtractedField,
    ExtractedRow,
    record_attempt,
)
from crawler.extraction.target_fields import ROW_FIELDS, unresolved
from crawler.extraction.readers.dates import (
    DateText,
    find_dates,
    jalali_to_gregorian,
    to_iso,
)
from crawler.extraction.readers.labels import (
    detect_board_type,
    detect_nights,
    detect_room_type,
    detect_star_rating,
)
from crawler.extraction.readers.prices import (
    PriceText,
    find_prices,
    normalize_digits,
    pair_dual_currency,
)

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
_ATTRIBUTE_SCOPE = f"{METHOD_HEURISTIC_HTML}:data_attribute"
_HEADING_ROW_SCOPE = f"{METHOD_HEURISTIC_HTML}:heading_row"

# A Jalali date written as an attribute value: "1405-06-19". Any ``data-*``
# holding one is read, whatever it is named, because the name is the site's
# choice and the shape is not.
_DATA_JALALI = re.compile(r"^\s*(1[34]\d{2})-(\d{1,2})-(\d{1,2})\s*$")

# How far above a row its dates may sit. bisanseir.com writes them on the
# reserve button of the hotel block, three levels above the room row, and
# stops there; walking further would reach the page's own date picker, whose
# dates belong to the other departures.
_ATTRIBUTE_LEVELS = 4

# What the block enclosing the rows can supply, and the field each one fills.
# These are read once per block and shared by its rows, because the rows are
# the room choices for that one hotel.
_BLOCK_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "hotel_name"),
    ("star_rating", "hotel_star_rating"),
    ("board_type", "board_type"),
    ("nights", "nights"),
)
# "2 شب" — a length the card is priced at. Used only to tell one card that
# sells two lengths from one that sells a single stay.
_A_LENGTH = re.compile(r"([\d۰-۹٠-٩]{1,2})[\s‌]*شب")

_ATTEMPTED_FIELDS: tuple[str, ...] = (
    "room_type",
    "hotel_name",
    "hotel_star_rating",
    "board_type",
    "nights",
    "departure_date",
    "return_date",
    "price",
)


# Where a hotel's name ends inside a row: at a number, or at the next thing the
# row says about it. Anything after the name belongs to the offer, not to the
# hotel — its position, the nights on sale, the price, the button.
# The Latin words a name is made of, taken from the end of what comes before
# the lodging word.
_LATIN_WORDS = re.compile(r"([A-Za-z][A-Za-z&'.\-]*(?:\s+[A-Za-z][A-Za-z&'.\-]*){0,3})\s*$")

_AFTER_A_NAME = re.compile(
    r"[\d۰-۹٠-٩]|موقعیت|رزرو|جزئیات|مشاهده|قیمت|شب|ستاره|خدمات"
    r"|تخته|نفره|کودک|نوزاد|اتاق|(?<![آ-ی])(?:دو|یک|سه|تک)(?![آ-ی])"
)


def _hotel_in(text: str) -> Optional[str]:
    """The hotel a row names for itself, where the row is the whole offer.

    Read here and not only from the block above, because on some sites there is
    no block above: nahalgasht sells five Mashhad hotels as five rows, each
    naming its own, and with no name on any of them build_tour grouped all five
    into one offer and kept the cheapest. Four hotels and four prices went out
    of the output without a trace.

    The name runs from the lodging word to the first thing that cannot be part
    of it. What comes back is checked the same way a block's name is, so a row
    that only mentions lodging in passing yields nothing.

    Example:
        >>> _hotel_in("پیشنهاد ویژه2 ستارههتل انقلاب موقعیت مکانی2 شب 8,200,000تومان")
        'هتل انقلاب'
        >>> _hotel_in("6 شب اقامتBBشهر: کوش آداسیهتل MELIKE جزئیات بیشتر")
        'هتل MELIKE'
        >>> _hotel_in("Kemer Star Hotel دو تخته 55,000,000 تومان")
        'Kemer Star Hotel'
        >>> _hotel_in("دو تخته 92,000,000 تومان") is None
        True
    """
    low = text.lower()
    for word in _HOTEL_WORDS:
        at = low.find(word)
        while at >= 0:
            here = text[at: at + len(word)]
            rest = text[at + len(word):]
            end = _AFTER_A_NAME.search(rest)
            name = _clean(here + rest[: end.start() if end else None])
            # A Latin name is written before the word — "Kemer Star Hotel" —
            # and a Persian one after it, so the Latin case reaches backwards.
            if word.isascii():
                before = _LATIN_WORDS.search(text[:at])
                if before:
                    name = _clean(before.group(1) + " " + name)
            if _looks_like_a_name(name) and name.strip().lower() != word:
                return name_only(name)
            at = low.find(word, at + 1)
    return None


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
        >>> row.attempts["hotel_name"]   # looked for, and this row names none
        ['heuristic_html:row_text']
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
    record("hotel_name", _hotel_in(text))

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
    headings = _heading_row(container)

    rows: list[ExtractedRow] = []
    for index, text in enumerate(container.texts()):
        row = map_row_text(text, index=index, source_path=container.path)
        if not row.fields:
            continue
        _apply_hotel_block(row, block)
        if index < len(container.members):
            _apply_attribute_dates(row, container.members[index])
        if "room_type" not in row.fields and index < len(headings):
            row.fields["room_type"] = ExtractedField(
                headings[index], METHOD_HEURISTIC_HTML, _HEADING_ROW_SCOPE
            )
            row.attempts.pop("room_type", None)
            row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)
        rows.extend(_split_by_length(row, text))
    return rows


def _split_by_length(row: ExtractedRow, text: str) -> list[ExtractedRow]:
    """One row per length a card sells, where it sells the same stay at several.

    nahalgasht prices its Mashhad hotels twice on one line — "2 شب 3 شب
    8,200,000تومان 9,050,000تومان" — and reading the first of each threw away
    half of what every one of those pages sells.

    Split only where the shape leaves no room to pair them wrongly: two or more
    lengths, all different, exactly as many prices, and every length printed
    before the first price. Anything else is one row, because a row that sells
    one stay often mentions a number of nights somewhere else in its text.

    Example:
        >>> row = map_row_text("2 شب 3 شب 8,200,000تومان9,050,000تومان")
        >>> [(r.fields["nights"].value, r.fields["price"].value["amount"])
        ...  for r in _split_by_length(row, "2 شب 3 شب 8,200,000تومان9,050,000تومان")]
        [(2, 8200000.0), (3, 9050000.0)]
        >>> len(_split_by_length(map_row_text("دو تخته 92,000,000 تومان"),
        ...                      "دو تخته 92,000,000 تومان"))
        1
    """
    lengths = list(_A_LENGTH.finditer(text))
    prices = find_prices(text)
    if len(lengths) < 2 or len(lengths) != len(prices):
        return [row]

    counts = [int(normalize_digits(match.group(1))) for match in lengths]
    if len(set(counts)) != len(counts):
        return [row]
    if lengths[-1].end() > prices[0].start:
        return [row]

    out: list[ExtractedRow] = []
    for nights, price in zip(counts, prices):
        copy = deepcopy(row)
        copy.fields["nights"] = ExtractedField(
            nights, METHOD_HEURISTIC_HTML, _ROW_TEXT_SCOPE
        )
        copy.fields["price"] = ExtractedField(
            _price_value(price, None), METHOD_HEURISTIC_HTML, _ROW_TEXT_SCOPE
        )
        copy.unresolved_fields = unresolved(copy.fields, ROW_FIELDS)
        out.append(copy)
    return out


def _heading_row(container: RowContainer) -> list[str]:
    """The room labels a block prints once above its prices, one per member.

    A rates table names its columns in a header and its rows say only the
    figures. Written with a table that is obvious; written with a grid it looks
    like any other block, and nahalgasht writes it with a grid — four labels
    and then four unlabelled prices, all children of one element. Read row by
    row those pages carry a price and nothing to file it under, which is why
    single, child and infant all sat at 0% there while the page named them.

    The labels have to be the whole run before the prices and to number exactly
    as many as there are prices. Anything else is a coincidence of layout, and
    pairing on a coincidence would put the child's fare in the single column.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> from crawler.extraction.structure.html_rows import find_row_containers
        >>> tree = HTMLParser(
        ...     '<div class="grid"><div>دو تخته</div><div>یک تخته</div>'
        ...     '<div class="c">92,000,000 تومان</div>'
        ...     '<div class="c">99,000,000 تومان</div></div>'
        ... )
        >>> _heading_row(find_row_containers(tree)[0])
        ['double', 'single']
    """
    if not container.members:
        return []
    parent = container.members[0].parent
    if parent is None:
        return []

    labels: list[str] = []
    for child in parent.iter():
        room = detect_room_type(" ".join((child.text(deep=True) or "").split()))
        if room is None:
            break  # the header has ended; what follows is the prices
        labels.append(room)
    return labels if len(labels) == len(container.members) else []


def _dates_in_attributes(node) -> list[str]:
    """The Jalali dates a row's markup states in ``data-*`` attributes, in order.

    Read from the nearest level that carries any, so a room row picks up its
    own hotel's departure and not the page's date picker. The names are not
    matched, only the values: bisanseir.com calls them ``data-date`` and
    ``data-end-date``, and the next site will call them something else.

    Example:
        >>> tree = HTMLParser('<div class="hotel">'
        ...     '<button data-date="1405-06-19" data-end-date="1405-06-27"></button>'
        ...     '<div class="beds"><div class="bed">دو تخته</div></div></div>')
        >>> _dates_in_attributes(tree.css_first("div.bed"))
        ['1405-06-19', '1405-06-27']
    """
    current = node
    for _ in range(_ATTRIBUTE_LEVELS):
        if current is None:
            break
        found: list[str] = []
        for element in [current, *current.css("*")]:
            for name, value in element.attributes.items():
                if not name.startswith("data-") or not value:
                    continue
                if _DATA_JALALI.match(value) and value.strip() not in found:
                    found.append(value.strip())
        if found:
            return sorted(found)
        current = current.parent
    return []


def _apply_attribute_dates(row: ExtractedRow, node) -> None:
    """Give a row the dates its own markup carries, where its text said none.

    The earlier date is the departure and the later one the return, the same
    reading :func:`_assign_dates` gives a row's text. Nothing is guessed from a
    single date: one date is when the tour leaves, never when it comes back.

    This is why it is worth doing at all. bisanseir.com prints every row's
    departure and return in the markup and neither in the text, so both came
    back empty — ``return_date`` at 0% on five of five pages audited — while
    the answer sat in the same element as the price.

    Example:
        >>> tree = HTMLParser('<div class="hotel">'
        ...     '<button data-date="1405-06-19" data-end-date="1405-06-27"></button>'
        ...     '<div class="bed">دو تخته</div></div>')
        >>> row = map_row_text("دو تخته")
        >>> _apply_attribute_dates(row, tree.css_first("div.bed"))
        >>> row.fields["departure_date"].value["raw"], row.fields["return_date"].value["iso"]
        ('1405-06-19', '2026-09-18')
    """
    if "departure_date" in row.fields and "return_date" in row.fields:
        return
    found = [value for value in _dates_in_attributes(node) if _to_value(value)]
    if not found:
        return
    for name, value in (("departure_date", found[0]),
                        ("return_date", found[1] if len(found) > 1 else None)):
        if value is None or name in row.fields:
            continue
        row.fields[name] = ExtractedField(
            _to_value(value), METHOD_HEURISTIC_HTML, _ATTRIBUTE_SCOPE
        )
        row.attempts.pop(name, None)
    row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)


def _to_value(jalali: str) -> Optional[dict]:
    """One attribute date in the shape a row's date field uses.

    Example:
        >>> _to_value("1405-06-19")
        {'iso': '2026-09-10', 'raw': '1405-06-19', 'time': None}
        >>> _to_value("1405-13-40") is None
        True
    """
    try:
        year, month, day = (int(part) for part in jalali.split("-"))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        iso = date(*jalali_to_gregorian(year, month, day)).isoformat()
    except (ValueError, TypeError):
        return None
    return {"iso": iso, "raw": jalali, "time": None}


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

    The grade is dropped from the name and kept in its own field, so a hotel
    written "هتل آرگ ۴ ستاره" is named "هتل آرگ" here.

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
        >>> [r.fields["hotel_name"].value for r in rows]
        ['هتل آرگ', 'هتل آرگ', 'هتل هما', 'هتل هما']
        >>> [r.fields["hotel_star_rating"].value for r in rows]
        [4, 4, 5, 5]
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
