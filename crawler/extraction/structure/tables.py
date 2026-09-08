"""Reading a rates table, where the header names the fields.

:mod:`crawler.extraction.structure.html_rows` finds repeating siblings and
judges them by shape, because most of these sites build their offers out of
unlabelled ``<div>``s. A ``<table>`` needs none of that guesswork: it says what
each column is.

    نام هتل                | درجه | اتاق دبل    | اتاق سینگل  | کودک با تخت
    Diamond of Bodrum      | 5*   | 169,600,000 | 276,850,000 |  89,150,000
    Yasmin                 | 5*   | 173,520,000 | 284,700,000 |  90,240,000

That is exactly the grain the client asked for — one hotel, one room type, one
price — stated more plainly than anywhere else on these sites. It was being
missed entirely: tahagasht.com serves 30 sampled pages of it and we read zero
rows from all of them, because the block finder requires each member to hold a
*price* and :func:`crawler.extraction.readers.prices.find_prices` needs a
currency word beside the number. In a table the currency is named once, in the
heading or the caption, and never repeated in the cells.

So two things are read differently here:

* **The field a value belongs to comes from its column**, not from the text
  beside it. ``detect_room_type`` reads the header once instead of every cell.
* **A bare number is a price**, because the column said so. The currency comes
  from the header, the caption, or failing those the rest of the page, and a
  table whose currency cannot be established is left alone rather than guessed
  at — a Toman figure read as Rials is out by a factor of ten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from selectolax.parser import HTMLParser, Node

from crawler.extraction.models import (
    METHOD_HEURISTIC_HTML,
    ExtractedField,
    ExtractedRow,
    record_attempt,
)
from crawler.extraction.readers.labels import detect_room_type, detect_star_rating
from crawler.extraction.readers.prices import CURRENCY_WORDS, normalize_digits
from crawler.extraction.target_fields import ROW_FIELDS, unresolved

# What a header cell can name, besides a room type. Matched on the whole header
# text, so "نام هتل" is the name column and "درجه" the grade.
_HOTEL_NAME_WORDS = ("نام هتل", "هتل", "اقامتگاه", "hotel", "name")
_GRADE_WORDS = ("درجه", "ستاره", "grade", "star")
_CITY_WORDS = ("شهر", "مقصد", "city")
_BOARD_WORDS = ("خدمات", "سرویس", "board", "meal")

# A price cell is digits and separators only. Written strictly so a note in the
# same column ("تماس بگیرید") is not read as a number.
_BARE_AMOUNT = re.compile(r"^[\d,٬.\s]+$")

# The smallest amount a real tour price can be. Rates tables also carry columns
# of small numbers — a night count, a grade — and this keeps those out.
_MIN_PRICE = 100_000

# A table needs at least this many priced rows to count as a rates table rather
# than a two-line summary box that happens to use <table>.
_MIN_ROWS = 1

# The currency word on its own, without an amount before it. A rates table
# names its unit once, in a caption or a heading, never in the cells.
_CURRENCY_WORD = re.compile("|".join(CURRENCY_WORDS))

_SOURCE = f"{METHOD_HEURISTIC_HTML}:rates_table"

# A rates table prices a room; it never says when the tour leaves. Saying so is
# the point: without it tahagasht's table pages reported "nothing looked for
# it" on four of five pages, which reads as a rule that was never written when
# the truth is a table that has no such column.
_NOT_IN_A_RATES_TABLE = ("departure_date", "return_date")


@dataclass
class TableColumns:
    """What each column of a rates table holds, read from its header.

    ``prices`` maps a column index to the room type priced in it, which is what
    turns one table row into several offers.

    Example:
        >>> TableColumns(hotel_name=0, grade=1, prices={2: "double", 3: "single"}).is_usable
        True
        >>> TableColumns().is_usable
        False
    """

    hotel_name: Optional[int] = None
    grade: Optional[int] = None
    city: Optional[int] = None
    board: Optional[int] = None
    prices: dict[int, str] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """Whether the header named at least one priced room type."""
        return bool(self.prices)


def read_header(cells: list[str]) -> TableColumns:
    """Work out what each column is from the header row.

    The room type is read by the same reader the rows use elsewhere, so a table
    and a div block agree on what "کودک با تخت" means.

    Example:
        >>> cols = read_header(["نام هتل", "درجه", "اتاق دبل", "اتاق سینگل", "کودک با تخت"])
        >>> cols.hotel_name, cols.grade, cols.prices
        (0, 1, {2: 'double', 3: 'single', 4: 'child_with_bed'})
    """
    columns = TableColumns()
    for index, text in enumerate(cells):
        low = text.strip().lower()
        room = detect_room_type(text)
        if room is not None:
            columns.prices[index] = room
        elif columns.hotel_name is None and any(w in low for w in _HOTEL_NAME_WORDS):
            columns.hotel_name = index
        elif columns.grade is None and any(w in low for w in _GRADE_WORDS):
            columns.grade = index
        elif columns.city is None and any(w in low for w in _CITY_WORDS):
            columns.city = index
        elif columns.board is None and any(w in low for w in _BOARD_WORDS):
            columns.board = index
    return columns


def _amount(text: str) -> Optional[float]:
    """Read a bare figure from a price cell, or ``None`` if it is not one.

    Example:
        >>> _amount("262,230,000")
        262230000.0
        >>> _amount("۱۶۹٬۶۰۰٬۰۰۰")
        169600000.0
        >>> _amount("تماس بگیرید") is None
        True
        >>> _amount("5") is None
        True
    """
    cleaned = normalize_digits(text).strip()
    if not cleaned or not _BARE_AMOUNT.match(cleaned):
        return None
    digits = re.sub(r"[^\d]", "", cleaned)
    if not digits:
        return None
    value = float(digits)
    return value if value >= _MIN_PRICE else None


def table_currency(table: Node, page_text: str) -> Optional[str]:
    """The currency a table's figures are in, from the nearest place that says.

    Looked for in the table's own text first — its caption or header usually
    carries "(تومان)" — and only then in the page around it. Returns ``None``
    when nothing says, and the caller then skips the table: reading Toman as
    Rial is a tenfold error, and an empty column is cheaper than that.

    The currency *word* is what is looked for, not a whole price: a caption
    reads "نرخ‌ها به تومان" with no figure beside it, so
    :func:`~crawler.extraction.readers.prices.find_prices` finds nothing there.

    Example:
        >>> tree = HTMLParser('<table><caption>نرخ‌ها به تومان</caption>'
        ...                   '<tr><td>1</td></tr></table>')
        >>> table_currency(tree.css_first("table"), "")
        'IRT'
        >>> bare = HTMLParser('<table><tr><td>1</td></tr></table>').css_first("table")
        >>> table_currency(bare, "") is None
        True
        >>> table_currency(bare, "شروع قیمت از 92,000,000 تومان")
        'IRT'
    """
    own = " ".join((table.text(deep=True) or "").split())
    for text in (own, page_text):
        if not text:
            continue
        hit = _CURRENCY_WORD.search(text)
        if hit:
            return CURRENCY_WORDS[hit.group(0)]
    return None


def read_rates_table(
    table: Node, page_text: str = "", source_path: Optional[str] = None
) -> list[ExtractedRow]:
    """Turn one rates table into a row per hotel per room type.

    Returns an empty list for any table this cannot read with confidence — a
    header that names no room type, or figures whose currency nothing states.

    Example:
        >>> tree = HTMLParser(
        ...     '<table><tr><th>نام هتل</th><th>درجه</th>'
        ...     '<th>اتاق دبل</th><th>اتاق سینگل</th></tr>'
        ...     '<tr><td>Diamond of Bodrum</td><td>5*</td>'
        ...     '<td>169,600,000</td><td>276,850,000</td></tr></table>'
        ... )
        >>> rows = read_rates_table(tree.css_first("table"), "قیمت 1,000,000 تومان")
        >>> len(rows)
        2
        >>> rows[0].fields["hotel_name"].value, rows[0].fields["room_type"].value
        ('Diamond of Bodrum', 'double')
        >>> rows[0].fields["price"].value["amount"], rows[0].fields["hotel_star_rating"].value
        (169600000.0, 5)
    """
    all_rows = table.css("tr")
    if len(all_rows) < 2:
        return []

    header = [" ".join((c.text(deep=True) or "").split()) for c in all_rows[0].css("th, td")]
    columns = read_header(header)
    if not columns.is_usable:
        return []

    currency = table_currency(table, page_text)
    if currency is None:
        return []

    rows: list[ExtractedRow] = []
    for line in all_rows[1:]:
        cells = [" ".join((c.text(deep=True) or "").split()) for c in line.css("th, td")]
        if len(cells) < 2:
            continue

        def cell(index: Optional[int]) -> Optional[str]:
            if index is None or index >= len(cells):
                return None
            return cells[index] or None

        shared: dict[str, object] = {
            "hotel_name": cell(columns.hotel_name),
            "hotel_city": cell(columns.city),
            "board_type": cell(columns.board),
        }
        grade = cell(columns.grade)
        shared["hotel_star_rating"] = detect_star_rating(grade) if grade else None

        for index, room in columns.prices.items():
            amount = _amount(cell(index) or "")
            if amount is None:
                continue  # this hotel is not sold in this occupancy
            fields: dict[str, ExtractedField] = {
                "room_type": ExtractedField(room, METHOD_HEURISTIC_HTML, _SOURCE),
                "price": ExtractedField(
                    {
                        "amount": amount,
                        "currency": currency,
                        "foreign_amount": None,
                        "foreign_currency": None,
                        "raw": f"{cell(index)} {currency}",
                    },
                    METHOD_HEURISTIC_HTML,
                    _SOURCE,
                ),
            }
            for name, value in shared.items():
                if value not in (None, ""):
                    fields[name] = ExtractedField(value, METHOD_HEURISTIC_HTML, _SOURCE)

            attempts: dict[str, list[str]] = {}
            for name in _NOT_IN_A_RATES_TABLE:
                record_attempt(attempts, name, _SOURCE)

            rows.append(
                ExtractedRow(
                    fields=fields,
                    attempts=attempts,
                    unresolved_fields=unresolved(fields, ROW_FIELDS),
                    source_path=source_path or "table.rates",
                    index=len(rows),
                )
            )

    return rows if len(rows) >= _MIN_ROWS else []


def find_rates_tables(tree: HTMLParser, page_text: str = "") -> list[ExtractedRow]:
    """Read every rates table on a page, in the order they appear.

    A page may carry more than one — a table per departure, or per city — and
    they are read together for the same reason
    :func:`crawler.extraction.structure.html_rows.offer_blocks` keeps every
    block rather than the best one.

    Example:
        >>> tree = HTMLParser(
        ...     '<body><table><tr><th>نام هتل</th><th>اتاق دبل</th></tr>'
        ...     '<tr><td>Yasmin</td><td>173,520,000</td></tr></table>'
        ...     '<table><tr><th>روز</th><th>برنامه</th></tr>'
        ...     '<tr><td>1</td><td>پرواز</td></tr></table></body>'
        ... )
        >>> rows = find_rates_tables(tree, "1,000,000 تومان")
        >>> len(rows), rows[0].fields["hotel_name"].value
        (1, 'Yasmin')
    """
    rows: list[ExtractedRow] = []
    for index, table in enumerate(tree.css("table")):
        found = read_rates_table(table, page_text, source_path=f"table[{index}]")
        for row in found:
            row.index = len(rows)
            rows.append(row)
    return rows
