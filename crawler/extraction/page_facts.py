"""Facts a page states once for the whole tour, not once per row.

:mod:`crawler.extraction.row_mapping` reads a row from its own text, and
:mod:`crawler.extraction.structure.hotel_block` reaches one level up to the
block around it. Neither reaches the top of the page — and that is where some
sites put the two things a row most needs.

tahagasht.com prints its departure in a flight box far below the price table
("پرواز رفت شماره پرواز W5079 ۰۶ شهریور") and its hotels under a heading of
their own ("هتل‌های محل اقامت"), while the price rows say only "هر نفر در اتاق
۲ تخته". nahalgasht.com and liyaseir.com do the same in their summary boxes.
Read row by row those pages have prices and nothing to attach them to.

What is read here is applied only to rows that lack it, and carries its own
source, so a value that belongs to the whole page is never mistaken for one
read from a row. That distinction matters: the page's departure is right for
every row on a page selling one departure, and wrong on a page selling several
— which is why :func:`crawler.extraction.structure.html_rows.offer_blocks`
rejecting date pickers had to come first.

Two things this deliberately refuses to do:

* **Invent a return date.** liyaseir prints "حرکت: 15 شهریور" and then repeats
  the same day under "تاریخ‌های برگزاری". Taking the second date found as the
  return produced a tour that came back the day it left, so a return is only
  read when the page labels one.
* **Take the first label it sees.** tahagasht says "پرواز رفت و برگشت با
  هواپیمایی ماهان" in its service list, hundreds of characters before the real
  departure. Every occurrence is tried, and the first one that actually holds a
  date wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from selectolax.parser import HTMLParser

from crawler.extraction.models import METHOD_HEURISTIC_HTML, ExtractedField
from crawler.extraction.readers.dates import DateText, find_dates, to_iso
from crawler.extraction.target_fields import ROW_FIELDS, unresolved
from crawler.extraction.readers.transport import main_text
from crawler.extraction.structure.hotel_block import (
    _clean,
    _describes_a_hotel,
    _looks_like_a_name,
)

# How a page introduces its own departure. Ordered by how specific the wording
# is, so a page using several is read by its clearest one.
_DEPARTURE_LABELS = ("پرواز رفت", "تاریخ رفت", "رفت :", "رفت:", "حرکت:", "حرکت :")

# How it introduces the way back. Required before a return date is believed.
_RETURN_LABELS = ("پرواز برگشت", "برگشت", "بازگشت")

# How much text after a label may hold its date. Wide enough for tahagasht's
# "پرواز رفت شماره پرواز W5079 ۰۶ شهریور" and the return that follows it,
# narrow enough not to reach the day-by-day itinerary below.
_LABEL_WINDOW = 240

# A hotels section is found by its heading. The names under it are short; a
# long run is the description that follows them.
_HOTEL_HEADING_TAGS = ("h1", "h2", "h3", "h4", "strong", "b")
_MAX_HOTELS = 6

# How a value filled in from the page is marked, so it is never read back as
# something found in the row itself.
_PAGE_SCOPE = f"{METHOD_HEURISTIC_HTML}:page"


@dataclass
class PageFacts:
    """What a page says once, for every offer on it.

    Every field is optional, and absent means the page did not say — never that
    a default should be assumed.

    Example:
        >>> PageFacts(hotel_names=["KEMPINSKI BEIJING", "GRAND HYATT SHANGHAI"]).hotels
        'KEMPINSKI BEIJING + GRAND HYATT SHANGHAI'
        >>> PageFacts().hotels is None
        True
    """

    departure: Optional[DateText] = None
    return_: Optional[DateText] = None
    hotel_names: list[str] = field(default_factory=list)

    @property
    def hotels(self) -> Optional[str]:
        """The hotels as one string, joined the way a stay across cities reads."""
        return " + ".join(self.hotel_names) if self.hotel_names else None


def _first_date_after(text: str, labels: tuple[str, ...]) -> tuple[Optional[DateText], int]:
    """The first date printed after any of ``labels``, and where its window began.

    Every occurrence of every label is tried, not just the first: a site names
    the outbound flight in its service list long before it states when the
    flight leaves.

    Example:
        >>> found, _ = _first_date_after(
        ...     "پرواز رفت و برگشت با ماهان ... پرواز رفت شماره W5079 ۰۶ شهریور", ("پرواز رفت",))
        >>> found.day, found.month
        (6, 6)
    """
    for label in labels:
        start = text.find(label)
        while start >= 0:
            window = text[start: start + _LABEL_WINDOW]
            dates = find_dates(window)
            if dates:
                return dates[0], start
            start = text.find(label, start + 1)
    return None, -1


def read_dates(text: str) -> tuple[Optional[DateText], Optional[DateText]]:
    """The tour's own departure and return, as the page states them.

    A return is read only where the page labels one. Without that rule the
    second date on a liyaseir page — a repeat of the departure under
    "تاریخ‌های برگزاری" — became a return on the day the tour left.

    Example:
        >>> out, back = read_dates("تاریخ رفت 06 شهریور پرواز برگشت 16 شهریور")
        >>> out.day, back.day
        (6, 16)
        >>> out, back = read_dates("حرکت: 15 شهریور تاریخ‌های برگزاری 15 شهریور")
        >>> out.day, back is None
        (15, True)
    """
    departure, at = _first_date_after(text, _DEPARTURE_LABELS)
    if departure is None:
        return None, None

    # The return has to be labelled, and has to come after the departure.
    after = text[at + departure.end:]
    returning, _ = _first_date_after(after, _RETURN_LABELS)
    if returning is not None and to_iso(returning) and to_iso(departure):
        if to_iso(returning) <= to_iso(departure):
            returning = None  # a "return" before the departure is a repeat
    return departure, returning


def _following(node) -> list:
    """The elements after ``node``, and after its parent, in reading order.

    A heading's list is usually its next sibling, but some layouts close the
    heading's wrapper first and put the list beside that wrapper instead.

    Example:
        >>> tree = HTMLParser('<div><h2>t</h2><p>a</p><p>b</p></div>')
        >>> [n.tag for n in _following(tree.css_first("h2"))]
        ['p', 'p']
    """
    out = []
    sibling = node.next
    while sibling is not None:
        if sibling.tag not in ("-text", "-undef"):
            out.append(sibling)
        sibling = sibling.next
    if not out and node.parent is not None:
        sibling = node.parent.next
        while sibling is not None:
            if sibling.tag not in ("-text", "-undef"):
                out.append(sibling)
            sibling = sibling.next
    return out


def read_hotels(html: str) -> list[str]:
    """The hotels a page lists under a heading of their own.

    Found by the heading rather than by position: the section may sit above or
    below the price table, and only its title is reliable across sites.

    Example:
        >>> read_hotels('<body><h2>هتل‌های محل اقامت</h2>'
        ...             '<div>KEMPINSKI HOTEL BEIJING 5*TOP</div>'
        ...             '<div>GRAND HYATT SHANGHAI 5*TOP</div></body>')
        ['KEMPINSKI HOTEL BEIJING 5*TOP', 'GRAND HYATT SHANGHAI 5*TOP']
    """
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript"):
        node.decompose()

    for tag in _HOTEL_HEADING_TAGS:
        for heading in tree.css(tag):
            title = _clean(heading.text(deep=True))
            if not title or not _describes_a_hotel(title) or len(title) > 40:
                continue
            # Only what comes *after* the heading. Reading the whole parent
            # swept in the paragraph above it, which on a real page was the
            # departure sentence and became a hotel name.
            names: list[str] = []
            for sibling in _following(heading):
                for node in [sibling, *sibling.css("*")]:
                    own = _clean(node.text(deep=False))
                    if not own or own == title:
                        continue
                    if _looks_like_a_name(own) and own not in names:
                        names.append(own)
                    if len(names) >= _MAX_HOTELS:
                        break
                if len(names) >= _MAX_HOTELS:
                    break
            if names:
                return names
    return []


def apply_page_facts(rows: list, facts: PageFacts) -> int:
    """Give the page's own facts to the rows that do not state their own.

    A row always wins: a departure printed in the row is about that row, while
    the page's is about the page. What is filled in here is marked
    ``heuristic_html:page`` so a value shared by every row can never be read
    back as one found in a row.

    Returns how many fields were filled, so a caller can report it.

    Example:
        >>> from crawler.extraction.models import ExtractedRow
        >>> from crawler.extraction.readers.dates import find_dates
        >>> rows = [ExtractedRow(), ExtractedRow()]
        >>> facts = PageFacts(departure=find_dates("06 شهریور")[0],
        ...                   hotel_names=["GRAND HYATT SHANGHAI"])
        >>> apply_page_facts(rows, facts)
        4
        >>> rows[0].fields["hotel_name"].value
        'GRAND HYATT SHANGHAI'
        >>> rows[0].fields["departure_date"].value["iso"]
        '2026-08-28'
    """
    filled = 0
    for row in rows:
        for name, value in (
            ("departure_date", _date_value(facts.departure)),
            ("return_date", _date_value(facts.return_)),
            ("hotel_name", facts.hotels),
        ):
            if value is None or name in row.fields:
                continue
            row.fields[name] = ExtractedField(value, METHOD_HEURISTIC_HTML, _PAGE_SCOPE)
            row.attempts.pop(name, None)
            filled += 1
        row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)
    return filled


def _date_value(parsed: Optional[DateText]) -> Optional[dict]:
    """One page-level date in the shape a row's date field uses.

    Example:
        >>> _date_value(find_dates("06 شهریور")[0])
        {'iso': '2026-08-28', 'raw': '06 شهریور', 'time': None}
        >>> _date_value(None) is None
        True
    """
    if parsed is None:
        return None
    return {"iso": to_iso(parsed), "raw": parsed.raw, "time": parsed.time}


def read_page_facts(html: str) -> PageFacts:
    """Everything a page states once, for the rows that state nothing.

    Example:
        >>> facts = read_page_facts(
        ...     '<body><p>تاریخ رفت 06 شهریور پرواز برگشت 16 شهریور</p>'
        ...     '<h2>هتل‌های محل اقامت</h2><div>GRAND HYATT SHANGHAI</div></body>')
        >>> facts.departure.day, facts.return_.day, facts.hotels
        (6, 16, 'GRAND HYATT SHANGHAI')
    """
    departure, returning = read_dates(main_text(html))
    return PageFacts(departure=departure, return_=returning, hotel_names=read_hotels(html))
