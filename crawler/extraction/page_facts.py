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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Optional

from selectolax.parser import HTMLParser

from crawler.extraction.models import METHOD_HEURISTIC_HTML, ExtractedField
from crawler.extraction.readers.dates import (
    DateText,
    find_dates,
    gregorian_to_jalali,
    to_iso,
)
from crawler.extraction.readers.labels import detect_nights, detect_star_rating
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

# How a page says the flight is charged separately. Matched as a phrase, since
# "پرواز" alone appears in every itinerary on these sites.
_FLIGHT_EXTRA = re.compile(r"(هزینه|قیمت)\s*پرواز|\+\s*پرواز|بدون\s*(بلیط|پرواز)")

# Where a page states the hotel's grade and the tour's length. Both are read
# from a window after their label rather than from the whole page: the page also
# describes a hotel per city and a night count per city, and the first match
# found was the wrong one.
_STAR_LABELS = ("هتل:", "هتل :", "ستاره هتل")
_NIGHT_LABELS = ("مدت تور", "مدت سفر", "مدت اقامت", "مدت")
_LABEL_VALUE_WINDOW = 40

# A tour page whose length is nowhere labelled almost always says it in the page
# title: "تور بدروم 6 شب تابستان 1405". Read there as a last resort.
_TITLE_MAX_CHARS = 120

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

    # Every departure the page states, in the order printed. liyaseir writes
    # "تاریخ‌های برگزاری 25 تیر، 12 و 29 مرداد، 19 شهریور" and vanrosh
    # "تاریخ رفت 9 , 23 مهرماه | 7 , 21 آبان ماه": one tour, four departures,
    # one price table serving all of them. Reading only the first took the
    # oldest, because sites list them in order, and a brazil page still selling
    # 19 شهریور was thrown away for a 25 تیر that had passed.
    departures: list[DateText] = field(default_factory=list)
    return_: Optional[DateText] = None
    hotel_names: list[str] = field(default_factory=list)
    nights: Optional[int] = None
    star_rating: Optional[int] = None
    # ``False`` when the page says the flight is charged on top, ``None`` when
    # it says nothing. Never ``True`` by inference: silence is not a promise.
    flight_included: Optional[bool] = None

    @property
    def departure(self) -> Optional[DateText]:
        """The first departure, for the callers that want one date."""
        return self.departures[0] if self.departures else None

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


def _departures_after(text: str, at: int, first: DateText) -> list[DateText]:
    """Every departure printed with ``first``, deduplicated, in page order.

    The window stops at the first return label. Without that, alefbatour's
    "پرواز رفت ... ۰۶ شهریور ... پرواز برگشت ... ۱۶ شهریور" sits inside one
    window and the return would be counted as a second departure — which would
    then multiply every row by a date the tour comes home on.

    Example:
        >>> text = "تاریخ‌های برگزاری 25 تیر، 12 و 29 مرداد، 19 شهریور مقصد"
        >>> found = find_dates(text)
        >>> [(d.day, d.month) for d in _departures_after(text, 0, found[0])]
        [(25, 4), (12, 5), (29, 5), (19, 6)]
    """
    window = text[at: at + _LABEL_WINDOW]
    for label in _RETURN_LABELS:
        stop = window.find(label)
        if stop >= 0:
            window = window[:stop]

    seen: set[tuple] = set()
    departures: list[DateText] = []
    for parsed in [first, *find_dates(window)]:
        key = (parsed.day, parsed.month, parsed.year)
        if parsed.day is None or key in seen:
            continue
        seen.add(key)
        departures.append(parsed)
    return departures or [first]


def read_dates(text: str) -> tuple[list[DateText], Optional[DateText]]:
    """The tour's departures and its return, as the page states them.

    A page may state several departures at once — see :class:`PageFacts` — so
    the first half of the answer is a list. A return is read only where the
    page labels one. Without that rule the second date on a liyaseir page — a
    repeat of the departure under "تاریخ‌های برگزاری" — became a return on the
    day the tour left.

    Example:
        >>> out, back = read_dates("تاریخ رفت 06 شهریور پرواز برگشت 16 شهریور")
        >>> [d.day for d in out], back.day
        ([6], 16)
        >>> out, back = read_dates("حرکت: 15 شهریور تاریخ‌های برگزاری 15 شهریور")
        >>> [d.day for d in out], back is None
        ([15], True)
        >>> out, _ = read_dates("حرکت: 25 تیر، 12 و 29 مرداد، 19 شهریور")
        >>> [(d.day, d.month) for d in out]
        [(25, 4), (12, 5), (29, 5), (19, 6)]
    """
    departure, at = _first_date_after(text, _DEPARTURE_LABELS)
    if departure is None:
        return [], None

    departures = _departures_after(text, at, departure)

    # The return has to be labelled, and has to come after the departure.
    after = text[at + departure.end:]
    returning, _ = _first_date_after(after, _RETURN_LABELS)
    if returning is not None and to_iso(returning) and to_iso(departure):
        if to_iso(returning) <= to_iso(departure):
            returning = None  # a "return" before the departure is a repeat
    return departures, returning


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


def title_nights(html: str) -> Optional[int]:
    """How many nights the page title says the tour is, if it says.

    Read only when no label states the length. tahagasht.com prints its length
    in a summary strip whose caption sits *under* the value ("۶ شب و ۷ روز" over
    "تعداد روز و شب"), so no label precedes it; alefbatour and avisatravel never
    label it at all. All three do put it in the title.

    The ``<title>`` is used and not the ``<h1>``, which was measured on 48 pages
    across the seven sites: the title agreed with the labelled value 7 times out
    of 7 and supplied it alone on 17 more pages, while the ``h1`` contradicted
    the title on 5 — a tahagasht page titled "7 شب و 8 روز" has an ``h1`` that
    reads 4, from the first city of the itinerary printed beside it.

    Example:
        >>> title_nights('<html><head><title>تور بدروم 6 شب تابستان 1405</title></head></html>')
        6
        >>> title_nights('<html><head><title>تور آلانیا 8مرداد 1405</title></head></html>') is None
        True
    """
    node = HTMLParser(html).css_first("title")
    if node is None:
        return None
    text = " ".join((node.text() or "").split())
    if not text or len(text) > _TITLE_MAX_CHARS:
        return None
    return detect_nights(text)


def _across_departures(rows: list, facts: PageFacts) -> list:
    """Pair every row with the departure it belongs to, copying where needed.

    A page with four departures and one price table is selling four tours, not
    one — vanrosh and liyaseir both print it that way, and in both the same
    table serves every date, which is what makes multiplying right here and
    wrong where a site prices each date separately.

    A row that states its own departure is paired with ``None`` and never
    copied: it has already said which date it belongs to.

    Returns ``(row, departure)`` pairs rather than rows, because two copies of
    one row are equal to each other and their position is the only thing that
    tells them apart.

    Example:
        >>> from crawler.extraction.models import ExtractedRow
        >>> from crawler.extraction.readers.dates import find_dates
        >>> facts = PageFacts(departures=find_dates("5 و 19 شهریور"))
        >>> [d.day for _, d in _across_departures([ExtractedRow()] * 2, facts)]
        [5, 19, 5, 19]
    """
    if len(facts.departures) < 2:
        return [(row, facts.departure) for row in rows]

    paired: list = []
    for row in rows:
        if "departure_date" in row.fields:
            paired.append((row, None))
            continue
        for when in facts.departures:
            paired.append((deepcopy(row), when))
    return paired


def apply_page_facts(rows: list, facts: PageFacts) -> tuple[list, int]:
    """Give the page's own facts to the rows that do not state their own.

    A row always wins: a departure printed in the row is about that row, while
    the page's is about the page. What is filled in here is marked
    ``heuristic_html:page`` so a value shared by every row can never be read
    back as one found in a row.

    Returns the rows and how many fields were filled. The rows come back
    because a page stating several departures ends with more of them than it
    started with — see :func:`_across_departures`.

    Example:
        >>> from crawler.extraction.models import ExtractedRow
        >>> from crawler.extraction.readers.dates import find_dates
        >>> rows, filled = apply_page_facts(
        ...     [ExtractedRow(), ExtractedRow()],
        ...     PageFacts(departures=find_dates("06 شهریور"),
        ...               hotel_names=["GRAND HYATT SHANGHAI"]))
        >>> filled
        4
        >>> rows[0].fields["hotel_name"].value
        'GRAND HYATT SHANGHAI'
        >>> rows[0].fields["departure_date"].value["iso"]
        '2026-08-28'
    """
    paired = _across_departures(rows, facts)
    rows = [row for row, _ in paired]
    filled = 0
    for row, mine in paired:
        for name, value in (
            ("departure_date", _date_value(mine)),
            ("return_date", _date_value(facts.return_ or _derive_return(mine, facts.nights))),
            ("hotel_name", facts.hotels),
            ("nights", facts.nights),
            ("hotel_star_rating", facts.star_rating),
        ):
            if value is None or name in row.fields:
                continue
            row.fields[name] = ExtractedField(value, METHOD_HEURISTIC_HTML, _PAGE_SCOPE)
            row.attempts.pop(name, None)
            filled += 1

        # Whether the flight is extra belongs *on* the price rather than beside
        # it: a row without a price has nothing for the flag to qualify.
        price = row.fields.get("price")
        if facts.flight_included is not None and price is not None:
            if isinstance(price.value, dict) and "flight_included" not in price.value:
                price.value["flight_included"] = facts.flight_included
                filled += 1
        row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)
    return rows, filled


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


def _derive_return(
    departure: Optional[DateText], nights: Optional[int]
) -> Optional[DateText]:
    """Work out when a tour comes back, when the page gives length instead.

    liyaseir.com states "مدت تور: 12 شب و 13 روز" and "حرکت: 15 شهریور" and
    never prints a return date at all. Departure plus nights is arithmetic, not
    a guess — the one thing a tour of N nights certainly does is end N days
    later.

    The result keeps the derived day in ``raw`` so the output shows plainly that
    the site did not print it.

    Example:
        >>> out = find_dates("15 شهریور")[0]
        >>> to_iso(out), to_iso(_derive_return(out, 12))
        ('2026-09-06', '2026-09-18')
        >>> _derive_return(out, 12).raw
        '15 شهریور + 12 شب'
    """
    if departure is None or not nights:
        return None
    iso = to_iso(departure)
    if not iso:
        return None
    ending = date.fromisoformat(iso) + timedelta(days=nights)
    derived = replace(departure, raw=f"{departure.raw} + {nights} شب", time=None)
    jy, jm, jd = gregorian_to_jalali(ending.year, ending.month, ending.day)
    derived.year, derived.month, derived.day = jy, jm, jd
    return derived


def read_page_facts(html: str) -> PageFacts:
    """Everything a page states once, for the rows that state nothing.

    Example:
        >>> facts = read_page_facts(
        ...     '<body><p>تاریخ رفت 06 شهریور پرواز برگشت 16 شهریور</p>'
        ...     '<h2>هتل‌های محل اقامت</h2><div>GRAND HYATT SHANGHAI</div></body>')
        >>> facts.departure.day, facts.return_.day, facts.hotels
        (6, 16, 'GRAND HYATT SHANGHAI')
        >>> flight = read_page_facts('<body>دو تخته 3790 دلار + هزینه پرواز</body>')
        >>> flight.flight_included
        False
    """
    text = main_text(html)
    departures, returning = read_dates(text)
    nights = _labelled(text, _NIGHT_LABELS, detect_nights, _LABEL_VALUE_WINDOW)
    if nights is None:
        nights = title_nights(html)
    if returning is None:
        returning = _derive_return(departures[0] if departures else None, nights)

    return PageFacts(
        departures=departures,
        return_=returning,
        hotel_names=read_hotels(html),
        nights=nights,
        star_rating=_labelled(text, _STAR_LABELS, detect_star_rating, _LABEL_VALUE_WINDOW),
        flight_included=False if _FLIGHT_EXTRA.search(text) else None,
    )


def _labelled(text: str, labels: tuple[str, ...], reader, window: int) -> Optional[int]:
    """Read a number from a short window after a label, not from the page.

    The page is the wrong scope for these. liyaseir writes "مدت تور: 12 شب و 13
    روز" at the top and "مقصد: 4 شب سئول + 4 شب پکن + 4 شب شانگهای" underneath,
    so reading nights from the whole page found 4 — and, because the return date
    is derived from it, gave a twelve-night tour a four-night one's return.

    Example:
        >>> _labelled("مدت تور: 12 شب و 13 روز مقصد: 4 شب سئول",
        ...           ("مدت تور",), detect_nights, 30)
        12
        >>> _labelled("روز سوم بازدید از یک هتل 4 ستاره",
        ...           ("هتل:",), detect_star_rating, 40) is None
        True
    """
    for label in labels:
        at = text.find(label)
        while at >= 0:
            found = reader(text[at: at + window])
            if found is not None:
                return found
            at = text.find(label, at + 1)
    return None
