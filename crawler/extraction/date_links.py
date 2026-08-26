"""Finding the links a page uses to offer its other departure dates.

A tour page shows one date's prices and links to the rest, and the prices are
genuinely different: on one bisanseir page the default view quotes 71,000,000
for a room the site sells at 67,000,000 on 11 شهریور — the date it badges as
its cheapest. Copying one price set across every date would produce nine times
the rows with eight ninths of them wrong, so each date has to be fetched.

Every target site declares the other dates in plain ``<a href>``, which is what
makes one shape-based rule enough. Measured on the live sites:

    bisanseir     9 dates   ?date=1405-06-11        the same page, asked again
    alefbatour    4 dates   /tour/d/173164/<slug>   a sibling page per date
    avisatravel   4 dates   /tour/d/97184/<slug>    the same

No JavaScript, no browser, no hidden API. The two shapes need telling apart
because of what they include: the query list holds *this* page's date as well,
while the sibling list is every date except this one.

What is deliberately not matched is nahalgasht's ``/tour/59262-istanbul-divan-
hotel/``. Those URLs differ by an id embedded in the slug, and the ten pages
sharing a stem there are ten different tours, not ten dates of one — the same
trap :mod:`crawler.filtering.tour_groups` documents. A date link has to differ
in a whole path segment that is nothing but digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qsl, unquote, urljoin, urlparse

from selectolax.parser import HTMLParser

# A Jalali date as a site writes it in a query value: "1405-06-11".
_QUERY_DATE = re.compile(r"^\s*(1[34]\d{2})-(\d{1,2})-(\d{1,2})\s*$")

# An id standing alone in a path segment. Four digits is the same floor
# tour_groups uses: below that the number is part of a title.
_ID_SEGMENT = re.compile(r"^\d{4,}$")


@dataclass(frozen=True)
class DateLink:
    """One departure the page offers, and the URL that asks for it.

    ``date`` is filled only where the link states it, which is the query shape.
    A sibling id says nothing about when it leaves, so there it stays ``None``
    and the date comes from the page once fetched.

    ``replaces_this_page`` marks the query shape, whose list includes the date
    already on screen. Keeping both would record that date twice.

    Example:
        >>> DateLink(url="https://x.com/t?date=1405-06-11", date="1405-06-11",
        ...          replaces_this_page=True)
        DateLink(url='https://x.com/t?date=1405-06-11', date='1405-06-11', replaces_this_page=True)
    """

    url: str
    date: Optional[str] = None
    replaces_this_page: bool = False


def _segments(url: str) -> list[str]:
    """The decoded path segments of a URL.

    Example:
        >>> _segments("https://www.alefbatour.com/tour/d/173167/malaysia")
        ['tour', 'd', '173167', 'malaysia']
    """
    return [s for s in unquote(urlparse(url).path).split("/") if s]


def _is_sibling(page_url: str, candidate: str) -> bool:
    """Whether ``candidate`` is the same tour under a different id segment.

    The two URLs must run side by side and part company in exactly one segment,
    and that segment must be all digits in both. A slug that merely contains a
    number — nahalgasht's ``59262-istanbul-divan-hotel`` — is a different tour.

    Example:
        >>> base = "https://www.alefbatour.com/tour/d/173167/malaysia"
        >>> _is_sibling(base, "https://www.alefbatour.com/tour/d/173164/malaysia")
        True
        >>> _is_sibling(base, "https://www.alefbatour.com/tour/d/173164/dubai")
        False
        >>> _is_sibling("https://nahalgasht.com/tour/59262-istanbul-divan-hotel/",
        ...             "https://nahalgasht.com/tour/59263-istanbul-park-hotel/")
        False
    """
    if urlparse(page_url).netloc != urlparse(candidate).netloc:
        return False
    here, there = _segments(page_url), _segments(candidate)
    if len(here) != len(there):
        return False
    differ = [i for i, (a, b) in enumerate(zip(here, there)) if a != b]
    if len(differ) != 1:
        return False
    at = differ[0]
    return bool(_ID_SEGMENT.match(here[at]) and _ID_SEGMENT.match(there[at]))


def _query_date(page_url: str, candidate: str) -> Optional[str]:
    """The Jalali date ``candidate`` asks this same page for, if it asks one.

    Example:
        >>> base = "https://bisanseir.com/tour/tbilisi"
        >>> _query_date(base, "https://bisanseir.com/tour/tbilisi?date=1405-06-11")
        '1405-06-11'
        >>> _query_date(base, "https://bisanseir.com/tour/tbilisi?sort=cheap") is None
        True
    """
    if urlparse(page_url).netloc != urlparse(candidate).netloc:
        return None
    if _segments(page_url) != _segments(candidate):
        return None
    for _, value in parse_qsl(urlparse(candidate).query):
        if _QUERY_DATE.match(value):
            return value.strip()
    return None


def find_date_links(html: str, page_url: str) -> list[DateLink]:
    """Every other departure this page offers, in the order it prints them.

    Returns an empty list for a page that offers none, which is the ordinary
    case: liyaseir and vanrosh list their dates as text with one price table
    for all of them, and those are multiplied rather than fetched — see
    :func:`crawler.extraction.page_facts._across_departures`.

    Example:
        >>> html = ('<a href="?date=1405-06-06">06</a>'
        ...         '<a href="?date=1405-06-11">11</a>'
        ...         '<a href="?date=1405-06-11">11 again</a>'
        ...         '<a href="/tour/other">a different tour</a>')
        >>> [(l.date, l.replaces_this_page)
        ...  for l in find_date_links(html, "https://bisanseir.com/tour/tbilisi")]
        [('1405-06-06', True), ('1405-06-11', True)]
    """
    found: list[DateLink] = []
    seen: set[str] = set()

    for node in HTMLParser(html).css("a"):
        href = (node.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        target = urljoin(page_url, href)

        date = _query_date(page_url, target)
        if date is not None:
            key = f"date:{date}"
            if key not in seen:
                seen.add(key)
                found.append(DateLink(url=target, date=date, replaces_this_page=True))
            continue

        if _is_sibling(page_url, target) and target not in seen:
            seen.add(target)
            found.append(DateLink(url=target))

    return found
