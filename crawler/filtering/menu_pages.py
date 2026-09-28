"""Telling a menu of tours apart from a tour, after the page has been read.

Some sites publish a page per destination that sells nothing itself and only
links to the tours underneath it — ``bisanseir.com/tours/تور-کیش`` holds two
real tours with all their dates. Measured on the run of 2026-09-23, 58 of that
site's 121 tours came from such pages: one empty package each, no price, no
hotel, and a ``tour_code`` minted and kept forever.

The path cannot decide it. Two other sites put their real tours under exactly
the plural path a listing uses:

    bisanseir.com/tours/…    58 listings
    vanrosh.com/tours/…     199 real tours
    tahagasht.com/tours/…    90 real tours

So the decision is made twice, on shape, after reading:

* ``says_nothing`` — the page produced no price in any currency, no hotel name
  and no star. All four have to be missing together. Dropping any one of them
  widens it from 58 pages to between 126 and 337, and real tours start being
  caught: a combined tour priced in dollars, or a five-star hotel named only
  by its grade.
* ``tour_links`` — the page carries links to tour pages with a price inside
  the link. A price is what separates an offer from the "تور های پیشنهادی"
  strip, which is a bare title and an image, and which would otherwise make
  every empty page look like a menu — the same trap as issue #40.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

_ANCHOR = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.S | re.I)

# Six digits or more: a tour price, never a night count or a star rating.
_MONEY = re.compile(r"[\d۰-۹٠-٩][\d۰-۹٠-٩,،٬]{5,}")

_TAGS = re.compile(r"<[^>]+>")


def says_nothing(packages: list[dict]) -> bool:
    """True when nothing sellable came out of the page.

    A package speaks if it states a price in any currency, names a hotel, or
    grades one. Only when every package is silent on all four is the page
    worth a second look.

    Example:
        >>> says_nothing([])
        True
        >>> says_nothing([{"rooms": [{"final_price": 0, "currency_price": ""}],
        ...                "hotels": [{"name": "", "star_code": "0"}]}])
        True
        >>> says_nothing([{"rooms": [{"final_price": 0, "currency_price": "775.00"}],
        ...                "hotels": [{"name": "", "star_code": ""}]}])
        False
        >>> says_nothing([{"rooms": [], "hotels": [{"name": "", "star_code": "5"}]}])
        False
    """
    return not any(_speaks(package) for package in packages)


def _speaks(package: dict) -> bool:
    """Whether one package states any of the four things a tour states."""
    rooms = package.get("rooms") or []
    hotels = package.get("hotels") or []
    return (
        any(_number(room.get("final_price")) > 0 for room in rooms)
        or any(_number(room.get("currency_price")) > 0 for room in rooms)
        or any((hotel.get("name") or "").strip() for hotel in hotels)
        or any(str(hotel.get("star_code") or "").strip() not in ("", "0")
               for hotel in hotels)
    )


def _number(value: object) -> float:
    """A price as a number, or zero when it is not one."""
    try:
        return float(str(value or 0).replace(",", ""))
    except ValueError:
        return 0.0


def tour_links(html: str, page_url: str) -> list[str]:
    """The tour pages this page offers, in the order they appear, without repeats.

    A link counts only when a price is written inside it. That is what the
    site's own offer blocks have and its recommendation strip does not, and it
    is why a page with seven links to other tours can still be empty.

    Example:
        >>> page = '''
        ...   <a href="/tour/kish"><span>تور کیش</span>
        ...      <b>23,900,000</b> تومان</a>
        ...   <a href="/tour/kish?date=1405-07-08">08 مهر <b>23,900,000</b></a>
        ...   <a href="/tour/china" class="suggested-tour"><img><span>تور چین</span></a>
        ... '''
        >>> tour_links(page, "https://bisanseir.com/tours/kish")
        ['https://bisanseir.com/tour/kish', 'https://bisanseir.com/tour/kish?date=1405-07-08']
    """
    here = urlsplit(page_url)
    found: list[str] = []
    for href, inside in _ANCHOR.findall(html):
        if not _MONEY.search(_TAGS.sub(" ", inside)):
            continue
        link = urljoin(page_url, href.strip()).split("#")[0]
        target = urlsplit(link)
        if target.netloc != here.netloc or target.path == here.path:
            continue
        if link not in found:
            found.append(link)
    return found
