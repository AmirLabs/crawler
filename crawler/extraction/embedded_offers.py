"""Offers a page publishes as data, in a shape its own markup never shows.

Most sites lay their packages out in HTML and
:mod:`crawler.extraction.row_mapping` reads them. chakavaktravel.com does not:
its markup holds a menu, a banner and nothing else, so the crawl reported the
site as empty. Every tour it sells is in the page all the same, streamed
through Next.js as data, and :func:`~crawler.extraction.extractor
._next_payload_objects` now hands it over.

What is read here is that site's own vocabulary — ``doubleBedPrice``,
``hotel_start``, ``gallery.path`` — the same bargain
:mod:`crawler.extraction.tour_api` already makes for the three sites running
the ``echo`` platform. It is kept in its own module for the same reason: a
site that renames a field breaks one file and nothing else.

The shape is recognised rather than the site. A list whose entries each carry
a departure date, a price and a mapping of hotels is this shape, whoever
serves it; a page that publishes anything else is left to the markup readers.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import quote, urljoin

from crawler.extraction.date_links import DateLink
from crawler.extraction.models import (
    METHOD_EXACT_KEY,
    ExtractedField,
    ExtractedRow,
)
from crawler.extraction.readers.dates import find_dates, to_iso
from crawler.extraction.readers.prices import CURRENCY_WORDS
from crawler.schema.output_models import (
    ROOM_CHILD_NO_BED,
    ROOM_CHILD_WITH_BED,
    ROOM_DOUBLE,
    ROOM_INFANT,
    ROOM_SINGLE,
)

# The key holding the list of offers, and the keys one offer is known by. All
# three have to be there before this reader claims a blob: a page carries a
# great deal of JSON, and only this shape is a price list.
_OFFERS = "hotel_data"
_REQUIRED = ("start_date", "hotels")

# A date as the offers write it: "1405/07/18".
_SLASH_DATE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")

# What each price key sells, in the order the client's rooms are named.
_ROOMS: tuple[tuple[str, str, str], ...] = (
    ("doubleBedPrice", "second_doubleBedPrice", ROOM_DOUBLE),
    ("singleBedPrice", "second_singleBedPrice", ROOM_SINGLE),
    ("childBedPrice", "second_childBedPrice", ROOM_CHILD_WITH_BED),
    ("childNoBedPrice", "second_childNoBedPrice", ROOM_CHILD_NO_BED),
    ("babyPrice", "second_babyPrice", ROOM_INFANT),
)


def offers_in(blobs: list[dict]) -> Optional[list[dict]]:
    """The offer list inside these blobs, or ``None`` where none of them is one.

    Example:
        >>> offers_in([{"data": {"hotel_data": [
        ...     {"start_date": "1405/07/11", "hotels": {}}]}}])
        [{'start_date': '1405/07/11', 'hotels': {}}]
        >>> offers_in([{"data": {"hotel_data": [{"start_date": "1405/07/11"}]}}])
        >>> offers_in([{"title": "تور کیش"}])
    """
    for blob in blobs:
        for holder in (blob, blob.get("data")):
            if not isinstance(holder, dict):
                continue
            found = holder.get(_OFFERS)
            if not isinstance(found, list) or not found:
                continue
            first = found[0]
            if isinstance(first, dict) and all(k in first for k in _REQUIRED):
                return found
    return None


def offer_date_links(blobs: list[dict], page_url: str) -> list[DateLink]:
    """The other departures listed beside the offers, as ``?date=`` links.

    The offers hold the date on screen only; every date sits in a list beside
    them, and the server answers ``?date=1405/07/18`` with that date's offers.
    Measured on chakavaktravel: 29 departures to come, 9 read.

    Example:
        >>> offer_date_links([{"data": {"filter": ["1405/07/11", "1405/07/18", "1405/07/18"],
        ...     "hotel_data": [{"start_date": "1405/07/11", "hotels": {}}]}}], "https://x.com/tour/a")
        [DateLink(url='https://x.com/tour/a?date=1405/07/11', date='1405-07-11', replaces_this_page=True), DateLink(url='https://x.com/tour/a?date=1405/07/18', date='1405-07-18', replaces_this_page=True)]
        >>> offer_date_links([{"title": "تور کیش"}], "https://x.com/tour/a")
        []
    """
    base = page_url.split("?")[0]
    for blob in blobs:
        for holder in (blob, blob.get("data")):
            if not isinstance(holder, dict) or offers_in([holder]) is None:
                continue
            for value in holder.values():
                if not isinstance(value, list) or not value:
                    continue
                parts = [_SLASH_DATE.match(str(item)) for item in value]
                if not all(parts):
                    continue
                links: dict[str, DateLink] = {}
                for raw, found in zip(value, parts):
                    jalali = "-".join(f"{int(n):02d}" for n in found.groups())
                    links.setdefault(jalali, DateLink(
                        url=f"{base}?date={quote(str(raw), safe='/')}",
                        date=jalali, replaces_this_page=True))
                return list(links.values())
    return []


def _number(value: Any) -> Optional[float]:
    """One of their price strings as a number, or nothing where it is blank.

    Example:
        >>> _number("150000000"), _number("0"), _number(""), _number(None)
        (150000000.0, None, None, None)
    """
    try:
        amount = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return amount or None


def _price(offer: dict, local_key: str, foreign_key: str) -> Optional[dict]:
    """One room's price, in whichever of the two currencies the offer names.

    Both halves are kept: these tours are sold as an amount in toman plus an
    amount in dollars, and dropping either leaves a price nobody is charged.

    Example:
        >>> _price({"currency": "تومان", "doubleBedPrice": "150000000",
        ...         "secondCurrency": "دلار", "second_doubleBedPrice": "240"},
        ...        "doubleBedPrice", "second_doubleBedPrice")
        {'amount': 150000000.0, 'currency': 'IRT', 'foreign_amount': 240.0, 'foreign_currency': 'USD', 'raw': '150000000 تومان + 240 دلار'}
        >>> _price({"currency": "تومان", "doubleBedPrice": "0"},
        ...        "doubleBedPrice", "second_doubleBedPrice") is None
        True
    """
    local = _number(offer.get(local_key))
    foreign = _number(offer.get(foreign_key))
    if local is None and foreign is None:
        return None
    local_word = str(offer.get("currency") or "").strip()
    foreign_word = str(offer.get("secondCurrency") or "").strip()
    parts = []
    if local is not None:
        parts.append(f"{offer.get(local_key)} {local_word}".strip())
    if foreign is not None:
        parts.append(f"{offer.get(foreign_key)} {foreign_word}".strip())
    return {
        "amount": local,
        "currency": CURRENCY_WORDS.get(local_word) if local is not None else None,
        "foreign_amount": foreign,
        "foreign_currency": (
            CURRENCY_WORDS.get(foreign_word) if foreign is not None else None
        ),
        "raw": " + ".join(parts),
    }


def _date(text: Any) -> Optional[dict]:
    """One of their dates in the shape a row's date field uses.

    Example:
        >>> _date("1405/07/11")
        {'iso': '2026-10-03', 'raw': '1405/07/11', 'time': None}
        >>> _date("") is None
        True
    """
    if not isinstance(text, str) or not text.strip():
        return None
    found = find_dates(text)
    if not found:
        return None
    return {"iso": to_iso(found[0]), "raw": text.strip(), "time": None}


def _photograph(hotel: dict, page_url: str) -> Optional[str]:
    """The hotel's picture, where the entry carries one.

    13 of this site's 29 offers name one and the rest give an empty list, so
    the absence is the site's answer rather than a gap in the reading.

    The path is written from the site root with no leading slash, so it is
    joined onto the root rather than onto the page: resolved against the page
    it would land under ``/tour/``, which is not where the file is.

    Example:
        >>> _photograph({"gallery": {"path": "images/upload/hotels/a.jpg"}},
        ...             "https://chakavaktravel.com/tour/kuala-lumpur-tour")
        'https://chakavaktravel.com/images/upload/hotels/a.jpg'
        >>> _photograph({"gallery": []}, "https://x.com/t") is None
        True
    """
    gallery = hotel.get("gallery")
    if isinstance(gallery, list):
        gallery = gallery[0] if gallery else None
    if not isinstance(gallery, dict):
        return None
    path = gallery.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    return urljoin(page_url, "/" + path.strip().lstrip("/"))


def rows_from_offers(offers: list[dict], page_url: str = "") -> list[ExtractedRow]:
    """One row per departure, per hotel and per room type.

    That is the grain the client asked for, and it is the grain this list is
    already in — an entry is one date and one hotel, and only the room types
    have to be unfolded.

    Example:
        >>> rows = rows_from_offers([{
        ...     "start_date": "1405/07/11", "end_date": "1405/07/18",
        ...     "currency": "تومان", "doubleBedPrice": "150000000",
        ...     "secondCurrency": "دلار", "second_doubleBedPrice": "240",
        ...     "hotels": {"470195": {"hotel_name": "LAZDANA", "hotel_start": 3,
        ...                           "night_count": 7, "board": "BB",
        ...                           "room_type": "STD", "city": "کوالالامپور"}}}])
        >>> len(rows), rows[0].fields["hotel_name"].value
        (1, 'LAZDANA')
        >>> rows[0].fields["hotel_star_rating"].value, rows[0].fields["nights"].value
        (3, 7)
    """
    out: list[ExtractedRow] = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        hotels = offer.get("hotels")
        if not isinstance(hotels, dict):
            continue
        departure, ret = _date(offer.get("start_date")), _date(offer.get("end_date"))
        for hotel in hotels.values():
            if not isinstance(hotel, dict):
                continue
            for local_key, foreign_key, room in _ROOMS:
                price = _price(offer, local_key, foreign_key)
                if price is None:
                    continue
                fields: dict[str, ExtractedField] = {}

                def put(name: str, value: Any, source: str) -> None:
                    if value not in (None, "", []):
                        fields[name] = ExtractedField(value, METHOD_EXACT_KEY, source)

                put("price", price, f"{_OFFERS}[].{local_key}")
                put("room_type", room, f"{_OFFERS}[].{local_key}")
                put("departure_date", departure, f"{_OFFERS}[].start_date")
                put("return_date", ret, f"{_OFFERS}[].end_date")
                put("hotel_name", hotel.get("hotel_name"), "hotels{}.hotel_name")
                put("board_type", hotel.get("board"), "hotels{}.board")
                put("hotel_city", hotel.get("city"), "hotels{}.city")
                put("hotel_image", _photograph(hotel, page_url), "hotels{}.gallery")
                grade, nights = hotel.get("hotel_start"), hotel.get("night_count")
                put("hotel_star_rating",
                    int(grade) if isinstance(grade, (int, float)) else None,
                    "hotels{}.hotel_start")
                put("nights",
                    int(nights) if isinstance(nights, (int, float)) else None,
                    "hotels{}.night_count")
                out.append(ExtractedRow(fields=fields))
    return out
