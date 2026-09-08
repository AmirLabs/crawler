"""Reading a tour from the site's own API instead of its markup.

alefbatour.com and avisatravel.com are built on one platform, and that platform
serves the tour twice: once as a page whose hotel list is filled in by
JavaScript, and once as JSON. We were reaching the hotels by driving a headless
browser — two requests, a browser launch and an eight second wait per page. The
JSON is one small GET, and it is better data as well as cheaper:

* ``isExpired`` instead of matching Persian notices in the page text.
* ``departDateTime`` as an ISO timestamp instead of parsing "۲۹ مرداد ساعت ۰۵:۴۵".
* ``grade: 4.0`` instead of counting stars.
* ``currencyCode`` and ``isNative`` instead of deciding which half of a price
  is the local one.
* ``origin.city`` and ``destination.city`` as fields, which for these two sites
  removes the LLM call that reads places out of the title.

Nothing here is hardcoded to a domain. The page states where its API lives:

    const windowPreludeData = {"baseUrl":{"queryBaseUrl":"https://echo.alefbatour.com/api", ...

so the base is read from the page and the tour's id from its own URL. A site
that does not publish that declaration simply has no API path, and extraction
falls back to reading the markup.

The API supplements the page rather than replacing it. The picture, the agency
name and the logo come from the document's Open Graph tags, which the JSON does
not carry; the offers, dates, places and transport come from the JSON, which
knows them exactly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from crawler.extraction.readers.places import is_a_departure_city
from crawler.extraction.models import (
    METHOD_EXACT_KEY,
    ExtractedField,
    ExtractedPage,
    ExtractedRow,
)
from crawler.extraction.readers.labels import detect_room_type
from crawler.extraction.target_fields import PAGE_FIELDS, ROW_FIELDS, unresolved

# Where the platform declares its read-only API. Matched on the declaration
# rather than on a host name, so a site on the same platform under any domain
# is found the same way.
_QUERY_BASE = re.compile(r'"queryBaseUrl"\s*:\s*"([^"]+)"')

# The tour's id inside its own URL: /tour/d/152839/<slug>. Taken as the longest
# run of digits in the path, since the slug may hold small numbers of its own
# ("تور-7شب-ارمنستان").
_ID_IN_PATH = re.compile(r"/(\d{4,})(?:/|$)")

# How the platform names a way of travelling, and what our schema calls it.
_TRAVEL_MODES = {
    "flight": "air",
    "bus": "bus",
    "train": "train",
    "ship": "ship",
    "car": "road",
}


def find_api_base(html: str) -> Optional[str]:
    """The read-only API base a page declares for itself.

    Example:
        >>> find_api_base('<script>const d = {"baseUrl":'
        ...               '{"queryBaseUrl":"https://echo.alefbatour.com/api"}}</script>')
        'https://echo.alefbatour.com/api'
        >>> find_api_base('<html><body>hi</body></html>') is None
        True
    """
    match = _QUERY_BASE.search(html)
    return match.group(1).rstrip("/") if match else None


def tour_id_from_url(url: str) -> Optional[str]:
    """The tour's numeric id, taken from its own URL.

    Example:
        >>> tour_id_from_url("https://www.alefbatour.com/tour/d/152839/%D8%AA%D9%88%D8%B1")
        '152839'
        >>> tour_id_from_url("https://bisanseir.com/tour/tour-baku") is None
        True
    """
    found = _ID_IN_PATH.findall(urlparse(url).path)
    return max(found, key=len) if found else None


def api_url_for(page_url: str, html: str) -> Optional[str]:
    """The JSON address for one tour page, or ``None`` if the site has no API.

    Example:
        >>> api_url_for("https://www.alefbatour.com/tour/d/152839/x",
        ...             '{"queryBaseUrl":"https://echo.alefbatour.com/api"}')
        'https://echo.alefbatour.com/api/tours/152839'
    """
    base = find_api_base(html)
    tour_id = tour_id_from_url(page_url)
    if not base or not tour_id:
        return None
    return f"{base}/tours/{tour_id}"


async def fetch_tour_json(
    client: httpx.AsyncClient, api_url: str
) -> tuple[Optional[dict], Optional[str]]:
    """Fetch one tour's JSON, returning the reason rather than raising on failure.

    A site that changes its API should leave the crawl reading markup, not stop
    it, so every failure comes back as a message for the page's notes.

    Example:
        >>> async with httpx.AsyncClient() as c:
        ...     data, error = await fetch_tour_json(
        ...         c, "https://echo.alefbatour.com/api/tours/152839")
        >>> data["stayCount"], error
        (7, None)
    """
    try:
        response = await client.get(api_url, headers={"Accept": "application/json"})
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if response.status_code != 200:
        return None, f"api returned {response.status_code}"
    try:
        data = response.json()
    except Exception:
        return None, "api did not return JSON"
    return (data, None) if isinstance(data, dict) else (None, "api returned a list")


def _iso_date(stamp: Optional[str]) -> Optional[dict]:
    """One API timestamp in the shape a row's date field uses.

    Example:
        >>> _iso_date("2026-08-21T13:55:00")
        {'iso': '2026-08-21', 'raw': '2026-08-21T13:55:00', 'time': '13:55'}
        >>> _iso_date(None) is None
        True
    """
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return {"iso": when.date().isoformat(), "raw": stamp, "time": when.strftime("%H:%M")}


def _price_value(entry: dict) -> Optional[dict]:
    """One price, with the halves told apart by the API rather than by us.

    ``isNative`` marks the agency's own currency, so there is nothing to infer
    from the order the amounts appear in.

    Example:
        >>> _price_value({"price": [
        ...     {"amount": 43900000.0, "currencyCode": "IRT", "isNative": True},
        ...     {"amount": 592.0, "currencyCode": "USD", "isNative": False}]})
        {'amount': 43900000.0, 'currency': 'IRT', 'foreign_amount': 592.0,
         'foreign_currency': 'USD', 'raw': '43900000.0 IRT + 592.0 USD'}
    """
    parts = [p for p in (entry.get("price") or []) if isinstance(p, dict)]
    if not parts:
        return None
    local = next((p for p in parts if p.get("isNative")), parts[0])
    foreign = next((p for p in parts if p is not local), None)
    raw = f"{local.get('amount')} {local.get('currencyCode')}"
    if foreign:
        raw += f" + {foreign.get('amount')} {foreign.get('currencyCode')}"
    return {
        "amount": local.get("amount"),
        "currency": local.get("currencyCode"),
        "foreign_amount": foreign.get("amount") if foreign else None,
        "foreign_currency": foreign.get("currencyCode") if foreign else None,
        "raw": raw,
    }


def _legs(data: dict) -> tuple[Optional[dict], Optional[dict]]:
    """The journey out of Iran and the one back, whatever order they are listed.

    Not simply the first and the second. radingasht's Antalya tour lists four
    legs and the first is a coach from اسپارتا to آنتالیا, a transfer inside the
    destination — taking it made the tour leave from اسپارتا by bus on a date
    six weeks before it sells. The leg that starts at an Iranian airport is the
    one the tour departs on, and the leg that ends at one is the way home.

    Falls back to the listed order where no leg names a place the table knows,
    which is what a tour between two foreign cities looks like.

    Example:
        >>> out, back = _legs({"transportation": [
        ...     {"departDateTime": "2026-08-21T13:55:00"},
        ...     {"departDateTime": "2026-08-28T14:00:00"}]})
        >>> out["departDateTime"], back["departDateTime"]
        ('2026-08-21T13:55:00', '2026-08-28T14:00:00')
        >>> out, back = _legs({"transportation": [
        ...     {"origin": {"city": "اسپارتا"}, "destination": {"city": "آنتالیا"}},
        ...     {"origin": {"city": "تهران"}, "destination": {"city": "آنتالیا"}},
        ...     {"origin": {"city": "آنتالیا"}, "destination": {"city": "تهران"}}]})
        >>> out["origin"]["city"], back["destination"]["city"]
        ('تهران', 'تهران')
    """
    legs = [t for t in (data.get("transportation") or []) if isinstance(t, dict)]
    if not legs:
        return None, None

    def city(leg: dict, end: str) -> str:
        return ((leg.get(end) or {}).get("city") or "").strip()

    leaving = next((leg for leg in legs if is_a_departure_city(city(leg, "origin"))), None)
    coming = next(
        (leg for leg in reversed(legs)
         if is_a_departure_city(city(leg, "destination"))), None
    )
    if leaving is None and coming is None:
        return legs[0], legs[1] if len(legs) > 1 else None
    return leaving or legs[0], coming or (legs[-1] if len(legs) > 1 else None)


def _rows_from_bundles(data: dict) -> list[ExtractedRow]:
    """One row per hotel per room type, which is the grain the client asked for.

    Example:
        >>> rows = _rows_from_bundles({
        ...     "stayCount": 7,
        ...     "transportation": [{"departDateTime": "2026-08-21T13:55:00"}],
        ...     "accommodationBundles": [{
        ...         "accommodations": [{"name": "هتل باکسوز", "grade": 3.0,
        ...             "locationName": "ایروان",
        ...             "boardType": {"boardType": "BB"}}],
        ...         "prices": [{"title": "قیمت 2 تخته (هرنفر)",
        ...             "price": [{"amount": 68990000.0, "currencyCode": "IRT",
        ...                        "isNative": True}]}]}]})
        >>> len(rows), rows[0].fields["hotel_name"].value, rows[0].fields["room_type"].value
        (1, 'هتل باکسوز', 'double')
    """
    outbound, inbound = _legs(data)
    departure = _iso_date((outbound or {}).get("departDateTime"))
    ret = _iso_date((inbound or {}).get("departDateTime"))
    nights = data.get("stayCount")

    rows: list[ExtractedRow] = []
    for bundle in data.get("accommodationBundles") or []:
        if not isinstance(bundle, dict):
            continue
        stays = [a for a in (bundle.get("accommodations") or []) if isinstance(a, dict)]
        # The flat names stay for the readers that can only carry one hotel;
        # ``hotels`` below is the same stays with nothing thrown away.
        names = " + ".join(a.get("name") or a.get("latinName") or "" for a in stays).strip(" +")
        grades = [a.get("grade") for a in stays if isinstance(a.get("grade"), (int, float))]
        boards = [(a.get("boardType") or {}).get("boardType") for a in stays]
        cities = [a.get("locationName") for a in stays if a.get("locationName")]

        for entry in bundle.get("prices") or []:
            if not isinstance(entry, dict):
                continue
            price = _price_value(entry)
            if price is None:
                continue
            fields: dict[str, ExtractedField] = {}

            def put(name: str, value: Any, source: str) -> None:
                if value not in (None, "", []):
                    fields[name] = ExtractedField(value, METHOD_EXACT_KEY, source)

            put("hotel_name", names or None, "accommodations[].name")
            put("hotel_star_rating", int(grades[0]) if grades else None, "accommodations[].grade")
            put("board_type", boards[0] if boards else None, "accommodations[].boardType")
            put("hotel_city", " + ".join(cities) if cities else None, "accommodations[].locationName")
            put("hotels", [_stay(a) for a in stays] or None, "accommodations[]")
            put("room_type", detect_room_type(entry.get("title") or ""), "prices[].title")
            put("price", price, "prices[].price")
            put("nights", nights, "stayCount")
            put("departure_date", departure, "transportation[0].departDateTime")
            put("return_date", ret, "transportation[1].departDateTime")

            rows.append(
                ExtractedRow(
                    fields=fields,
                    unresolved_fields=unresolved(fields, ROW_FIELDS),
                    source_path="api:accommodationBundles[].prices[]",
                    index=len(rows),
                )
            )
    return rows


def _stay(accommodation: dict) -> dict:
    """One accommodation as the fields a hotel needs, nothing joined.

    The API states every hotel of a multi-city stay in full — its own grade,
    board and ``stayCount`` — and the flat ``hotel_*`` fields could carry only
    the first, so a Bali tour reported 4 stars and lost the 3 nights in Ubud
    entirely.

    Example:
        >>> _stay({"name": "ده ادن", "latinName": "The Eden", "grade": 4.0,
        ...        "locationName": "بالی", "stayCount": 4,
        ...        "boardType": {"boardType": "BB"}})
        {'name': 'ده ادن', 'name_latin': 'The Eden', 'star_rating': 4, 'city': 'بالی', 'board_type': 'BB', 'nights': 4}
    """
    grade = accommodation.get("grade")
    stay = accommodation.get("stayCount")
    board = (accommodation.get("boardType") or {}).get("boardType")
    return {
        "name": accommodation.get("name") or accommodation.get("latinName"),
        "name_latin": accommodation.get("latinName"),
        "star_rating": int(grade) if isinstance(grade, (int, float)) else None,
        "city": accommodation.get("locationName"),
        "board_type": board,
        "nights": int(stay) if isinstance(stay, (int, float)) else None,
    }


def _stamp_transport(rows: list[ExtractedRow], outbound: Optional[dict]) -> None:
    """Give every row the carrier and the flight time the API already stated.

    These are row fields because a page can sell several departures, but the
    API states them once for the journey, so they are stamped rather than read
    per row. Both sat at 0% on all ten sites while three of them published the
    answer: the carrier's Persian name and the leg's duration were simply never
    taken out of the JSON.

    Example:
        >>> rows = [ExtractedRow()]
        >>> _stamp_transport(rows, {"travelDuration": "03:00",
        ...     "carrierInfo": {"persianName": "کاسپین "}})
        >>> rows[0].fields["transport_name"].value
        'کاسپین'
        >>> rows[0].fields["transport_duration"].value
        '03:00'
    """
    if not outbound:
        return
    carrier = ((outbound.get("carrierInfo") or {}).get("persianName") or "").strip()
    duration = (outbound.get("travelDuration") or "").strip()
    for row in rows:
        for name, value, source in (
            ("transport_name", carrier, "transportation[].carrierInfo.persianName"),
            ("transport_duration", duration, "transportation[].travelDuration"),
        ):
            if value and name not in row.fields:
                row.fields[name] = ExtractedField(value, METHOD_EXACT_KEY, source)
                row.attempts.pop(name, None)
        row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)


def enrich_from_api(page: ExtractedPage, data: dict) -> ExtractedPage:
    """Replace what the markup guessed with what the API states.

    Only the fields the JSON actually carries are overwritten. The picture, the
    agency name and the logo stay as the document's Open Graph tags gave them,
    because the JSON has no equivalent.

    ``expired_rows`` is left alone: this sets the offers, and whether they have
    already departed is decided afterwards by
    :func:`crawler.extraction.availability.split_by_availability`, so one rule
    covers both the markup path and this one.

    Example:
        >>> page = ExtractedPage(url="https://www.alefbatour.com/tour/d/1/x")
        >>> page = enrich_from_api(page, {
        ...     "title": "تور 7شب ارمنستان", "isExpired": False, "stayCount": 7,
        ...     "transportation": [{"travelMode": "Flight",
        ...         "origin": {"city": "تهران"}, "destination": {"city": "ایروان"}}],
        ...     "accommodationBundles": []})
        >>> page.page_fields["origin"].value, page.page_fields["destination"].value
        ('تهران', 'ایروان')
        >>> page.page_fields["transportation_type"].value
        'air'
    """
    outbound, _ = _legs(data)

    def put(name: str, value: Any, source: str) -> None:
        if value not in (None, "", []):
            page.page_fields[name] = ExtractedField(value, METHOD_EXACT_KEY, source)

    put("tour_name", data.get("title"), "title")
    if outbound:
        mode = (outbound.get("travelMode") or "").lower()
        put("transportation_type", _TRAVEL_MODES.get(mode), "transportation[].travelMode")
        put("origin", (outbound.get("origin") or {}).get("city"), "transportation[].origin.city")
        destination = (outbound.get("destination") or {}).get("city")
        put("destination", destination, "transportation[].destination.city")
        put("destinations", [destination] if destination else None,
            "transportation[].destination.city")

    page.unresolved_page_fields = unresolved(page.page_fields, PAGE_FIELDS)
    page.rows = _rows_from_bundles(data)
    page.notes.append(
        f"offers from the site's API: {len(page.rows)} rows"
        + (" (the site marks this tour expired)" if data.get("isExpired") else "")
    )
    return page
