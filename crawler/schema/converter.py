"""Step 4: turning what was extracted into the validated output shape.

:class:`~crawler.extraction.models.ExtractedPage` describes a *page* — named
values and the rows they came from. :class:`~crawler.schema.output_models.Tour`
describes a *product* — a tour with the options a customer can book. This
module is the step between, and it makes one judgement that the extraction
stage deliberately left alone: which rows are the same bookable option.

A row is whatever the page repeated, and the target sites repeat different
things. bisanseir.com repeats room types::

    دو تخته      82,000,000 تومان
    یک تخته      82,000,000 تومان
    کودک با تخت  80,000,000 تومان

Those are not three offers. They are one hotel on one date with three prices —
which is exactly why :class:`~crawler.schema.output_models.TourOffer` keys its
prices by room type. Emitting one offer each would put three near-identical
rows in front of the search engine and make the cheapest look like three
different tours.

avisatravel.com repeats departure dates instead, and there each row really is
its own offer. Both fall out of one rule: **rows are grouped by the departure
date and hotel they name, and the room type decides which price slot they
fill.** Nothing site-specific is needed to tell the two apart.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from crawler.extraction.models import ExtractedPage, ExtractedRow
from crawler.extraction.text_dates import gregorian_to_jalali
from crawler.extraction.text_labels import detect_nights
from crawler.schema.output_models import (
    ROOM_DOUBLE,
    Agency,
    Hotel,
    Money,
    Price,
    Provenance,
    Tour,
    TourDate,
    TourOffer,
    Transport,
)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _value(row_or_page: dict[str, Any], name: str) -> Optional[Any]:
    """Read a resolved field's value, or ``None`` when it was not resolved.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_EXACT_KEY
        >>> _value({"tour_name": ExtractedField("تور دبی", METHOD_EXACT_KEY)}, "tour_name")
        'تور دبی'
        >>> _value({}, "tour_name") is None
        True
    """
    field = row_or_page.get(name)
    return field.value if field is not None else None


def _jalali_string(iso: Optional[str]) -> Optional[str]:
    """Render a Gregorian ISO date back in the Jalali calendar.

    Derived from ``iso`` rather than from the raw text on purpose: the raw is
    what the page printed, which is usually missing its year and carries the
    departure time along with it. Converting the resolved date instead gives
    every row the same complete, sortable shape.

    Example:
        >>> _jalali_string("2026-08-29")
        '1405-06-07'
        >>> _jalali_string(None) is None
        True
    """
    if not iso:
        return None
    try:
        parsed = date.fromisoformat(iso)
    except ValueError:
        return None
    jy, jm, jd = gregorian_to_jalali(parsed.year, parsed.month, parsed.day)
    return f"{jy:04d}-{jm:02d}-{jd:02d}"


def _tour_date(value: Optional[Any]) -> TourDate:
    """Build a :class:`TourDate` from what :mod:`row_mapping` recorded.

    Example:
        >>> _tour_date({"iso": "2026-08-29", "raw": "07 شهریور", "time": "22:30"})
        TourDate(raw='07 شهریور', iso='2026-08-29', jalali='1405-06-07', time='22:30')
        >>> _tour_date(None)
        TourDate(raw=None, iso=None, jalali=None, time=None)
    """
    if isinstance(value, dict):
        iso = value.get("iso")
        iso = iso if isinstance(iso, str) and _ISO_DATE.match(iso) else None
        return TourDate(
            raw=value.get("raw"),
            iso=iso,
            jalali=_jalali_string(iso),
            time=value.get("time"),
        )
    if isinstance(value, str):
        # A plain string reached here from an older shape; keep it as raw.
        iso = value if _ISO_DATE.match(value) else None
        return TourDate(raw=value, iso=iso, jalali=_jalali_string(iso))
    return TourDate()


def _price(value: Optional[Any]) -> Optional[Price]:
    """Build a :class:`Price`, keeping both currency halves.

    Example:
        >>> _price({"amount": 92000000.0, "currency": "IRT", "foreign_amount": 1090.0,
        ...         "foreign_currency": "EUR", "raw": "92,000,000 تومان + 1,090 یورو"})
        Price(local=Money(amount=92000000.0, currency='IRT'), foreign=Money(amount=1090.0, currency='EUR'), basis='per_person', raw='92,000,000 تومان + 1,090 یورو')
    """
    if not isinstance(value, dict):
        return None

    amount, currency = value.get("amount"), value.get("currency")
    local = Money(amount=amount, currency=currency) if amount is not None and currency else None

    foreign_amount, foreign_currency = value.get("foreign_amount"), value.get("foreign_currency")
    foreign = (
        Money(amount=foreign_amount, currency=foreign_currency)
        if foreign_amount is not None and foreign_currency
        else None
    )

    if local is None and foreign is None:
        return None
    return Price(local=local, foreign=foreign, raw=value.get("raw"))


# Flight times that move a night off the hotel bill. A tour leaving at 22:30
# spends its first night in the air, and one landing back at 02:00 checked out
# the evening before. Both bounds were read off 203 pages that state their own
# night count: the plain calendar span is right on 90% of them, and correcting
# for these two cases makes it right on all of them.
_LATE_DEPARTURE_HOUR = 21
_EARLY_RETURN_HOUR = 5


def _hour_of(clock: Optional[str]) -> Optional[int]:
    """Read the hour out of a ``HH:MM`` string.

    Example:
        >>> _hour_of("22:30"), _hour_of("02:00"), _hour_of(None)
        (22, 2, None)
    """
    if not clock:
        return None
    try:
        return int(clock.split(":")[0])
    except ValueError:
        return None


def _nights_between(departure: TourDate, ret: TourDate) -> Optional[int]:
    """Count the nights between two dates, as a last resort only.

    Calendar arithmetic is the weakest source for this number and is reached
    only when neither the row nor the tour's name states it — anything the site
    printed wins, because a derived figure that is wrong looks just as
    reasonable as a right one.

    The span is adjusted for the departure and return times, which is what
    makes it agree with the sites: a tour sold as "7 شب" that leaves at 22:30
    and lands back at 02:00 spans nine calendar days.

    Example:
        >>> _nights_between(TourDate(iso="2026-08-23", time="11:00"),
        ...                 TourDate(iso="2026-08-30", time="14:00"))
        7
        >>> _nights_between(TourDate(iso="2026-08-29", time="22:30"),
        ...                 TourDate(iso="2026-09-07", time="02:00"))
        7
        >>> _nights_between(TourDate(iso="2026-08-23"), TourDate()) is None
        True
    """
    if not departure.iso or not ret.iso:
        return None
    try:
        nights = (date.fromisoformat(ret.iso) - date.fromisoformat(departure.iso)).days
    except ValueError:
        return None

    departure_hour = _hour_of(departure.time)
    if departure_hour is not None and departure_hour >= _LATE_DEPARTURE_HOUR:
        nights -= 1

    return_hour = _hour_of(ret.time)
    if return_hour is not None and return_hour <= _EARLY_RETURN_HOUR:
        nights -= 1

    return nights if 1 <= nights <= 60 else None


def _group_key(row: ExtractedRow) -> tuple[Optional[str], Optional[str]]:
    """What makes two rows the same bookable option.

    The departure date and the hotel are what a customer books; the room type
    is what they choose *within* it, so it is deliberately not part of the key.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> row = ExtractedRow(fields={
        ...     "departure_date": ExtractedField({"iso": "2026-08-29"}, METHOD_HEURISTIC_HTML),
        ...     "room_type": ExtractedField("double", METHOD_HEURISTIC_HTML),
        ... })
        >>> _group_key(row)
        ('2026-08-29', None)
    """
    departure = _value(row.fields, "departure_date")
    if isinstance(departure, dict):
        date_key = departure.get("iso") or departure.get("raw")
    else:
        date_key = departure
    return date_key, _value(row.fields, "hotel_name")


def _build_offer(
    rows: list[ExtractedRow], nights_from_name: Optional[int] = None
) -> TourOffer:
    """Fold rows that share a date and hotel into one bookable option.

    Each row contributes its price under its own room type. A row that names no
    room type is filed under double occupancy, because that is the price these
    sites advertise as *the* price — the same assumption
    :attr:`~crawler.schema.output_models.TourOffer.adult_price` already makes.

    The night count is taken from the strongest source available: what the row
    stated, then what the tour's name stated, then :func:`_nights_between`.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> def row(room, amount):
        ...     return ExtractedRow(fields={
        ...         "room_type": ExtractedField(room, METHOD_HEURISTIC_HTML),
        ...         "price": ExtractedField(
        ...             {"amount": amount, "currency": "IRT"}, METHOD_HEURISTIC_HTML),
        ...     })
        >>> offer = _build_offer([row("double", 82_000_000), row("single", 99_000_000)])
        >>> sorted(offer.prices), offer.adult_price.local.amount
        (['double', 'single'], 82000000.0)
    """
    first = rows[0].fields

    departure = _tour_date(_value(first, "departure_date"))
    ret = _tour_date(_value(first, "return_date"))
    nights = (
        _value(first, "nights")
        or nights_from_name
        or _nights_between(departure, ret)
    )

    hotel = Hotel(
        name=_value(first, "hotel_name"),
        city=_value(first, "hotel_city"),
        star_rating=_value(first, "hotel_star_rating"),
        board_type=_value(first, "board_type"),
        nights=nights,
    )

    transport_name = _value(first, "transport_name")
    transport_duration = _value(first, "transport_duration")
    transport = (
        Transport(name=transport_name, duration=transport_duration)
        if transport_name or transport_duration
        else None
    )

    prices: dict[str, Price] = {}
    for row in rows:
        price = _price(_value(row.fields, "price"))
        if price is None:
            continue
        room_type = _value(row.fields, "room_type") or ROOM_DOUBLE
        prices.setdefault(room_type, price)

    return TourOffer(
        departure=departure,
        ret=ret,
        nights=nights,
        hotel=hotel,
        transport=transport,
        prices=prices,
        source_key=rows[0].source_path,
    )


def _field_methods(page: ExtractedPage) -> dict[str, str]:
    """Record which method filled each field, for the record on the tour.

    Page-level entries keep their own name; row-level ones are prefixed, so a
    ``price`` read from a row is never mistaken for a page-level one.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_EXACT_KEY
        >>> page = ExtractedPage(url="u", page_fields={
        ...     "tour_name": ExtractedField("x", METHOD_EXACT_KEY)})
        >>> _field_methods(page)
        {'tour_name': 'exact_key_match'}
    """
    methods = {name: field.method for name, field in page.page_fields.items()}
    for row in page.rows:
        for name, field in row.fields.items():
            methods.setdefault(f"offer.{name}", field.method)
    return methods


def build_tour(page: ExtractedPage) -> Tour:
    """Convert one extracted page into the tour it describes.

    Offers come out in the order their rows appeared, so a page listing
    departures by date produces them in that order rather than a shuffled set.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> page = ExtractedPage(
        ...     url="https://bisanseir.com/tour/1",
        ...     page_fields={"tour_name": ExtractedField("تور پاتایا", METHOD_HEURISTIC_HTML)},
        ...     rows=[ExtractedRow(fields={
        ...         "room_type": ExtractedField("double", METHOD_HEURISTIC_HTML),
        ...         "price": ExtractedField({"amount": 82000000.0, "currency": "IRT"},
        ...                                 METHOD_HEURISTIC_HTML),
        ...     })],
        ... )
        >>> tour = build_tour(page)
        >>> tour.tour_name, len(tour.offers), tour.offers[0].adult_price.local.amount
        ('تور پاتایا', 1, 82000000.0)
    """
    groups: dict[tuple[Optional[str], Optional[str]], list[ExtractedRow]] = {}
    for row in page.rows:
        groups.setdefault(_group_key(row), []).append(row)

    # Tour names carry the duration far more often than rows do —
    # "تور 7شب ارمنستان" — and a stated figure beats a derived one.
    tour_name = _value(page.page_fields, "tour_name")
    nights_from_name = detect_nights(tour_name) if isinstance(tour_name, str) else None

    return Tour(
        tour_url=page.url,
        tour_name=tour_name,
        origin=_value(page.page_fields, "origin"),
        destination=_value(page.page_fields, "destination"),
        transportation_type=_value(page.page_fields, "transportation_type"),
        image_url=_value(page.page_fields, "image_url"),
        is_installment_available=_value(page.page_fields, "is_installment_available"),
        agency=Agency(
            name=_value(page.page_fields, "agency_name"),
            website_url=_value(page.page_fields, "website_url"),
            logo_url=_value(page.page_fields, "logo_url"),
        ),
        offers=[
            _build_offer(rows, nights_from_name=nights_from_name)
            for rows in groups.values()
        ],
        provenance=Provenance(
            source_url=page.url,
            fetched_at=page.fetched_at,
            content_hash=page.content_hash,
            field_methods=_field_methods(page),
            llm_used=any(
                field.method == "llm"
                for field in list(page.page_fields.values())
                + [f for row in page.rows for f in row.fields.values()]
            ),
        ),
    )
