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
from datetime import date, timedelta
from typing import Any, Optional

from crawler.extraction.models import ExtractedPage, ExtractedRow
from crawler.extraction.readers.dates import gregorian_to_jalali, jalali_to_gregorian
from crawler.extraction.readers.labels import detect_nights
from crawler.extraction.readers.dates import find_period
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


# The year a stated month is sent under. The sites never write it, and the
# client wants a date rather than a month name, so one is chosen here. Bump it
# when the season turns: this is a fixed year, not a calculated one.
MONTH_YEAR = 1405

# The last day of each Jalali month. اسفند is 29 in an ordinary year.
_LAST_DAY = {1: 31, 2: 31, 3: 31, 4: 31, 5: 31, 6: 31,
             7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 29}


def _month_only(raw: Optional[Any]) -> Optional[str]:
    """The month a dateless departure names, as the client is sent it.

    nahalgasht sells by the month rather than the day — "مهر ماه" under
    تاریخ رفت — and 471 of its 738 tours reached the client with no departure
    at all. The month is sent in its place, without the "ماه" the site adds.

    The month is sent as its last day, which is what the client asked for so
    that every departure they hold is one shape. A tour sold in مهر is sent as
    1405-07-30.

    A season is its last day too. SEASONS already reads a season as the month
    it ends in, so "پاییز 1405" leaves on 1405-09-30 — the last day a tour sold
    for that season can leave.

    Example:
        >>> _month_only("مهر ماه")
        '1405-07-30'
        >>> _month_only("آبان ماه"), _month_only("خرداد ماه")
        ('1405-08-30', '1405-03-31')
        >>> _month_only("پاییز 1405"), _month_only("ویژه زمستان")
        ('1405-09-30', '1405-12-29')
        >>> _month_only("07 شهریور") is None
        True
        >>> _month_only(None) is None
        True
    """
    if not isinstance(raw, str):
        return None
    period = find_period(raw)
    if period is None or not period.month:
        return None
    return f"{MONTH_YEAR}-{period.month:02d}-{_LAST_DAY[period.month]:02d}"


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
            jalali=_jalali_string(iso) or _month_only(value.get("raw")),
            time=value.get("time"),
        )
    if isinstance(value, str):
        # A plain string reached here from an older shape; keep it as raw.
        iso = value if _ISO_DATE.match(value) else None
        return TourDate(raw=value, iso=iso, jalali=_jalali_string(iso))
    return TourDate()


def _without_zero_half(
    amount: Optional[float], foreign_amount: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """Drop the half of a two-part price that  is zero beside a real number.

    avisatravel's own API answers ``0 IRT + 950 EUR`` for a single room: the
    Toman half was never filled in and the 950 is the price. Kept, the zero
    reaches the client as "a single room costs nothing", and a wrong number
    costs more than a missing one.

    A zero standing alone is left alone. There the site has said zero with
    nothing beside it to contradict it, and that is the site's statement to
    make. Measured over the 165 audited rows of alefbatour and avisatravel:
    10 lone zeros, 1 zero beside a real number.

    Example:
        >>> _without_zero_half(0.0, 950.0)
        (None, 950.0)
        >>> _without_zero_half(0.0, None)
        (0.0, None)
    """
    if amount == 0 and foreign_amount not in (None, 0):
        return None, foreign_amount
    if foreign_amount == 0 and amount not in (None, 0):
        return amount, None
    return amount, foreign_amount


def _price(value: Optional[Any]) -> Optional[Price]:
    """Build a :class:`Price`, keeping both currency halves.

    Example:
        >>> _price({"amount": 92000000.0, "currency": "IRT", "foreign_amount": 1090.0,
        ...         "foreign_currency": "EUR", "raw": "92,000,000 تومان + 1,090 یورو"})
        Price(local=Money(amount=92000000.0, currency='IRT'),
              foreign=Money(amount=1090.0, currency='EUR'), basis='per_person',
              raw='92,000,000 تومان + 1,090 یورو', flight_included=None)
    """
    if not isinstance(value, dict):
        return None

    amount, currency = value.get("amount"), value.get("currency")
    foreign_amount, foreign_currency = value.get("foreign_amount"), value.get("foreign_currency")
    # ``raw`` is deliberately left as printed, so a dropped zero can still be
    # traced back to what the source actually said.
    amount, foreign_amount = _without_zero_half(amount, foreign_amount)

    local = Money(amount=amount, currency=currency) if amount is not None and currency else None
    foreign = (
        Money(amount=foreign_amount, currency=foreign_currency)
        if foreign_amount is not None and foreign_currency
        else None
    )

    if local is None and foreign is None:
        return None
    return Price(
        local=local,
        foreign=foreign,
        raw=value.get("raw"),
        flight_included=value.get("flight_included"),
    )


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


def _group_key(row: ExtractedRow) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """What makes two rows the same bookable option.

    The departure date, the hotel and how long the stay is. The room type is
    what a customer chooses *within* an option, so it is deliberately not part
    of the key; the length is not — nahalgasht sells the same hotel on the same
    date for two nights and for three at different prices, and with the length
    out of the key the second folded into the first and was lost.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> row = ExtractedRow(fields={
        ...     "departure_date": ExtractedField({"iso": "2026-08-29"}, METHOD_HEURISTIC_HTML),
        ...     "room_type": ExtractedField("double", METHOD_HEURISTIC_HTML),
        ... })
        >>> _group_key(row)
        ('2026-08-29', None, None)
    """
    departure = _value(row.fields, "departure_date")
    if isinstance(departure, dict):
        date_key = departure.get("iso") or departure.get("raw")
    else:
        date_key = departure
    return date_key, _value(row.fields, "hotel_name"), _value(row.fields, "nights")


def _impossible(departure: TourDate, ret: TourDate) -> bool:
    """Whether a stated return cannot belong to the departure beside it.

    A tour that comes home on the day it leaves, or before, is not a tour. It
    happens where a page sells several departures and one return is printed
    once: radingasht gave "9 شهریور → 9 شهریور" for a six-night tour on 25 rows.
    Those are dropped and counted from the night count instead.

    A return that merely disagrees with the night count is left alone. The page
    stated it, the count may be what is wrong, and which of the two to believe
    is not this function's call — issue #2 measures it.

    Example:
        >>> _impossible(TourDate(iso="2026-09-01"), TourDate(iso="2026-09-01"))
        True
        >>> _impossible(TourDate(iso="2026-09-01"), TourDate(iso="2026-09-08"))
        False
        >>> _impossible(TourDate(), TourDate(iso="2026-09-08"))
        False
    """
    return bool(departure.iso and ret.iso and ret.iso <= departure.iso)


def _return_from_nights(departure: TourDate, nights: Optional[int]) -> Optional[TourDate]:
    """The day a tour comes home, when the page said only how long it is.

    This belongs here rather than in step 3 because step 3 never holds both
    halves at once: bisanseir.com prints no night count in its rows at all, and
    reads 3 from the tour's name — which only happens in :func:`build_tour`.
    Deriving it earlier looked right and filled nothing, 0% on five of five
    pages audited.

    Only a row with no return at all is given one. A page that printed a return
    we could not resolve into a day keeps what it printed: what the site said
    is Amir's to judge, and replacing it with arithmetic hides the question.

    ``raw`` names the arithmetic, so a derived return is never read back as one
    the page stated. Whether the answer is the departure plus the nights or one
    day more is measured in issue #2; until that is settled the plus-nights
    reading is used, the same one :mod:`crawler.extraction.page_facts` uses.

    Example:
        >>> _return_from_nights(TourDate(iso="2026-09-12", raw="1405-06-21"), 3)
        TourDate(raw='1405-06-21 + 3 شب', iso='2026-09-15', jalali='1405-06-24', time=None)
        >>> _return_from_nights(TourDate(), 3) is None
        True
        >>> _return_from_nights(TourDate(iso="2026-09-12"), None) is None
        True
    """
    if not departure.iso or not nights:
        return None
    try:
        ends = date.fromisoformat(departure.iso) + timedelta(days=nights)
    except ValueError:
        return None
    iso = ends.isoformat()
    return TourDate(
        raw=f"{departure.raw or departure.iso} + {nights} شب",
        iso=iso,
        jalali=_jalali_string(iso),
    )


def _hotels_named(
    name: Optional[Any],
    city: Optional[Any],
    star_rating: Optional[Any],
    board_type: Optional[Any],
    nights: Optional[int],
    image_url: Optional[Any] = None,
    boards: Optional[Any] = None,
    stars: Optional[Any] = None,
    each_nights: Optional[Any] = None,
    cities: Optional[Any] = None,
) -> list[Hotel]:
    """One hotel per hotel, out of the one string the markup path joins them into.

    :func:`crawler.extraction.structure.hotel_block.read_hotel_block` joins the
    hotels of a multi-city stay with a plus, because a row has one field to put
    them in. The client's schema has a list, and sending four hotels as one
    name loses every one of them: lachinseir's Asia tour arrived as a single
    hotel called "IMPIANA HOTEL + MERCURE MAKKASAN + AYUNG RESORT + MAMAKA".

    The picture is not split with the name. It was read once for the stay and
    there is nothing in the markup saying which hotel it belongs to, so it
    stays on the first and the rest carry only what the page actually named.

    The board, the grade and the nights are split where the page did state one
    per hotel — nahalgasht prints "هتل پنجکنت پلازا 3 ستاره ... - 2 شب" under
    the hotel above it — and otherwise behave like the picture.

    Example:
        >>> [h.name for h in _hotels_named("A + B", None, 4, "BB", 6)]
        ['A', 'B']
        >>> [(h.star_rating, h.nights) for h in _hotels_named("A + B", None, 4, "BB", 6)]
        [(4, 6), (None, None)]
        >>> [h.board_type for h in _hotels_named("A + B", None, None, "BB", 6)]
        ['BB', None]
        >>> [h.board_type for h in _hotels_named("A + B + C", None, None, "BB", 6,
        ...                                      boards=("BB", "BB", "FB"))]
        ['BB', 'BB', 'FB']
        >>> [(h.star_rating, h.nights) for h in _hotels_named("A + B", None, 4, None, 6,
        ...                                 stars=(4, 3), each_nights=(3, 2))]
        [(4, 3), (3, 2)]
        >>> [h.city for h in _hotels_named("A + B", None, None, None, 6,
        ...                                cities=("Dushanbe", "Panjakent"))]
        ['Dushanbe', 'Panjakent']
        >>> [h.name for h in _hotels_named("Kemer Star Hotel", None, None, None, 3)]
        ['Kemer Star Hotel']
        >>> _hotels_named("A + B", None, None, None, 3,
        ...               image_url="https://x.com/a.jpg")[1].image_url is None
        True
    """
    parts = [part.strip() for part in str(name).split(" + ")] if name else []
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return [Hotel(name=name, city=city, star_rating=star_rating,
                      board_type=board_type, nights=nights,
                      image_url=image_url)]
    def each(values: Optional[Any], index: int) -> Optional[Any]:
        listed = list(values) if values else []
        return listed[index] if index < len(listed) else None

    return [Hotel(name=parts[0], city=each(cities, 0) or city,
                  star_rating=each(stars, 0) or star_rating,
                  board_type=each(boards, 0) or board_type,
                  nights=each(each_nights, 0) or nights,
                  image_url=image_url)] + [
        Hotel(name=part,
              city=each(cities, index),
              star_rating=each(stars, index),
              board_type=each(boards, index),
              nights=each(each_nights, index))
        for index, part in enumerate(parts[1:], start=1)
    ]


def _nights_of_every_hotel(each: Optional[Any]) -> Optional[int]:
    """The nights of a stay that moves city: every hotel's, added up.

    The flat ``nights`` field holds what was read once for the block, which on
    a multi-city card is the first hotel alone — nahalgasht sold "7 شب و 8 روز"
    to Kuala Lumpur and Singapore as 4 nights, because its first hotel is
    4 nights and its second is 3. Only counted where the page stated one
    against each hotel; a single figure for the stay is left as it is.

    Example:
        >>> _nights_of_every_hotel((4, 3))
        7
        >>> _nights_of_every_hotel((4,)) is None
        True
        >>> _nights_of_every_hotel(None) is None
        True
        >>> _nights_of_every_hotel((4, None))
        4
    """
    if not isinstance(each, (list, tuple)) or len(each) < 2:
        return None
    counted = [value for value in each if isinstance(value, int)]
    return sum(counted) or None


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
        (['double', 'single'], 82000000)
    """
    first = rows[0].fields

    departure = _tour_date(_value(first, "departure_date"))
    ret = _tour_date(_value(first, "return_date"))
    nights = (
        _nights_of_every_hotel(_value(first, "hotel_nights"))
        or _value(first, "nights")
        or nights_from_name
        or _nights_between(departure, ret)
    )
    if not ret.iso and not ret.raw:
        ret = _return_from_nights(departure, nights) or ret
    elif _impossible(departure, ret):
        ret = _return_from_nights(departure, nights) or TourDate()

    stays = _value(first, "hotels")
    if isinstance(stays, list) and stays:
        # Where the source stated each hotel in full, keep them all: a stay
        # that moves city has a grade, a board and a night count per hotel,
        # and the flat fields below can only carry the first of each.
        hotels = [
            Hotel(
                name=stay.get("name"),
                name_latin=stay.get("name_latin"),
                city=stay.get("city"),
                star_rating=stay.get("star_rating"),
                board_type=stay.get("board_type"),
                nights=stay.get("nights"),
                image_url=stay.get("image_url"),
            )
            for stay in stays
            if isinstance(stay, dict)
        ]
    else:
        hotels = _hotels_named(
            _value(first, "hotel_name"),
            city=_value(first, "hotel_city"),
            star_rating=_value(first, "hotel_star_rating"),
            board_type=_value(first, "board_type"),
            nights=nights,
            image_url=_value(first, "hotel_image"),
            boards=_value(first, "board_types"),
            stars=_value(first, "hotel_stars"),
            each_nights=_value(first, "hotel_nights"),
            cities=_value(first, "hotel_cities"),
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
        hotels=hotels,
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


def _calendar_offer(entry) -> TourOffer:
    """One departure a page stated as data, as the listing needs to see it.

    The price goes in under a double room because that is the room a tour is
    quoted from, and because the listing reads only the adult rooms. Nothing
    else is known about it: a calendar names no hotel, so this must never
    reach the package structure, which is one row per hotel and per room.

    Example:
        >>> from crawler.extraction.structure.date_calendar import Departure
        >>> offer = _calendar_offer(Departure("1405-06-30", 45_000_000.0, "IRT",
        ...                                   255.0, "EUR"))
        >>> offer.departure.iso, offer.prices["double"].local.amount
        ('2026-09-21', 45000000.0)
        >>> offer.hotels
        []
    """
    year, month, day = (int(part) for part in entry.jalali.split("-"))
    gregorian = jalali_to_gregorian(year, month, day)
    return TourOffer(
        departure=TourDate(
            raw=entry.jalali,
            iso=date(*gregorian).isoformat(),
            jalali=entry.jalali,
        ),
        prices={ROOM_DOUBLE: Price(
            local=(Money(amount=entry.local, currency=entry.local_currency)
                   if entry.local and entry.local_currency else None),
            foreign=(Money(amount=entry.foreign, currency=entry.foreign_currency)
                     if entry.foreign and entry.foreign_currency else None),
            raw=entry.jalali,
        )},
    )


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
        family=page.family,
        tour_name=tour_name,
        origin=_value(page.page_fields, "origin"),
        origin_latin=_value(page.page_fields, "origin_latin"),
        destination=_value(page.page_fields, "destination"),
        destinations=_value(page.page_fields, "destinations") or [],
        destination_city_latin=_value(page.page_fields, "destination_city_latin"),
        destination_country=_value(page.page_fields, "destination_country"),
        destination_country_latin=_value(
            page.page_fields, "destination_country_latin"),
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
        listing_only=[_calendar_offer(entry) for entry in page.calendar],
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
