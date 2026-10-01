"""Turning a read tour into the JSON the client asked for.

The client sends one object per bookable package: a tour on one departure date,
the hotels it sleeps in, and the price of every room type. Their own sample is
in ``data/safarbank.json`` and this writes the same 62 keys, in the same types
— everything a string except ``provider_id`` and the four numbers inside
``rooms``.

Two of their conventions are easy to get backwards and expensive when you do:

* ``_r`` is the **rial** half of a price and ``_a`` the foreign half. The
  ``rooms`` list carries the same rial figure: every amount we send is in
  rial, and :func:`rial_of` is the one place toman becomes it.
* ``infant_without_bed`` is named "کودک بدون تخت" in their own ``rooms`` list,
  and ``infant_without_chair`` is "نوزاد". Trusting the key names swaps a child
  price for an infant one.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from crawler.extraction.readers.dates import gregorian_to_jalali, jalali_to_gregorian
from crawler.api.queries import as_packed
from crawler.schema.identifiers import PROVIDER_ID, IdStore
from crawler.schema.output_models import (
    CURRENCY_TOMAN,
    ROOM_CHILD_NO_BED,
    ROOM_CHILD_WITH_BED,
    ROOM_DOUBLE,
    ROOM_INFANT,
    ROOM_SINGLE,
    ROOM_TRIPLE,
    Hotel,
    Price,
    Tour,
    TourOffer,
)

MONTHS: tuple[str, ...] = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
)

# Spelled as their own sample spells them: "سه شنبه" apart, "پنجشنبه" joined.
WEEKDAYS: dict[int, str] = {
    5: "شنبه", 6: "یکشنبه", 0: "دوشنبه", 1: "سه شنبه",
    2: "چهارشنبه", 3: "پنجشنبه", 4: "جمعه",
}

CURRENCY_NAMES: dict[str, str] = {
    "USD": "دلار", "EUR": "یورو", "AED": "درهم", "TRY": "لیر", "GBP": "پوند",
}

# Their room list, in their order. The Persian name, the machine key, the
# display order and the per-person coefficient are all theirs; the ROOM_* value
# is ours. Order skips 4 to 6, which their system holds for rooms these tours
# do not sell.
ROOMS: tuple[tuple[str, str, str, int, int, str], ...] = (
    ("single_room",           ROOM_SINGLE,         "یک تخته",       1, 1, "oneBed"),
    ("double_room",           ROOM_DOUBLE,         "دو تخته",       2, 2, "twoBed"),
    ("three_room",            ROOM_TRIPLE,         "سه تخته",       3, 3, "threeBed"),
    ("child_with_bed",        ROOM_CHILD_WITH_BED, "کودک با تخت",   7, 1, "childwithbed"),
    ("infant_without_bed",    ROOM_CHILD_NO_BED,   "کودک بدون تخت", 8, 1, "babywithoutbed"),
    ("infant_without_chair",  ROOM_INFANT,         "نوزاد",         9, 1, "babywithoutchair"),
)

# The flat price and capacity keys come in a different order from the rooms
# list, so their sample can be laid beside ours and read straight down.
PRICE_ORDER: tuple[str, ...] = (
    "three_room", "double_room", "single_room",
    "child_with_bed", "infant_without_bed", "infant_without_chair",
)

ROOM_TYPE: dict[str, str] = {
    "oneBed": "adult", "twoBed": "adult", "threeBed": "adult",
    "childwithbed": "child", "babywithoutbed": "infant",
    "babywithoutchair": "infant",
}

# Values their sample carries on every row and we have no source for. Copied as
# they send them rather than left empty, which is Amir's decision.
DEFAULT_CAPACITY = "9"
DEFAULT_PREPAYMENT = "50"
RIAL_PER_TOMAN = 10


def to_jalali_parts(iso: Optional[str]) -> Optional[tuple[int, int, int]]:
    """The Jalali year, month and day behind a Gregorian ISO date.

    Example:
        >>> to_jalali_parts("2026-08-26")
        (1405, 6, 4)
        >>> to_jalali_parts(None) is None
        True
    """
    if not iso:
        return None
    try:
        year, month, day = (int(part) for part in iso.split("-"))
    except ValueError:
        return None
    return gregorian_to_jalali(year, month, day)


def packed_date(parts: Optional[tuple[int, int, int]]) -> str:
    """Their eight-digit departure date, or empty where we have none.

    Example:
        >>> packed_date((1405, 6, 4))
        '14050604'
        >>> packed_date(None)
        ''
    """
    if parts is None:
        return ""
    return f"{parts[0]:04d}{parts[1]:02d}{parts[2]:02d}"


def human_date(parts: Optional[tuple[int, int, int]]) -> str:
    """Their readable date. The day carries no leading zero, as they write it.

    Example:
        >>> human_date((1405, 6, 4))
        '4/شهریور/1405'
        >>> human_date(None)
        ''
    """
    if parts is None:
        return ""
    return f"{parts[2]}/{MONTHS[parts[1] - 1]}/{parts[0]}"


def weekday(parts: Optional[tuple[int, int, int]]) -> str:
    """The Persian day of the week, read from the Gregorian date behind it.

    Example:
        >>> weekday((1405, 6, 4))
        'چهارشنبه'
        >>> weekday((1405, 6, 10))
        'سه شنبه'
    """
    if parts is None:
        return ""
    return WEEKDAYS[date(*jalali_to_gregorian(*parts)).weekday()]


def rial_of(price: Optional[Price]) -> int:
    """The local half of a price in rial, or zero where there is none.

    The client asked for rial and these sites all quote toman, so every figure
    we send them is converted. It happens here and nowhere else: the conversion
    used to live inside :func:`price_halves`, and the two surfaces added later —
    the ``rooms`` list and the second structure's ``start_price`` — were each
    written without it and each sent toman for months.

    A price quoted in anything but toman is **not** converted. It is the
    foreign half of the price, and multiplying 590 dollars by ten is a worse
    answer than the 590 it started as.

    Example:
        >>> from crawler.schema.output_models import Money
        >>> rial_of(Price(local=Money(92_000_000, "IRT")))
        920000000
        >>> rial_of(Price(local=Money(590, "USD")))
        0
        >>> rial_of(None)
        0
    """
    if price is None or price.local is None:
        return 0
    if price.local.currency != CURRENCY_TOMAN:
        return 0
    return int(round(price.local.amount * RIAL_PER_TOMAN))


def price_halves(price: Optional[Price]) -> tuple[str, str, Optional[str]]:
    """One price as their rial string, their foreign string, and its currency.

    Example:
        >>> from crawler.schema.output_models import Money
        >>> price_halves(Price(local=Money(92_000_000, "IRT"), foreign=Money(135, "USD")))
        ('920000000', '135.00', 'دلار')
        >>> price_halves(Price(local=Money(590, "USD")))
        ('0', '590.00', 'دلار')
        >>> price_halves(None)
        ('0', '0.00', None)
    """
    if price is None:
        return "0", "0.00", None
    rial = str(rial_of(price))

    foreign, name = "0.00", None
    money = price.foreign or (
        price.local if price.local and price.local.currency != CURRENCY_TOMAN else None
    )
    if money is not None:
        foreign = f"{money.amount:.2f}"
        name = CURRENCY_NAMES.get(money.currency, money.currency)
    return rial, foreign, name


def _hotel(hotel: Hotel, ids: IdStore, site: str) -> dict:
    """One hotel as they send it.

    Example:
        >>> out = _hotel(Hotel(name="The Sukosol", city="بانکوک", star_rating=5,
        ...                    nights=2, image_url="https://x.com/a.webp"),
        ...              IdStore(path=None), "tahagasht")
        >>> out["night"], out["logo"]
        ('2', 'https://x.com/a.webp')
        >>> _hotel(Hotel(name="The Sukosol"), IdStore(path=None), "tahagasht")["logo"]
        ''
    """
    return {
        "id": ids.hotel_id(site, hotel.name),
        "name": hotel.name or "",
        "name_en": hotel.name_latin or "",
        "trip_advisor": "0",
        "star_code": str(hotel.star_rating) if hotel.star_rating else "",
        # Their field for the hotel's one picture. Named logo, but it is the
        # photograph of the building, which is what they asked us to send.
        "logo": hotel.image_url or "",
        "region": "0",
        "type_code": "1",
        "city_name": hotel.city or "",
        "city_name_en": "",
        "facilities": [],
        "gallery": [],
        "night": str(hotel.nights) if hotel.nights else "",
        "room_service": hotel.board_type or "",
    }


def offer_to_object(
    tour: Tour, offer: TourOffer, package: int, ids: IdStore, site: str
) -> dict:
    """One departure of one tour as one of their objects.

    Example:
        >>> tour = Tour(tour_url="https://x.com/tour/1", tour_name="تور دبی")
        >>> offer = TourOffer(hotels=[Hotel(name="Sukosol", nights=3)])
        >>> out = offer_to_object(tour, offer, 1, IdStore(path=None), "x")
        >>> out["tour_code"], out["provider_id"], out["number_package"]
        ('31540001', 110110110, '1')
    """
    month = "" if offer.departure.iso else (offer.departure.jalali or "")
    tour_code = ids.tour_code(tour.tour_url, month, tour.family)
    start = to_jalali_parts(offer.departure.iso)
    end = to_jalali_parts(offer.ret.iso)
    packed = packed_date(start) or as_packed(month) or ""

    fk = ids.fk_tour_id(tour_code, packed)
    out: dict = {
        "tour_code": tour_code,
        "start_date": packed,
        "change_price": "0",
        "discount_type": "",
        "discount": "",
        "prepayment_percentage": DEFAULT_PREPAYMENT,
        "adult_price_one_day_tour_r": "0",
        "child_price_one_day_tour_r": "0",
        "infant_price_one_day_tour_r": "0",
        "id": ids.row_id(f"package|{fk}|{package}"),
        "fk_tour_id": fk,
    }

    by_prefix = {prefix: room for prefix, room, _, _, _, _ in ROOMS}
    currency: Optional[str] = None
    for prefix in PRICE_ORDER:
        rial, foreign, name = price_halves(offer.prices.get(by_prefix[prefix]))
        out[f"{prefix}_price_r"] = rial
        out[f"{prefix}_price_a"] = foreign
        currency = currency or name
    out["currency_type"] = currency or ""

    for prefix in PRICE_ORDER:
        out[f"{prefix}_capacity"] = DEFAULT_CAPACITY

    out["number_package"] = str(package)
    out["custom_room"] = "[]"
    out["is_del"] = "no"
    out["hotels"] = [_hotel(h, ids, site) for h in offer.hotels]
    out["rooms"] = [
        {
            "name": persian,
            "price": rial_of(offer.prices.get(room)),
            "currency_price": price_halves(offer.prices.get(room))[1].rstrip("0").rstrip(".")
            if price_halves(offer.prices.get(room))[1] != "0.00" else "",
            "currency_name": currency or "",
            "coefficient": coefficient,
            "index": index,
            "order": order,
            "type": ROOM_TYPE[index],
            "final_price": rial_of(offer.prices.get(room)),
            "capacity": DEFAULT_CAPACITY,
        }
        for _, room, persian, order, coefficient, index in ROOMS
    ]
    out["start_date_human_string"] = human_date(start)
    out["start_date_week"] = weekday(start)
    out["end_date_human_string"] = human_date(end)
    out["end_date_week"] = weekday(end)
    out["provider_id"] = PROVIDER_ID
    return out


def tour_to_objects(tour: Tour, ids: IdStore, site: str) -> list[dict]:
    """Every package of one tour, numbered from one within each departure.

    Their ``number_package`` restarts at 1 for each departure date, which is
    what both of their samples show: six packages under 4 شهریور and six more
    under 5 شهریور, each run numbered 1 to 6.

    Example:
        >>> tour = Tour(tour_url="https://x.com/tour/1")
        >>> tour.offers = [TourOffer(hotels=[Hotel(name="a")]),
        ...                TourOffer(hotels=[Hotel(name="b")])]
        >>> [o["number_package"] for o in tour_to_objects(tour, IdStore(path=None), "x")]
        ['1', '2']
    """
    out: list[dict] = []
    seen: dict[str, int] = {}
    for offer in tour.offers:
        packed = packed_date(to_jalali_parts(offer.departure.iso))
        seen[packed] = seen.get(packed, 0) + 1
        out.append(offer_to_object(tour, offer, seen[packed], ids, site))
    return out
