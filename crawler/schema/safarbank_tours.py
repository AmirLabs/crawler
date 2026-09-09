"""The client's second JSON: one object per tour on one departure date.

Their first structure — :mod:`crawler.schema.safarbank` — is one object per
bookable **package**: a date, the hotels it sleeps in, and a price for every
room type. This one is a listing. It says a tour leaves on a date, goes to a
place, and starts from a price, and it names no hotel and no room at all.

Their own sample is ``data/second_structure_safarbank.json``. Reading it says
three things that the field names alone do not:

* **The grain is tour × departure.** Twenty-two objects carry eight distinct
  ``tour_name``s: one tour appears once for each date it leaves on. ``id_same``
  is their own number for the tour and ``id`` is the departure; their developer
  asked for ``id_same`` empty, since a number of ours means nothing on their
  side, so ``api_id`` carries the empty middle they expect.
* **``api_id`` is built, not stored.** It is ``id-id_same-startdate`` with the
  date packed, on 22 of 22 rows.
* **A night is a night.** ``end_date`` minus ``start_date`` equals ``night`` on
  every row, and ``day`` is ``night`` plus one. Their own data settles the
  question issue #2 asked of ours.

Everything is a string except ``code``, the price inside ``start_price``, and
the ``coefficient``-style numbers they do not use here. A field we cannot read
is written as an empty string rather than left out: their sample does the same
— ``origin_country_name_en`` is empty on all 22 — and a key that comes and goes
is harder to consume than one that is reliably blank.
"""

from __future__ import annotations

from typing import Optional

from crawler.extraction.readers.places import is_a_departure_city
from crawler.schema.identifiers import PROVIDER_ID, IdStore
from crawler.schema.safarbank import CURRENCY_NAMES, packed_date, to_jalali_parts
from crawler.schema.output_models import (
    ROOM_DOUBLE,
    ROOM_SINGLE,
    ROOM_TRIPLE,
    Price,
    Tour,
    TourOffer,
)

# What they call the way a tour travels, against the value step 3 reads. Their
# own sample uses only the first two; the third is here because our transport
# reader can return it and writing "هواپیما" for a train would be a lie.
VEHICLE_NAMES: dict[str, str] = {
    "air": "هواپیما",
    "bus": "اتوبوس",
    "train": "قطار",
}

# The same fact under their other name. ``tour_type`` tags a tour with the
# adjective — "هوایی" — while ``type_vehicle_name`` names the vehicle itself,
# "هواپیما". Two vocabularies for one thing, and their sample keeps them apart
# on every row.
VEHICLE_TAGS: dict[str, str] = {
    "air": "هوایی",
    "bus": "زمینی",
    "train": "زمینی",
}

# The tags they hang on a tour. Their vocabulary is wider than this — it also
# holds "پرطرفدار" and "تور اقساطی", which are the agency's own judgement and
# not ours to make — so only the three that follow from what we read are set.
# The ids are theirs and we do not have the table, so they stay empty.
TOUR_TYPE_FOREIGN = "تور خارجی"
TOUR_TYPE_DOMESTIC = "تور داخلی"
TOUR_TYPE_MULTI_DAY = "تور چند روزه"

# Their numbers for the tags, confirmed by their developer. Read off their own
# sample, where each id carried one name on all 22 rows and never a second.
# The four we do not set are here so the table stays whole: three are the
# agency's own judgement, and "آسیایی" wants the continent over the country,
# which the sites do not file by.
TOUR_TYPE_IDS: dict[str, str] = {
    TOUR_TYPE_MULTI_DAY: "2",
    "زمینی": "3",
    "پرطرفدار": "4",
    "هوایی": "5",
    "آسیایی": "8",
    TOUR_TYPE_DOMESTIC: "9",
    "راهنما": "12",
    TOUR_TYPE_FOREIGN: "16",
    "تور اقساطی": "17",
}

# Iran, in their country table. The only id of theirs we can read off their
# sample with any confidence: origin_country_id is "1" on all 22 rows and every
# one of them leaves from Tehran.
IRAN_COUNTRY_ID = "1"

# The rooms an adult can be sold. A child's or an infant's fare is cheaper and
# is not what a tour starts from.
ADULT_ROOMS: frozenset[str] = frozenset({ROOM_DOUBLE, ROOM_SINGLE, ROOM_TRIPLE})


def _price_parts(offer: TourOffer) -> tuple[int, str, str]:
    """The price a tour starts from, as their three ``start_price`` fields.

    Their ``price`` is the local amount as a number, ``price_currency`` the
    foreign half as a string, and ``Currency_title`` the foreign currency's
    Persian name — which their sample fills even where the foreign half is
    "0.00", so it is filled here the same way.

    "Starts from" means the cheapest bed an adult can sleep in, not the
    cheapest number on the page. Read the other way a chakavaktravel package
    started from 17,000,000, which is its infant fare: a child's price is not a
    price the tour can be bought at.

    Example:
        >>> from crawler.schema.output_models import Money, Price
        >>> offer = TourOffer(prices={"double": Price(
        ...     local=Money(amount=56790000.0, currency="IRT"),
        ...     foreign=Money(amount=225.0, currency="USD"))})
        >>> _price_parts(offer)
        (56790000, '225.00', 'دلار')
        >>> _price_parts(TourOffer())
        (0, '0.00', '')
        >>> cheap = Price(local=Money(amount=17000000.0, currency="IRT"))
        >>> _price_parts(TourOffer(prices={"infant": cheap}))
        (0, '0.00', '')
    """
    cheapest: Optional[Price] = None
    for room, price in offer.prices.items():
        if room not in ADULT_ROOMS:
            continue
        if price.local is None or price.local.amount is None:
            continue
        if cheapest is None or price.local.amount < cheapest.local.amount:
            cheapest = price
    if cheapest is None:
        return 0, "0.00", ""

    foreign = cheapest.foreign
    amount = f"{foreign.amount:.2f}" if foreign and foreign.amount else "0.00"
    name = CURRENCY_NAMES.get(foreign.currency, "") if foreign else ""
    return int(cheapest.local.amount), amount, name


def _tour_types(tour: Tour, offer: TourOffer) -> list[dict]:
    """The tags that follow from what was read, and no others.

    An empty list where nothing could be told. Their sample never shows that
    case — all 22 rows carry at least two tags — so this is their developer's
    word, asked for by phone, and not a shape read off the data.

    Example:
        >>> tour = Tour(tour_url="u", origin="تهران", destination="پاتایا",
        ...             transportation_type="air")
        >>> _tour_types(tour, TourOffer(nights=7))
        [{'id': '5', 'name': 'هوایی'}, {'id': '16', 'name': 'تور خارجی'},
         {'id': '2', 'name': 'تور چند روزه'}]
        >>> tour.transportation_type = None
        >>> [t["name"] for t in _tour_types(tour, TourOffer(nights=1))]
        ['تور خارجی']
        >>> _tour_types(Tour(tour_url="u"), TourOffer())
        []
    """
    names: list[str] = []
    tag = VEHICLE_TAGS.get(tour.transportation_type or "")
    if tag:
        names.append(tag)
    if tour.origin and tour.destination:
        names.append(
            TOUR_TYPE_DOMESTIC if is_a_departure_city(tour.destination)
            else TOUR_TYPE_FOREIGN
        )
    if offer.nights and offer.nights > 1:
        names.append(TOUR_TYPE_MULTI_DAY)
    return [{"id": TOUR_TYPE_IDS.get(name, ""), "name": name} for name in names]


def _leg(
    tour: Tour, offer: TourOffer, going: bool, number: int, row: str, leg_id: str
) -> dict:
    """One half of the journey, in the shape their ``info_tour_rout`` uses.

    ``fk_tour_id`` points at the departure this leg belongs to — their own data
    says so on 46 legs out of 46 — so it carries our ``id`` for that departure.
    The number is ours and means nothing to them, but the link does: they load
    departures and routes into two tables, and a leg without it is an orphan
    that nobody can tell the outward flight of one date from another by.

    The rest of the ids are theirs, out of tables we do not have, and stay
    empty. The night count sits on the outward leg, where their sample puts it.

    Example:
        >>> tour = Tour(tour_url="u", destination="پاتایا", transportation_type="air")
        >>> leg = _leg(tour, TourOffer(nights=7), True, 1, row="4", leg_id="9")
        >>> leg["tour_title"], leg["night"], leg["destination_city_name"]
        ('dept', '7', 'پاتایا')
        >>> leg["id"], leg["fk_tour_id"]
        ('9', '4')
    """
    when = offer.departure if going else offer.ret
    return {
        "id": leg_id,
        "fk_tour_id": row,
        "tour_title": "dept" if going else "return",
        "number_rout": str(number),
        "night": str(offer.nights) if going and offer.nights else "0",
        "day": "0",
        "destination_continent_id": "",
        "destination_country_id": "",
        "destination_country_name": tour.destination_country or "",
        "destination_city_id": "",
        "destination_city_name": tour.destination or "",
        "destination_region_id": "",
        "destination_region_name": "",
        "type_vehicle_id": "",
        "type_vehicle_name": VEHICLE_NAMES.get(tour.transportation_type or "", ""),
        "airline_id": "",
        "airline_name": (offer.transport.name if offer.transport else "") or "",
        "class": "",
        "exit_hours": when.time or "",
        "hours": "",
        "is_del": "no",
        "is_route_fake": "",
        "name": tour.destination or "",
        "name_en": "",
        "country_name": tour.destination_country or "",
        "country_name_en": tour.destination_country_latin or "",
    }


def offer_to_listing(tour: Tour, offer: TourOffer, ids: IdStore) -> dict:
    """One departure of one tour, as one of their listing objects.

    Example:
        >>> from crawler.schema.output_models import Agency, TourDate
        >>> tour = Tour(tour_url="https://x.com/tour/1", tour_name="تور پاتایا",
        ...             origin="تهران", destination="پاتایا",
        ...             transportation_type="air",
        ...             agency=Agency(name="پا به پا سفر"))
        >>> offer = TourOffer(nights=7,
        ...     departure=TourDate(iso="2026-09-12", jalali="1405-06-21", time="22:45"),
        ...     ret=TourDate(iso="2026-09-19", jalali="1405-06-28"))
        >>> out = offer_to_listing(tour, offer, IdStore(path=None))
        >>> out["api_id"], out["night"], out["day"]
        ('1-31540001-14050621', '7', '8')
        >>> out["start_date"], out["end_date"], out["exit_hours"]
        ('1405-06-21', '1405-06-28', '22:45')
    """
    same = ids.tour_code(tour.tour_url)
    packed = packed_date(to_jalali_parts(offer.departure.iso))
    row = ids.fk_tour_id(same, packed)
    price, foreign, currency = _price_parts(offer)
    vehicle = VEHICLE_NAMES.get(tour.transportation_type or "", "")
    airline = (offer.transport.name if offer.transport else "") or ""

    return {
        "api_id": f"{row}-{same}-{packed}",
        "provider_id": str(PROVIDER_ID),
        "agency_name": tour.agency.name or "",
        "id": row,
        "id_same": "",
        "tour_name": tour.tour_name or "",
        "tour_name_en": "",
        "tour_detail_address": tour.tour_url,
        "tour_type": _tour_types(tour, offer),
        "tour_code": same,
        "start_date": offer.departure.jalali or "",
        "end_date": offer.ret.jalali or "",
        "night": str(offer.nights) if offer.nights else "",
        "day": str(offer.nights + 1) if offer.nights else "",
        "tour_pic": tour.image_url or "",
        "origin_country_id": IRAN_COUNTRY_ID if tour.origin else "",
        "is_special": "",
        "is_del": "no",
        "is_show": "yes",
        "tour_status": "",
        "origin_city_name": tour.origin or "",
        "origin_city_name_en": tour.origin_latin or "",
        "origin_city_id": "",
        "origin_region_name": "",
        "destination_country_id": "",
        "destination_country_name": tour.destination_country or "",
        "destination_country_name_en": tour.destination_country_latin or "",
        "origin_country_name_en": "",
        "destination_city_name": tour.destination or "",
        "destination_city_name_en": "",
        "destination_region_name": "",
        "vehicle_name": airline,
        "type_vehicle_name": vehicle,
        "exit_hours": offer.departure.time or "",
        "vehicle_id": "",
        "type_vehicle_id": "",
        "start_price": {
            "price": price,
            "price_currency": foreign,
            "Currency_title": currency,
        },
        "vehicles": {
            "dept": {
                "type_vehicle_name": vehicle,
                "airline_name": airline,
                "tour_title": "رفت",
            },
            "return": {
                "type_vehicle_name": vehicle,
                "airline_name": airline,
                "tour_title": "برگشت",
            },
        },
        "info_tour_rout": [
            _leg(tour, offer, True, 1, row, ids.row_id()),
            _leg(tour, offer, False, 2, row, ids.row_id()),
        ],
        "agency_info": {
            "AgencyName": tour.agency.name or "",
            "Address": "",
            "Phone": "",
            "Mobile": "",
            "Logo": tour.agency.logo_url or "",
        },
    }


def tour_to_listings(tour: Tour, ids: IdStore) -> list[dict]:
    """Every departure of one tour, once each.

    A tour's offers repeat per hotel and per room, and this structure has no
    room in it, so the departures are folded: the cheapest price across every
    package that leaves on a date is what that date starts from.

    Example:
        >>> from crawler.schema.output_models import Money, Price, TourDate
        >>> tour = Tour(tour_url="https://x.com/tour/1")
        >>> def offer(day, amount):
        ...     return TourOffer(
        ...         departure=TourDate(iso=f"2026-09-{day:02d}", jalali=f"1405-06-{day-3:02d}"),
        ...         prices={"double": Price(local=Money(amount=amount, currency="IRT"))})
        >>> tour.offers = [offer(12, 90_000_000), offer(12, 70_000_000), offer(13, 80_000_000)]
        >>> [(o["start_date"], o["start_price"]["price"]) for o in tour_to_listings(tour, IdStore(path=None))]
        [('1405-06-09', 70000000), ('1405-06-10', 80000000)]
    """
    cheapest: dict[str, TourOffer] = {}
    for offer in tour.offers:
        key = packed_date(to_jalali_parts(offer.departure.iso))
        best = cheapest.get(key)
        if best is None or _price_parts(offer)[0] < _price_parts(best)[0]:
            cheapest[key] = offer
    return [offer_to_listing(tour, offer, ids)
            for _, offer in sorted(cheapest.items())]


def listing_document(objects: list[dict]) -> dict:
    """The wrapper their sample puts around the list.

    Example:
        >>> listing_document([])
        {'status': 'success', 'code': 200, 'message': 'Data Fetched Successfully', 'data': []}
    """
    return {
        "status": "success",
        "code": 200,
        "message": "Data Fetched Successfully",
        "data": objects,
    }
