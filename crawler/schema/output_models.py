"""Step 4: the structured output every site is converted into.

Fifty tour websites, fifty different layouts — one shape at the end. This module
is that shape. Every extractor, and the LLM, produce these objects and nothing
else.

Two findings from real target sites drove the design:

* A price is not one number. avisatravel.com sells a tour for
  "110,000,000 تومان + 225 دلار" — a local amount *plus* a foreign one. A single
  ``adult_price`` field throws the dollars away, and two rooms that differ only
  in the dollar part collapse into duplicates.
* A tour page is not one record. nahalgasht.com puts 4 departure dates x 7
  hotels x 4 room types on one page: 112 bookable combinations. Keeping one row
  per page keeps 1 and loses 111.

So a :class:`Tour` holds page-level facts (name, route, agency) plus a list of
:class:`TourOffer` rows, each one thing a customer can actually book.
:meth:`Tour.to_rows` flattens that into the flat records the search engine wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

# Toman and Rial are both printed on Persian sites and differ by a factor of 10,
# so the unit is always stored, never assumed.
CURRENCY_TOMAN = "IRT"
CURRENCY_RIAL = "IRR"
CURRENCY_USD = "USD"
CURRENCY_EUR = "EUR"
CURRENCY_AED = "AED"

# What one price covers. Persian sites mark per-person prices "(هرنفر)"; others
# quote a room total. Treating them as the same is a silent 2x error.
BASIS_PER_PERSON = "per_person"
BASIS_TOTAL = "total"

# Who the price is for. Both target sites price by room occupancy, not by a
# plain "adult / child / infant" split, and single vs double is a real price
# difference on the same hotel — so occupancy is its own dimension.
ROOM_DOUBLE = "double"                  # دو تخته / 2 تخته
ROOM_SINGLE = "single"                  # یک تخته / 1 تخته
ROOM_TRIPLE = "triple"                  # سه تخته
ROOM_CHILD_WITH_BED = "child_with_bed"  # کودک با تخت
ROOM_CHILD_NO_BED = "child_no_bed"      # کودک بدون تخت
ROOM_INFANT = "infant"                  # نوزاد

ROOM_TYPES = (
    ROOM_DOUBLE,
    ROOM_SINGLE,
    ROOM_TRIPLE,
    ROOM_CHILD_WITH_BED,
    ROOM_CHILD_NO_BED,
    ROOM_INFANT,
)

# How the traveller gets there. Kept coarse on purpose: this is the field the
# search engine filters on, while airline and flight details live on the offer.
TRANSPORT_AIR = "air"       # هوایی
TRANSPORT_BUS = "bus"       # زمینی / اتوبوس
TRANSPORT_TRAIN = "train"   # قطار
TRANSPORT_SHIP = "ship"     # کشتی


# --------------------------------------------------------------------------- #
# Small value objects
# --------------------------------------------------------------------------- #

@dataclass
class Money:
    """One amount in one currency."""

    amount: float
    currency: str  # one of the CURRENCY_* constants


@dataclass
class Price:
    """What a customer pays, which can be split across two currencies.

    Persian agencies commonly quote "local part + foreign part" for outbound
    tours: the Toman amount covers flight and agency fees, the dollar amount
    covers the hotel abroad. Both parts belong to the *same* price, so they are
    two fields of one object rather than two separate prices.
    """

    local: Optional[Money] = None    # e.g. 110_000_000 IRT
    foreign: Optional[Money] = None  # e.g. 225 USD, paid on top of ``local``
    basis: str = BASIS_PER_PERSON
    raw: Optional[str] = None        # text as printed, kept for traceability

    @property
    def is_empty(self) -> bool:
        return self.local is None and self.foreign is None


@dataclass
class TourDate:
    """A date as the site printed it, plus the machine-readable version.

    Persian sites print Jalali dates ("24 مرداد"), sometimes with Persian digits
    and often without a year. ``raw`` is kept exactly as found so a bad calendar
    conversion can always be traced back and re-done without re-crawling.
    """

    raw: Optional[str] = None   # "24 مرداد" / "۱۴۰۵/۰۵/۲۴"
    iso: Optional[str] = None   # "2026-08-15" after Jalali -> Gregorian
    time: Optional[str] = None  # "22:30" if a departure/arrival time was shown


@dataclass 
class Hotel:
    """One hotel option inside an offer."""

    name: Optional[str] = None         # "هتل کورت"
    name_latin: Optional[str] = None   # "asto" when the site prints both
    city: Optional[str] = None         # "کوالالامپور"
    star_rating: Optional[int] = None  # 3 from "3 ستاره"
    board_type: Optional[str] = None   # "BB", "HB", "ALL" — from "خدمات:BB"
    nights: Optional[int] = None       # "7 شب" — can differ per hotel


@dataclass
class Transport:
    """Flight (or other vehicle) details for one departure."""

    airline: Optional[str] = None            # "ایران ایرتور"
    origin_code: Optional[str] = None        # "IKA"
    destination_code: Optional[str] = None   # "KUL"
    cabin_class: Optional[str] = None        # "Economy"
    duration: Optional[str] = None           # "08:00" — مدت سفر
    flight_number: Optional[str] = None


@dataclass
class Agency:
    """Who is selling the tour. Same for every offer on the site."""

    name: Optional[str] = None
    website_url: Optional[str] = None
    logo_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# The bookable row
# --------------------------------------------------------------------------- #

@dataclass
class TourOffer:
    """One thing a customer can actually book: a date plus a hotel.

    This is the real grain of the data. Prices hang off it keyed by room type,
    because one date-plus-hotel row carries several prices at once (double,
    single, child with bed, ...) — exactly how both target sites lay them out.
    """

    departure: TourDate = field(default_factory=TourDate)
    ret: TourDate = field(default_factory=TourDate)
    nights: Optional[int] = None
    hotel: Optional[Hotel] = None
    transport: Optional[Transport] = None

    # Keyed by a ROOM_* value. A dict rather than fixed fields so an unusual
    # room type ("سه تخته") is captured instead of dropped.
    prices: dict[str, Price] = field(default_factory=dict)

    # Ties this row back to where it came from — e.g. nahalgasht's
    # ``hotel_id="1-3798"``, which joins its price JSON to its hotel markup.
    source_key: Optional[str] = None

    # --- convenience views onto the required output fields ------------------ #

    @property
    def adult_price(self) -> Optional[Price]:
        """The headline adult price.

        Double occupancy is what these sites advertise as *the* price, so it is
        the adult price. Single is a supplement on the same hotel, not a
        different traveller type, and stays available under ``prices``.
        """
        return self.prices.get(ROOM_DOUBLE) or self.prices.get(ROOM_SINGLE)

    @property
    def child_price(self) -> Optional[Price]:
        return self.prices.get(ROOM_CHILD_WITH_BED) or self.prices.get(ROOM_CHILD_NO_BED)

    @property
    def infant_price(self) -> Optional[Price]:
        return self.prices.get(ROOM_INFANT)


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

@dataclass
class Provenance:
    """How this record was produced, so a wrong value can be traced.

    Without this there is no way to tell a price the site published from one the
    LLM guessed, and no way to answer "why did this field change today?".
    """

    source_url: Optional[str] = None
    fetched_at: Optional[str] = None    # ISO timestamp, set by the caller
    content_hash: Optional[str] = None  # skip unchanged pages on a daily refresh
    # Which method filled each field: {"adult_price": "embedded_json", ...}
    field_methods: dict[str, str] = field(default_factory=dict)
    llm_used: bool = False


def _price_columns(prefix: str, price: Optional[Price]) -> dict:
    """Spread one Price across flat columns without losing the foreign part."""
    if price is None:
        return {
            f"{prefix}_price": None,
            f"{prefix}_price_currency": None,
            f"{prefix}_price_foreign": None,
            f"{prefix}_price_foreign_currency": None,
            f"{prefix}_price_basis": None,
        }
    return {
        f"{prefix}_price": price.local.amount if price.local else None,
        f"{prefix}_price_currency": price.local.currency if price.local else None,
        f"{prefix}_price_foreign": price.foreign.amount if price.foreign else None,
        f"{prefix}_price_foreign_currency": price.foreign.currency if price.foreign else None,
        f"{prefix}_price_basis": price.basis,
    }
    
@dataclass
class Tour:
    """One tour page, with every bookable offer found on it."""

    tour_url: str
    tour_name: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    transportation_type: Optional[str] = None  # a TRANSPORT_* value
    image_url: Optional[str] = None
    is_installment_available: Optional[bool] = None  # اقساط

    agency: Agency = field(default_factory=Agency)
    offers: list[TourOffer] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_rows(self) -> list[dict]:
        """Flatten to one flat record per offer, using the agreed field names.

        The search engine wants flat rows, so the required 18 fields keep their
        exact names here. Prices are split into ``*_price`` (the local number,
        which is what a search engine sorts and filters on) plus explicit
        currency and foreign-part columns, so the dual-currency case survives
        flattening instead of being rounded down to its Toman half.
        """
        rows: list[dict] = []
        for offer in self.offers:
            hotel = offer.hotel or Hotel()
            row: dict = {
                "tour_name": self.tour_name,
                "origin": self.origin,
                "destination": self.destination,
                "transportation_type": self.transportation_type,
                "departure_date": offer.departure.iso or offer.departure.raw,
                "return_date": offer.ret.iso or offer.ret.raw,
                "number_of_nights": offer.nights if offer.nights is not None else hotel.nights,
                # Singular at this grain: one row is one hotel on one date.
                "hotels": hotel.name,
                "hotel_star_rating": hotel.star_rating,
                "image_url": self.image_url,
                "tour_url": self.tour_url,
                "agency_name": self.agency.name,
                "website_url": self.agency.website_url,
                "logo_url": self.agency.logo_url,
                "is_installment_available": self.is_installment_available,
                # Extra context that would otherwise be lost in the flattening.
                "board_type": hotel.board_type,
                "hotel_city": hotel.city,
                "departure_time": offer.departure.time,
                "return_time": offer.ret.time,
                "airline": (offer.transport.airline if offer.transport else None),
            }
            row.update(_price_columns("adult", offer.adult_price))
            row.update(_price_columns("child", offer.child_price))
            row.update(_price_columns("infant", offer.infant_price))
            # Single occupancy is a genuinely different price, not a duplicate.
            row.update(_price_columns("single", offer.prices.get(ROOM_SINGLE)))
            rows.append(row)
        return rows


