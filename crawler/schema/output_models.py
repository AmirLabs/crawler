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

# The rooms an adult can be sold. A child's or an infant's fare is cheaper and
# is not what a tour starts from.
ADULT_ROOMS: frozenset[str] = frozenset({ROOM_DOUBLE, ROOM_SINGLE, ROOM_TRIPLE})

# How the traveller gets there. Kept coarse on purpose: this is the field the
# search engine filters on, while the carrier name and journey details live on
# the offer. The target sites sell these three and nothing else.
TRANSPORT_AIR = "air"       # هوایی / پرواز
TRANSPORT_TRAIN = "train"   # قطار / ریلی
TRANSPORT_BUS = "bus"       # زمینی / اتوبوس

TRANSPORT_TYPES = (
    TRANSPORT_AIR,
    TRANSPORT_TRAIN,
    TRANSPORT_BUS,
)

# The words the sites actually print, mapped to the value we store. Without this
# the field holds raw site text — "هوایی" on one site, "Air" on the next — and
# the search engine cannot filter across all fifty of them.
_TRANSPORT_WORDS: dict[str, str] = {
    "هوایی": TRANSPORT_AIR,
    "هواپیما": TRANSPORT_AIR,
    "پرواز": TRANSPORT_AIR,
    "air": TRANSPORT_AIR,
    "flight": TRANSPORT_AIR,
    "قطار": TRANSPORT_TRAIN,
    "ریلی": TRANSPORT_TRAIN,
    "train": TRANSPORT_TRAIN,
    "زمینی": TRANSPORT_BUS,
    "اتوبوس": TRANSPORT_BUS,
    "bus": TRANSPORT_BUS,
}


def normalize_transport(text: Optional[str]) -> Optional[str]:
    """Turn whatever a site printed into one of the TRANSPORT_* values.

    Returns ``None`` when nothing matches, which is deliberate: an unknown value
    is better left empty than guessed. An empty field shows up in the per-field
    fill-rate check, while a wrong one hides.
    """
    if not text:
        return None
    low = text.strip().lower()
    if low in TRANSPORT_TYPES:
        return low  # already normalized
    for word, value in _TRANSPORT_WORDS.items():
        if word in low:
            return value
    return None


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

    # Whether the flight is in this figure. ``None`` means the page did not say,
    # which is not the same as "yes" — the difference decides whether two sites
    # can be compared at all. liyaseir.com quotes "3790 دلار + هزینه پرواز" on
    # every page and nahalgasht.com on nine of ten: an amount that looks far
    # cheaper than a rival's only because the flight is still to be added.
    flight_included: Optional[bool] = None

    @property
    def is_empty(self) -> bool:
        return self.local is None and self.foreign is None

    @property
    def is_complete(self) -> bool:
        """Whether this is the whole cost of travelling, as far as the page said.

        Example:
            >>> Price(local=Money(3790, "USD"), flight_included=False).is_complete
            False
            >>> Price(local=Money(3790, "USD")).is_complete
            True
        """
        return not self.is_empty and self.flight_included is not False


@dataclass
class TourDate:
    """A date as the site printed it, plus the machine-readable versions.

    Persian sites print Jalali dates ("24 مرداد"), sometimes with Persian digits
    and often without a year. ``raw`` is kept exactly as found so a bad calendar
    conversion can always be traced back and re-done without re-crawling.

    ``jalali`` is not the same thing as ``raw``: it is the tidied, complete
    Jalali date derived from ``iso``, with the year the page left out filled in
    and the time removed. Both are kept because they answer different questions
    — ``raw`` says what the page claimed, ``jalali`` says what that resolves to
    in the calendar the customer reads.

    ``iso`` stays inside the pipeline and is never shipped. It is what the date
    arithmetic runs on — the night count, and whether a departure has already
    gone — but the client reads Jalali, so ``to_rows`` sends ``jalali`` under
    the agreed ``departure_date`` and ``return_date`` names and no second pair
    of columns beside them.
    """

    raw: Optional[str] = None     # "24 مرداد" / "۱۴۰۵/۰۵/۲۴"
    iso: Optional[str] = None     # "2026-08-15" after Jalali -> Gregorian
    jalali: Optional[str] = None  # "1405-05-24", normalized back from ``iso``
    time: Optional[str] = None    # "22:30" if a departure/arrival time was shown


@dataclass 
class Hotel:
    """One hotel option inside an offer."""

    name: Optional[str] = None         # "هتل کورت"
    name_latin: Optional[str] = None   # "asto" when the site prints both
    city: Optional[str] = None         # "کوالالامپور"
    star_rating: Optional[int] = None  # 3 from "3 ستاره"
    board_type: Optional[str] = None   # "BB", "HB", "ALL" — from "خدمات:BB"
    nights: Optional[int] = None       # "7 شب" — can differ per hotel

    # One picture of the building, absolute so the client can fetch it as it
    # stands. A single image rather than a gallery: that is what was asked
    # for, and one the source ties to this hotel beats several we matched up
    # ourselves.
    image_url: Optional[str] = None


@dataclass
class Transport:
    """How one departure travels — air, train or bus, described the same way.

    Only fields that mean something for all three live here. Flight number and
    cabin class were dropped: nobody searches on them, and they made a class
    that claims to cover three transport types really model just one.

    Times are deliberately absent. A time belongs to a date, so departure and
    arrival times live on the :class:`TourDate` objects of the offer. Keeping a
    second copy here would let the two disagree, and then neither can be
    trusted. ``duration`` plus the departure time gives the arrival anyway.
    """

    # Airline, train company or bus company — "ایران ایرتور", "رجا", ...
    name: Optional[str] = None
    duration: Optional[str] = None  # "08:00" — مدت سفر

    # Airport or station codes when the site prints them. Free on some sites and
    # worth keeping: "IKA" pins the origin exactly, while the free-text
    # ``Tour.origin`` is spelled a different way on every one of the fifty
    # sites. Normally empty for bus.
    origin_code: Optional[str] = None       # "IKA"
    destination_code: Optional[str] = None  # "KUL"

    # Normally None. Set only when a page mixes transport types across its
    # departures, so that rare case does not need a schema change. The page
    # level value on :class:`Tour` is the usual home — see its docstring.
    transportation_type: Optional[str] = None


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
    """One thing a customer can actually book: a date plus where it sleeps.

    This is the real grain of the data. Prices hang off it keyed by room type,
    because one bookable option carries several prices at once (double, single,
    child with bed, ...) — exactly how the target sites lay them out.

    ``hotels`` is a list because a multi-city tour is one price covering a stay
    in each city: avisatravel.com's "تور بالی + اوبود" sleeps in two hotels for
    one number, and bisanseir.com's Bali tour in four. Half of avisatravel's
    rows and a fifth of bisanseir's are like that, so the single hotel this
    field used to hold was joining them into "ده ادن + سنس" — one string that
    no longer says which hotel is in which city, or how many nights each got.

    A single-hotel offer is a list of one. That keeps one rule for both, and
    matches the client's own JSON, which always sends an array.
    """

    departure: TourDate = field(default_factory=TourDate)
    ret: TourDate = field(default_factory=TourDate)
    nights: Optional[int] = None
    hotels: list[Hotel] = field(default_factory=list)
    transport: Optional[Transport] = None

    # Keyed by a ROOM_* value. A dict rather than fixed fields so an unusual
    # room type ("سه تخته") is captured instead of dropped.
    prices: dict[str, Price] = field(default_factory=dict)

    # Ties this row back to where it came from — e.g. nahalgasht's
    # ``hotel_id="1-3798"``, which joins its price JSON to its hotel markup.
    source_key: Optional[str] = None

    # --- convenience views onto the required output fields ------------------ #

    @property
    def hotel(self) -> Optional[Hotel]:
        """The first hotel, for the flat view that can only carry one.

        Example:
            >>> TourOffer(hotels=[Hotel(name="تکسیم استار")]).hotel.name
            'تکسیم استار'
            >>> TourOffer().hotel is None
            True
        """
        return self.hotels[0] if self.hotels else None

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
        """A child travelling with a bed of their own.

        Reading "or کودک بدون تخت" here is what this used to do, and it threw
        one of the two away on every page that priced both — bisanseir,
        alefbatour and avisatravel all do. On alefbatour they are plainly
        different products: 60,990,000 IRT + 135 USD against 63,900,000 IRT
        with no dollar part at all. Amir's decision: a column each.
        """
        return self.prices.get(ROOM_CHILD_WITH_BED)

    @property
    def triple_price(self) -> Optional[Price]:
        """Three sharing a room, which the sites price and nothing carried.

        ``سه تخته`` has been read into ``prices`` since room types were first
        detected, and had no column to land in — the same fault
        :attr:`child_no_bed_price` was given one for.
        """
        return self.prices.get(ROOM_TRIPLE)

    @property
    def child_no_bed_price(self) -> Optional[Price]:
        """A child sharing a bed, priced separately by every site that sells it."""
        return self.prices.get(ROOM_CHILD_NO_BED)

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
            f"{prefix}_price_includes_flight": None,
        }
    return {
        f"{prefix}_price": price.local.amount if price.local else None,
        f"{prefix}_price_currency": price.local.currency if price.local else None,
        f"{prefix}_price_foreign": price.foreign.amount if price.foreign else None,
        f"{prefix}_price_foreign_currency": price.foreign.currency if price.foreign else None,
        # Carried beside every amount rather than once per row: a page can quote
        # one price with the flight and another without.
        f"{prefix}_price_includes_flight": price.flight_included,
        f"{prefix}_price_basis": price.basis,
    }
    
@dataclass
class Tour:
    """One tour page, with every bookable offer found on it."""

    tour_url: str
    tour_name: Optional[str] = None
    origin: Optional[str] = None
    origin_latin: Optional[str] = None  # "tehran"

    # Where the tour goes, kept twice because the two uses pull in opposite
    # directions. About a third of the tours on the target sites visit several
    # places — "تور بالی + اوبود + سنگاپور + کوالالامپور" — and a customer
    # searching for Singapore has to find it, which a single string buried in
    # three other names does not allow. The joined form is what a result page
    # shows, since "بالی + سنگاپور" is what the agency is actually selling.
    destination: Optional[str] = None   # as written: "بالی + سنگاپور"
    destinations: list[str] = field(default_factory=list)  # ["بالی", "سنگاپور"]

    # The country those cities are in, under both names. The client's second
    # structure asks for both and no page writes either in its text — they come
    # from where the site files the tour. The latin one has no other source.
    destination_city_latin: Optional[str] = None     # "yerevan"
    destination_country: Optional[str] = None        # "ترکیه"
    destination_country_latin: Optional[str] = None  # "turkey"

    # A TRANSPORT_* value, and the home of this fact. It is one value for the
    # whole page: nahalgasht publishes it as a page level taxonomy, and a page
    # of 4 dates x 7 hotels would otherwise repeat "air" 28 times. Bus and train
    # pages often print no carrier details at all, so an offer level home would
    # leave a required output field empty on pages that plainly stated it.
    transportation_type: Optional[str] = None

    image_url: Optional[str] = None
    is_installment_available: Optional[bool] = None  # اقساط

    agency: Agency = field(default_factory=Agency)
    offers: list[TourOffer] = field(default_factory=list)

    # Departures known only as a line in a listing: a date and what it starts
    # from, with no hotel and no room behind it. A page can state these as
    # data while its rows state only one date, so they fill the gaps in the
    # listing and never become packages, which are per hotel and per room.
    listing_only: list[TourOffer] = field(default_factory=list)

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
            names = [h.name for h in offer.hotels if h.name]
            cities = [h.city for h in offer.hotels if h.city]
            row: dict = {
                "tour_name": self.tour_name,
                "origin": self.origin,
                "origin_latin": self.origin_latin,
                "destination": self.destination,
                # The searchable form. Kept a list rather than joined, so a
                # multi-city tour can be found by any one of its stops.
                "destinations": list(self.destinations),
                "destination_city_latin": self.destination_city_latin,
                "destination_country": self.destination_country,
                "destination_country_latin": self.destination_country_latin,
                # Page level value, unless this one departure overrode it.
                "transportation_type": (
                    (offer.transport.transportation_type if offer.transport else None)
                    or self.transportation_type
                ),
                "departure_date": offer.departure.jalali or offer.departure.raw,
                "return_date": offer.ret.jalali or offer.ret.raw,
                "number_of_nights": offer.nights if offer.nights is not None else hotel.nights,
                # Singular at this grain: one row is one hotel on one date.
                "hotels": " + ".join(names) if names else None,
                "hotel_star_rating": hotel.star_rating,
                "image_url": self.image_url,
                "tour_url": self.tour_url,
                "agency_name": self.agency.name,
                "website_url": self.agency.website_url,
                "logo_url": self.agency.logo_url,
                "is_installment_available": self.is_installment_available,
                # Extra context that would otherwise be lost in the flattening.
                "board_type": hotel.board_type,
                "hotel_city": " + ".join(cities) if cities else None,
                "departure_time": offer.departure.time,
                "return_time": offer.ret.time,
                "transport_name": (offer.transport.name if offer.transport else None),
                "transport_duration": (offer.transport.duration if offer.transport else None),
            }
            row.update(_price_columns("adult", offer.adult_price))
            row.update(_price_columns("child", offer.child_price))
            row.update(_price_columns("child_no_bed", offer.child_no_bed_price))
            row.update(_price_columns("infant", offer.infant_price))
            # Single occupancy is a genuinely different price, not a duplicate.
            row.update(_price_columns("single", offer.prices.get(ROOM_SINGLE)))
            row.update(_price_columns("triple", offer.triple_price))
            rows.append(row)
        return rows


