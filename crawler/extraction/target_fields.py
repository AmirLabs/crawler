"""The fields extraction tries to fill, and where each one lands.

Until now nothing said what step 3 is actually looking for, so
:attr:`~crawler.extraction.models.ExtractedPage.fill_rate` had no denominator
and "unresolved" had no list to draw from. This module is that list.

The names are the ones :meth:`crawler.schema.output_models.Tour.to_rows`
produces, so step 4 can read a resolved field straight across instead of
translating twice.

The split matters as much as the names. A tour page states its name once and
its prices many times, and the two need different handling:

* :data:`PAGE_FIELDS` — one value for the whole page.
* :data:`ROW_FIELDS` — one value per bookable option.

Which row fields a page can fill depends on what it repeats, and the target
sites repeat different things: bisanseir.com and vanrosh.com give one row per
room type, so their rows carry ``room_type`` and no date; alefbatour.com and
avisatravel.com give one row per departure, so their rows carry
``departure_date`` and no room type. Both are complete pages. A field left
unresolved is therefore normal, not a failure — which is why the row list is
what a page is *measured against* rather than what it must satisfy.
"""

from __future__ import annotations

# Facts that hold for the whole page. These fill the top level of
# :class:`~crawler.schema.output_models.Tour`.
PAGE_FIELDS: tuple[str, ...] = (
    "tour_name",
    "origin",
    # Both forms of where the tour goes: the joined string a result page shows,
    # and the list a search matches against. See
    # :class:`crawler.schema.output_models.Tour`.
    "destination",
    "destinations",
    # The country over those cities, which no page states in its text and
    # every one of them states in its filing. See
    # :mod:`crawler.extraction.readers.countries`.
    "destination_country",
    "destination_country_latin",
    "transportation_type",
    "image_url",
    "is_installment_available",
    "agency_name",
    "website_url",
    "logo_url",
)

# Facts that belong to one bookable option. These fill one
# :class:`~crawler.schema.output_models.TourOffer`.
ROW_FIELDS: tuple[str, ...] = (
    # when
    "departure_date",
    "return_date",
    "nights",
    # where
    "hotel_name",
    "hotel_star_rating",
    "hotel_city",
    "board_type",
    # who the price is for — set when the page repeats by occupancy rather
    # than by date, as bisanseir.com and vanrosh.com do
    "room_type",
    # how much
    "price",
    # how the traveller gets there
    "transport_name",
    "transport_duration",
)

# Every field name step 3 knows about, for validating a mapping rule against
# something rather than letting a typo become a silently unfilled field.
ALL_FIELDS: frozenset[str] = frozenset(PAGE_FIELDS) | frozenset(ROW_FIELDS)


def unresolved(found: dict[str, object], scope: tuple[str, ...]) -> list[str]:
    """List the fields of ``scope`` that ``found`` does not cover.

    Order follows ``scope`` rather than the dict, so two pages of the same site
    report their gaps in the same order and can be compared at a glance.

    Example:
        >>> unresolved({"hotel_name": "هتل کورت", "price": 92000000}, ROW_FIELDS)
        ['departure_date', 'return_date', 'nights', 'hotel_star_rating',
         'hotel_city', 'board_type', 'room_type', 'transport_name',
         'transport_duration']
    """
    return [name for name in scope if name not in found]
