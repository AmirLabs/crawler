"""Step 3's entry point: one URL in, one :class:`ExtractedPage` out.

Everything before this module reads one kind of thing — where a page repeats,
what a price says, which date is printed. This is where those parts are put
together, and it is the first place the whole stage runs end to end.

The page is read in two passes, because a tour page states two kinds of fact:

* **Once per page** — the tour's name, its picture, who is selling it. These
  come from the document head and the agency's own markup, which is the one
  part of these sites that follows a shared convention: all seven publish
  Open Graph tags, so ``og:title`` and ``og:image`` work everywhere without a
  per-site rule.
* **Once per bookable option** — handed to
  :func:`crawler.extraction.row_mapping.map_blocks`, which reads every
  repeating block :func:`crawler.extraction.structure.html_rows.offer_blocks`
  picked out. Every, not the best one: bisanseir.com gives each of its hotels
  its own block of room rows, so the best block alone is one hotel of sixteen.

Both passes read markup. Where a site publishes its own tour JSON —
:mod:`crawler.extraction.tour_api` finds that from the page's own declaration —
the JSON answers instead, because it states exactly what the markup only
implies: an ISO departure time rather than "۲۹ مرداد ساعت ۰۵:۴۵", a numeric
star grade rather than counted stars, an explicit currency rather than a
guessed one. The document still supplies the picture, the agency name and the
logo, which the JSON does not carry.

``origin`` and ``destination`` are left unresolved on the markup path. They are
readable only from prose on most of the target sites, and a guess would be
worse than an empty field that the fill rate can point at; step 5 reads them
from the tour's title instead.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from crawler.config import DEFAULT_TIMEOUT_SECONDS
from crawler.extraction.extractor import fetch_page, render_beside
from crawler.extraction.structure.html_rows import (
    MIN_ROW_SCORE,
    find_row_containers,
    offer_blocks,
)
from crawler.extraction.structure.json_paths import flatten_json
from crawler.extraction.structure.tables import find_rates_tables
from crawler.extraction.models import (
    METHOD_EXACT_KEY,
    METHOD_HEURISTIC_HTML,
    ExtractedField,
    ExtractedPage,
    record_attempt,
)
from crawler.extraction.availability import find_closed_notice, split_by_availability
from crawler.extraction.date_links import find_date_links

# How a departure taken from the link that asked for it is marked, so it is
# never read back as one printed inside the row.
_DATE_LINK_SCOPE = f"{METHOD_HEURISTIC_HTML}:date_link"
from crawler.extraction.page_facts import apply_page_facts, read_page_facts
from crawler.extraction.row_mapping import map_blocks
from crawler.extraction.tour_api import (
    api_url_for,
    enrich_from_api,
    fetch_tour_json,
)
from crawler.extraction.readers.countries import (
    find_city,
    find_country,
    is_a_region,
)
from crawler.extraction.readers.dates import jalali_to_gregorian
from crawler.extraction.readers.places import (
    find_destination,
    find_origin,
    is_a_departure_city,
    is_an_iranian_city,
    latin_city,
    CITY_LATIN,
    read_flight_legs,
    read_title_places,
    without_a_hotel_word,
    without_the_hotel,
)
from crawler.extraction.readers.transport import detect_transport, main_text
from crawler.extraction.target_fields import PAGE_FIELDS, ROW_FIELDS, unresolved
from crawler.profiler.models import SiteProfile

# JSON-LD keys that hold the publisher's logo. Yoast, which several of the
# target sites use, writes the first of these.
_LOGO_PATH_ENDINGS = (".logo.url", ".logo.contentUrl", ".publisher.logo.url")

# The two names the origin default is written in. Iran is the country a tour
# has to state its origin on, since a domestic one leaves from anywhere.
IRAN = "ایران"
TEHRAN = "تهران"

# Which page-level fields this module actually looks for, and where it looks.
# ``is_installment_available`` is missing on purpose: no rule reads it yet, and
# saying so is more useful than implying it was searched for and absent.
_PAGE_ATTEMPTS: dict[str, str] = {
    "tour_name": f"{METHOD_HEURISTIC_HTML}:h1/og:title",
    "destination": f"{METHOD_HEURISTIC_HTML}:tour_name",
    "destination_city_latin": f"{METHOD_EXACT_KEY}:site_taxonomy",
    "destinations": f"{METHOD_HEURISTIC_HTML}:tour_name",
    "image_url": f"{METHOD_HEURISTIC_HTML}:og:image",
    "agency_name": f"{METHOD_HEURISTIC_HTML}:og:site_name",
    "website_url": f"{METHOD_EXACT_KEY}:page_host",
    "logo_url": f"{METHOD_EXACT_KEY}:json_ld_logo",
    "transportation_type": f"{METHOD_HEURISTIC_HTML}:transport",
    "origin": f"{METHOD_HEURISTIC_HTML}:departure_airport",
    "origin_latin": f"{METHOD_EXACT_KEY}:city_latin",
}


def _meta_content(tree: HTMLParser, *names: str) -> Optional[str]:
    """Read the first Open Graph or ``name=`` meta tag that is present.

    Example:
        >>> tree = HTMLParser('<meta property="og:title" content="تور کوالالامپور">')
        >>> _meta_content(tree, "og:title")
        'تور کوالالامپور'
    """
    for name in names:
        for attribute in ("property", "name"):
            node = tree.css_first(f'meta[{attribute}="{name}"]')
            if node is not None:
                content = (node.attributes.get("content") or "").strip()
                if content:
                    return content
    return None


def _read_tour_name(tree: HTMLParser) -> Optional[str]:
    """Read the tour's name from the page.

    The ``<h1>`` is preferred over ``og:title`` because the Open Graph title
    usually carries the agency's name too — "تور کوالالامپور | نهال گشت" —
    while the heading is the tour alone.

    Example:
        >>> _read_tour_name(HTMLParser('<h1> تور کوالالامپور ۷ شب </h1>'))
        'تور کوالالامپور ۷ شب'
    """
    heading = tree.css_first("h1")
    if heading is not None:
        text = " ".join((heading.text(deep=True) or "").split())
        if text:
            return text

    for value in (_meta_content(tree, "og:title", "twitter:title"), None):
        if value:
            return value

    title = tree.css_first("title")
    if title is not None:
        text = " ".join((title.text(deep=True) or "").split())
        if text:
            return text
    return None


def _read_logo_url(embedded_json: list[dict], base: str) -> Optional[str]:
    """Find the agency's logo in the page's JSON-LD, if it published one.

    Example:
        >>> _read_logo_url([{"publisher": {"logo": {"url": "/img/logo.png"}}}], "https://x.com")
        'https://x.com/img/logo.png'
    """
    for blob in embedded_json:
        for path, values in flatten_json(blob).items():
            if any(path.endswith(ending) for ending in _LOGO_PATH_ENDINGS):
                if values and isinstance(values[0], str):
                    return urljoin(base, values[0])
    return None


def read_page_fields(
    tree: HTMLParser,
    url: str,
    embedded_json: Optional[list[dict]] = None,
    html: Optional[str] = None,
) -> dict[str, ExtractedField]:
    """Read the facts that hold for the whole page.

    Example:
        >>> tree = HTMLParser(
        ...     '<html><head>'
        ...     '<meta property="og:image" content="/img/tour.jpg">'
        ...     '<meta property="og:site_name" content="نهال گشت">'
        ...     '</head><body><h1>تور کوالالامپور</h1>'
        ...     '<p>از فرودگاه امام خمینی</p></body></html>'
        ... )
        >>> fields = read_page_fields(tree, "https://nahalgasht.com/tour/1")
        >>> fields["tour_name"].value, fields["agency_name"].value
        ('تور کوالالامپور', 'نهال گشت')
        >>> fields["image_url"].value
        'https://nahalgasht.com/img/tour.jpg'
        >>> fields["destinations"].value, fields["destination"].value
        (['کوالالامپور'], 'کوالالامپور')
        >>> fields["transportation_type"].value, fields["transportation_type"].source_key
        ('air', 'transport:journey')
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else url
    fields: dict[str, ExtractedField] = {}

    def record(name: str, value: object, method: str, source: str) -> None:
        if value in (None, ""):
            return
        fields[name] = ExtractedField(value=value, method=method, source_key=source)

    record("tour_name", _read_tour_name(tree), METHOD_HEURISTIC_HTML, "h1/og:title")

    # Every title measured on the ten sites names where the tour goes, and
    # subtracting the four noise vocabularies reads 22 of the 23 real titles —
    # so this is answered here, for free, rather than sent to a model. The
    # model pass in :mod:`crawler.llm.tour_names` now only sees what is left.
    title = fields["tour_name"].value if "tour_name" in fields else None
    places = read_title_places(title) if isinstance(title, str) else None
    # Three sources, in order of how much of the answer is the site's own. The
    # crumb is where it files the tour, the flight box is what it printed, and
    # the title is what we rebuilt by subtracting noise — which is how
    # "دور آسیا نوروز" became a city.
    words = main_text(html) if html else ""
    filed = find_city(url, embedded_json, tree)
    flown = find_destination(html) if html else None
    if filed is not None:
        record("destinations", [filed.name], METHOD_EXACT_KEY, "site_taxonomy")
        record("destination", filed.name, METHOD_EXACT_KEY, "site_taxonomy")
        record("destination_city_latin", filed.latin, METHOD_EXACT_KEY,
               "site_taxonomy")
    elif places is not None and places.destinations:
        # The title still wins over the flight box when it already names that
        # city: a two-city tour says both in its title and the box names only
        # where the plane lands — vanrosh sells "چشمه و ازمیر", flying to ازمیر.
        if flown and flown not in places.destinations:
            record("destinations", [flown], METHOD_HEURISTIC_HTML, "flight_block")
            record("destination", flown, METHOD_HEURISTIC_HTML, "flight_block")
        else:
            record("destinations", places.destinations, METHOD_HEURISTIC_HTML, "tour_name")
            record("destination", places.display, METHOD_HEURISTIC_HTML, "tour_name")
    elif flown:
        record("destinations", [flown], METHOD_HEURISTIC_HTML, "flight_block")
        record("destination", flown, METHOD_HEURISTIC_HTML, "flight_block")

    # The stops a multi-city tour flies between are added to the searchable
    # list, never to the display name. ``destinations`` exists so a tour can be
    # found by any one of its stops — see ``Tour.to_rows`` — and lachinseir's
    # "تور دور آسیا ویژه نوروز" names none of its three, while its own text
    # lists them leg by leg.
    legs = [
        city for city in read_flight_legs(words)
        if not is_a_departure_city(city)
    ]
    if legs and "destinations" in fields:
        known = list(fields["destinations"].value)
        fields["destinations"] = ExtractedField(
            value=known + [city for city in legs if city not in known],
            method=METHOD_HEURISTIC_HTML,
            source_key=f"{fields['destinations'].source_key}+flight_legs",
        )

    # Where the tour leaves from, read from the airport the page names rather
    # than from a city it mostly does not. Silence stays silent: see
    # :func:`~crawler.extraction.readers.places.find_origin` for why Tehran is
    # not assumed.
    if html:
        record("origin", find_origin(words), METHOD_HEURISTIC_HTML,
               "departure_airport")

    image = _meta_content(tree, "og:image", "twitter:image")
    record("image_url", urljoin(base, image) if image else None, METHOD_HEURISTIC_HTML, "og:image")

    record(
        "agency_name",
        _meta_content(tree, "og:site_name", "application-name"),
        METHOD_HEURISTIC_HTML,
        "og:site_name",
    )
    # The agency's site is the page's own host, which is a fact rather than a
    # reading, so it is recorded as an exact match.
    record("website_url", base if parsed.netloc else None, METHOD_EXACT_KEY, "page host")

    record(
        "logo_url",
        _read_logo_url(embedded_json or [], base),
        METHOD_EXACT_KEY,
        "json-ld logo",
    )

    # Transport is read by its own rules rather than by scanning the page: the
    # navigation menu names every kind of tour a site sells, which used to
    # answer for tours it had nothing to do with. See
    # :mod:`crawler.extraction.readers.transport`.
    tour_name = fields["tour_name"].value if "tour_name" in fields else None
    transport, source = detect_transport(
        html if html is not None else (tree.html or ""),
        tour_name if isinstance(tour_name, str) else None,
    )
    record(
        "transportation_type",
        transport,
        METHOD_HEURISTIC_HTML,
        f"transport:{source}",
    )

    return fields


def extract_page_from_html(
    url: str,
    html: str,
    embedded_json: Optional[list[dict]] = None,
    today: Optional[date] = None,
) -> ExtractedPage:
    """Build an :class:`ExtractedPage` from markup already in hand.

    Kept separate from :func:`extract_page` so a rule can be tried against a
    saved page without going back to the network, which is how every mapping
    rule in this stage was checked.

    Example:
        >>> page = extract_page_from_html(
        ...     "https://x.com/tour/1",
        ...     '<html><body><h1>تور دبی</h1>'
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div></body></html>',
        ... )
        >>> page.row_count, page.page_fields["tour_name"].value
        (2, 'تور دبی')
    """
    tree = HTMLParser(html)
    # Scripts and styles hold code, not content, and their text would otherwise
    # be read as part of a row.
    for node in tree.css("script, style, noscript"):
        node.decompose()

    page = ExtractedPage(url=url, html_raw=html, embedded_json=embedded_json or [])
    page.page_fields = read_page_fields(tree, url, page.embedded_json, html=html)
    _stamp_country(page, tree)
    _drop_the_country_from_destination(page)
    _stamp_origin(page)
    page.unresolved_page_fields = unresolved(page.page_fields, PAGE_FIELDS)
    for name, scope in _PAGE_ATTEMPTS.items():
        if name not in page.page_fields:
            record_attempt(page.page_attempts, name, scope)

    # A rates table states its fields in its header, so it needs none of the
    # shape guessing the block finder does. It is read alongside the blocks
    # rather than instead of them: a page can carry both, and tahagasht.com's
    # hotel-choice tours are a table while its fixed packages are a block.
    table_rows = find_rates_tables(tree, main_text(html))

    containers = find_row_containers(tree, page_url=url)
    blocks = offer_blocks(containers)

    if not blocks and not table_rows:
        if not containers:
            page.notes.append("no repeating block with prices found")
        else:
            best = containers[0]
            page.notes.append(
                f"best block scored {best.score:.2f}, below {MIN_ROW_SCORE} "
                f"({'; '.join(best.reasons)})"
            )
        return page

    rows = map_blocks(blocks) + table_rows
    if table_rows:
        page.notes.append(f"{len(table_rows)} rows from a rates table")
    # Some sites state the departure and the hotels once, above or below the
    # price table, and leave the rows saying only "2 تخته". Those facts are
    # filled in before availability is judged, so a page-level departure can
    # retire a page-level offer.
    facts = read_page_facts(html)
    before = len(rows)
    rows, added = apply_page_facts(rows, facts)
    page.rows, page.expired_rows = split_by_availability(rows, today)
    _drop_the_hotel_from_destination(page)
    if added:
        page.notes.append(f"filled {added} fields from facts stated once for the page")
    if len(rows) != before:
        page.notes.append(
            f"{before} rows x {len(facts.departures)} departures the page states"
        )
    if blocks:
        page.notes.append(
            f"rows from {len(blocks)} x {blocks[0].signature} "
            f"(score {blocks[0].score:.2f}, first at {blocks[0].path})"
        )
    if page.expired_rows:
        page.notes.append(f"{len(page.expired_rows)} offers had already departed")

    # The site's own notice explains an empty page; it never decides one. On
    # alefbatour and avisatravel it sits above departures that are still open.
    if not page.sellable:
        notice = find_closed_notice(main_text(html))
        page.notes.append(
            f"nothing left to sell; the page says {notice!r}" if notice
            else "nothing left to sell and the page gives no reason"
        )
    return page


def _stamp_country(page: ExtractedPage, tree: HTMLParser) -> None:
    """Record the country over the tour's destination, once the city is final.

    Read after the other page fields because the sources it falls back on are
    the tour's own name and destination, which the reading above settles.

    Example:
        >>> page = ExtractedPage(url="https://x.com/tour/turkey-tour/1")
        >>> _stamp_country(page, HTMLParser("<html></html>"))
        >>> page.page_fields["destination_country"].value
        'ترکیه'
        >>> page.page_fields["destination_country_latin"].value
        'turkey'
    """
    def named(field: str) -> Optional[str]:
        found = page.page_fields.get(field)
        return found.value if found and isinstance(found.value, str) else None

    country = find_country(
        page.url,
        page.embedded_json,
        tree,
        title=named("tour_name"),
        city=named("destination"),
    )
    if country is None:
        for name in ("destination_country", "destination_country_latin"):
            record_attempt(page.page_attempts, name, "site_taxonomy")
        return
    for name, value in (("destination_country", country.name),
                        ("destination_country_latin", country.latin)):
        page.page_fields[name] = ExtractedField(
            value=value, method=METHOD_HEURISTIC_HTML, source_key="site_taxonomy"
        )


def _stamp_origin(page: ExtractedPage) -> None:
    """Fill the origin of a tour that goes abroad and never says it leaves Tehran.

    Four of the ten sites name no departure airport on any page, so the field
    was empty on every row they produce. They are not leaving from somewhere
    else — of the 77 origins the run of 2026-09-09 did read, all 77 were
    Tehran. They simply do not write down what their customers already know.

    So a tour known to go abroad is given Tehran, and a tour known to stay in
    the country is not: Mashhad and Kish tours leave from everywhere, and there
    the empty field is the true answer. Where neither is known, nothing is
    filled.

    The default carries its own source, so a run can still count what was read
    apart from what was assumed. This is the only field in the crawl that
    carries a value no page stated, and it stays countable for that reason.

    Example:
        >>> page = ExtractedPage(url="https://x.com/tour/1")
        >>> page.page_fields["destination_country"] = ExtractedField(
        ...     "ترکیه", METHOD_HEURISTIC_HTML, "site_taxonomy")
        >>> _stamp_origin(page)
        >>> page.page_fields["origin"].value, page.page_fields["origin"].source_key
        ('تهران', 'default:tehran')
        >>> page.page_fields["origin_latin"].value
        'tehran'

        A domestic tour keeps the empty field it earned:

        >>> home = ExtractedPage(url="https://x.com/tour/2")
        >>> home.page_fields["destination_country"] = ExtractedField(
        ...     "ایران", METHOD_HEURISTIC_HTML, "site_taxonomy")
        >>> _stamp_origin(home)
        >>> "origin" in home.page_fields
        False
    """
    def named(field: str) -> Optional[str]:
        found = page.page_fields.get(field)
        return found.value if found and isinstance(found.value, str) else None

    read = named("origin")
    if read:
        latin = latin_city(read)
        if latin:
            page.page_fields["origin_latin"] = ExtractedField(
                value=latin, method=METHOD_EXACT_KEY, source_key="city_latin"
            )
        return

    country = named("destination_country")
    if country == IRAN:
        return
    goes_abroad = country is not None
    if country is None:
        city = named("destination")
        goes_abroad = bool(city) and not is_an_iranian_city(city)
    if not goes_abroad:
        return

    for name, value in (("origin", TEHRAN), ("origin_latin", CITY_LATIN[TEHRAN])):
        page.page_fields[name] = ExtractedField(
            value=value, method=METHOD_HEURISTIC_HTML, source_key="default:tehran"
        )


def _drop_the_country_from_destination(page: ExtractedPage) -> None:
    """Take the country back out of a destination read from a title.

    A title names both — vanrosh sells "تور مالزی – 7 شب کوالالامپور" — and
    subtracting noise cannot tell which word is the country. Now that the
    country is read on its own, it can simply be removed, along with a
    continent: liyaseir's "اسکاندیناوی" is not a city either. Where nothing is
    left the field goes empty, which is the honest answer.

    Only this tour's own country goes. Every country is not dropped, because
    some are also the city — a tour to "کوالالامپور + سنگاپور" keeps both.

    Only a destination read from the title is touched. One filed by the site or
    printed in a flight box named a city to begin with.

    Example:
        >>> page = ExtractedPage(url="https://x.com/tour/1")
        >>> page.page_fields["destination_country"] = ExtractedField(
        ...     "مالزی", METHOD_HEURISTIC_HTML, "site_taxonomy")
        >>> page.page_fields["destinations"] = ExtractedField(
        ...     ["مالزی", "کوالالامپور"], METHOD_HEURISTIC_HTML, "tour_name")
        >>> page.page_fields["destination"] = ExtractedField(
        ...     "مالزی + کوالالامپور", METHOD_HEURISTIC_HTML, "tour_name")
        >>> _drop_the_country_from_destination(page)
        >>> page.page_fields["destination"].value
        'کوالالامپور'
        >>> page.page_fields["destinations"].value
        ['کوالالامپور']
    """
    field = page.page_fields.get("destination")
    if field is None or not field.source_key.endswith("tour_name"):
        return
    country = page.page_fields.get("destination_country")
    named = country.value if country else None

    listed = page.page_fields.get("destinations")
    places = list(listed.value) if listed and isinstance(listed.value, list) else []
    kept = [
        place for place in places
        if place != named and not is_a_region(place)
    ]
    if kept == places:
        return

    for name in ("destination", "destinations"):
        page.page_fields.pop(name, None)
        record_attempt(page.page_attempts, name, _PAGE_ATTEMPTS[name])
    if not kept:
        return
    page.page_fields["destinations"] = ExtractedField(
        value=kept, method=field.method, source_key=f"{field.source_key}-country"
    )
    page.page_fields["destination"] = ExtractedField(
        value=" + ".join(kept), method=field.method,
        source_key=f"{field.source_key}-country",
    )


def _drop_the_hotel_from_destination(page: ExtractedPage) -> None:
    """Take the hotel's name back out of a destination read from the title.

    Some titles name the hotel as well as the place — nahalgasht sells "تور
    زمینی تبریز هتل گسترش" — and subtracting noise cannot know which words
    belong to a hotel, because hotels are named after everything. It does not
    have to: the page states the hotel in its own rows, and by the time they
    are built the name is in hand.

    Only a destination read from the title is touched. One read from the flight
    box came from the page saying the city itself and has no hotel in it.

    Example:
        >>> page = ExtractedPage(url="u")
        >>> page.page_fields["destination"] = ExtractedField(
        ...     "زمینی تبریز هتل گسترش", METHOD_HEURISTIC_HTML, "tour_name")
        >>> page.rows = [ExtractedRow(fields={"hotel_name": ExtractedField(
        ...     "هتل گسترش", METHOD_HEURISTIC_HTML, "hotel_block")})]
        >>> _drop_the_hotel_from_destination(page)
        >>> page.page_fields["destination"].value
        'تبریز'
        >>> page.page_fields["destinations"].value
        ['تبریز']
    """
    field = page.page_fields.get("destination")
    if field is None or not isinstance(field.value, str):
        return
    if not (field.source_key or "").endswith("tour_name"):
        return

    left = without_a_hotel_word(field.value) or field.value
    for row in page.rows + page.expired_rows:
        hotel = row.fields.get("hotel_name")
        if hotel is not None and isinstance(hotel.value, str):
            left = without_the_hotel(left, hotel.value) or left
    if left == field.value:
        return

    # Read again, now that the hotel is out of the way. What the hotel's name
    # was hiding is ordinary noise — "تور زمینی تبریز هتل گسترش" leaves
    # "زمینی تبریز", and زمینی is how the tour travels, not where it goes.
    cleaned = read_title_places(left)
    places = cleaned.destinations or [left]
    page.page_fields["destination"] = ExtractedField(
        cleaned.display or left, field.method, f"{field.source_key}-hotel_name"
    )
    listed = page.page_fields.get("destinations")
    extra = [] if listed is None or not isinstance(listed.value, list) else [
        place for place in listed.value if place not in field.value
    ]
    page.page_fields["destinations"] = ExtractedField(
        places + [place for place in extra if place not in places],
        field.method, f"{field.source_key}-hotel_name",
    )


async def extract_page(
    url: str,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    today: Optional[date] = None,
    follow_dates: bool = True,
) -> ExtractedPage:
    """Fetch one tour URL and read everything step 3 knows how to read.

    A fetch that failed is returned as it came back, error and all, rather than
    being turned into an empty page: the reason a URL yielded nothing is worth
    as much to the daily refresh as the data would have been.

    ``follow_dates`` is what stops the walk after one step. A page that offers
    other dates links to them, and each of those pages links back, so the
    fetches this makes are told not to follow any of their own.

    Example:
        >>> profile = SiteProfile(domain="nahalgasht.com")
        >>> page = await extract_page("https://nahalgasht.com/tour/1", profile)
        >>> page.status_code
        200
    """
    page = await _extract_one(url, profile, client, timeout, today)
    if not follow_dates or not page.html_raw:
        return page
    return await _add_other_dates(page, profile, client, timeout, today)


async def _add_other_dates(
    page: ExtractedPage,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient],
    timeout: float,
    today: Optional[date],
) -> ExtractedPage:
    """Fetch the other departures the page links to, and keep their rows.

    Prices really do move between dates, so each set is read where it is
    published rather than copied. On a bisanseir page the same room is
    71,000,000 by default and 67,000,000 on the date the site badges cheapest.

    The two shapes are merged differently, because of what each list contains:

    * a query list (``?date=``) names every date **including** the one already
      on screen, so these rows replace the page's own;
    * a sibling list names every date **except** this one, so these rows are
      added to the page's own.

    A date already gone is not fetched at all, which is free to know: the query
    shape writes the date into the link. The sibling shape does not, so those
    are fetched and judged afterwards, as they always were.
    """
    links = find_date_links(page.html_raw or "", page.url)
    if not links:
        return page

    wanted = [link for link in links if _still_to_come(link.date, today)]
    skipped = len(links) - len(wanted)
    replaces = any(link.replaces_this_page for link in links)
    if replaces:
        page.rows, page.expired_rows = [], []

    added = 0
    for link in wanted:
        if link.url == page.url:
            continue
        other = await extract_page(
            link.url, profile, client=client, timeout=timeout, today=today,
            follow_dates=False,
        )
        _stamp_date(other.rows + other.expired_rows, link.date)
        page.rows.extend(other.rows)
        page.expired_rows.extend(other.expired_rows)
        added += len(other.rows)

    page.notes.append(
        f"{len(wanted)} of {len(links)} dates the page links to were fetched"
        + (f", {skipped} already gone" if skipped else "")
        + f"; {added} rows came back"
    )
    return page


def _stamp_date(rows: list, jalali: Optional[str]) -> None:
    """Give the rows of a per-date fetch the date that fetch asked for.

    Without this the eight date sets of a bisanseir page come back as 576 rows
    that all say nothing about when they leave — eight prices for one room with
    no way to tell them apart, which is worse than the 72 dateless rows the
    page gave before.

    A row that already names its own departure is left alone, and so is a link
    that did not say its date: the sibling shape carries only an id, and there
    the page itself states the date once fetched.

    Example:
        >>> from crawler.extraction.models import ExtractedRow
        >>> rows = [ExtractedRow()]
        >>> _stamp_date(rows, "1405-06-11")
        >>> rows[0].fields["departure_date"].value
        {'iso': '2026-09-02', 'raw': '1405-06-11', 'time': None}
    """
    if not jalali:
        return
    try:
        year, month, day = (int(part) for part in jalali.split("-"))
        iso = date(*jalali_to_gregorian(year, month, day)).isoformat()
    except (ValueError, TypeError):
        return
    value = {"iso": iso, "raw": jalali, "time": None}
    for row in rows:
        if "departure_date" in row.fields:
            continue
        row.fields["departure_date"] = ExtractedField(
            value, METHOD_HEURISTIC_HTML, _DATE_LINK_SCOPE
        )
        row.attempts.pop("departure_date", None)
        row.unresolved_fields = unresolved(row.fields, ROW_FIELDS)


def _still_to_come(jalali: Optional[str], today: Optional[date]) -> bool:
    """Whether a date written into a link has not already passed.

    A link that does not say its date is always fetched: silence is not
    evidence of expiry, the same rule
    :func:`crawler.extraction.availability.is_bookable` follows.

    Example:
        >>> _still_to_come(None, date(2026, 8, 25))
        True
        >>> _still_to_come("1405-06-25", date(2026, 8, 25))
        True
        >>> _still_to_come("1405-04-25", date(2026, 8, 25))
        False
    """
    if not jalali:
        return True
    try:
        year, month, day = (int(part) for part in jalali.split("-"))
        gy, gm, gd = jalali_to_gregorian(year, month, day)
        return date(gy, gm, gd) >= (today or date.today())
    except (ValueError, TypeError):
        return True


async def _extract_one(
    url: str,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient],
    timeout: float,
    today: Optional[date],
) -> ExtractedPage:
    """One URL, fetched and read, with no other date followed.

    The markup is read first, then the site's own JSON is asked, and only if
    both come back empty is a browser opened. That order is what the browser
    costs: 6 to 13 seconds against one small GET.

    Rendering last rather than first is what radingasht.com needed. It declares
    an API, so the fetch layer stood down — but the API answers 404 for its own
    tour ids, and the hotel list is drawn by JavaScript, so the page yielded
    nothing at all. Deciding on the declaration is deciding on a promise;
    deciding after it is deciding on what arrived.
    """
    fetched = await fetch_page(url, profile, client=client, timeout=timeout)
    if not fetched.html_raw:
        return fetched

    fetched = _settle_render_question(fetched, profile, today)
    page = await _read_everything(url, fetched, client, timeout, today)

    if page.rows or not _worth_rendering(fetched, profile):
        return page

    rendered = await render_beside(
        fetched,
        "neither the markup nor the API gave a row; rendered to see if there are any",
        timeout=timeout,
    )
    if rendered.html_before_render is None:
        return page  # the browser would not start; keep what we have

    rendered = _settle_render_question(rendered, profile, today)
    return await _read_everything(url, rendered, client, timeout, today)


async def _read_everything(
    url: str,
    fetched: ExtractedPage,
    client: Optional[httpx.AsyncClient],
    timeout: float,
    today: Optional[date],
) -> ExtractedPage:
    """Read one copy of the markup, and let the site's own JSON overrule it."""
    page = extract_page_from_html(url, fetched.html_raw, fetched.embedded_json, today=today)
    # Carry over what the fetch layer learned, so provenance survives.
    page.fetch_method = fetched.fetch_method
    page.status_code = fetched.status_code
    page.fetched_at = fetched.fetched_at
    page.content_hash = fetched.content_hash
    page.notes = fetched.notes + page.notes

    # A site that publishes its own tour JSON knows its offers exactly, so its
    # answer replaces what the markup was read for. Everything the JSON does not
    # carry — the picture, the agency, the logo — stays as the document gave it.
    api_url = api_url_for(url, fetched.html_raw or "")
    if api_url:
        data, error = await _tour_json(api_url, client, timeout)
        if data is not None:
            page = enrich_from_api(page, data)
            page.rows, page.expired_rows = split_by_availability(page.rows, today)
        else:
            page.notes.append(f"site publishes an API but it did not answer: {error}")

    return page


def _worth_rendering(fetched: ExtractedPage, profile: SiteProfile) -> bool:
    """Whether a browser is worth trying on a page that gave up nothing.

    ``needs_rendering`` asks whether the markup *looks* client-side, by counting
    ``{{ }}`` placeholders. That is the right question for a raw template and
    the wrong one for a compiled framework: Next.js and Vue turn their
    templates into code before serving, so nothing is left to count. Measured
    on the served markup of a real tour page:

        chakavaktravel   0 placeholders,   1,645 visible chars, 65,056 in script
        radingasht       0 placeholders,   3,559 visible chars,  5,316 in script

    Both come back with 0 rows plain and score 0.70 rendered. No property of
    the markup separates them from a site that needs no browser — liyaseir.com
    devotes a *smaller* share of its bytes to visible text than chakavaktravel
    does and reads fine without one. So the trigger is the outcome rather than
    the markup: nothing came out, so try the other way of looking.

    Two cases are left alone. A page already rendered has nothing to retry, and
    a site already measured as gaining nothing keeps that answer — which is
    what stops this costing 6 to 13 seconds on every empty page rather than
    once per site.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> empty = ExtractedPage(url="https://example.com/t/1", html_raw="<body></body>")
        >>> _worth_rendering(empty, profile)
        True
        >>> profile.rendering.render_helps = False
        >>> _worth_rendering(empty, profile)
        False
    """
    if fetched.html_before_render is not None:
        return False  # the browser has already had its turn on this page
    return profile.rendering.render_helps is not False


def _settle_render_question(
    fetched: ExtractedPage, profile: SiteProfile, today: Optional[date]
) -> ExtractedPage:
    """Decide once per site whether rendering earns its cost, and record it.

    The fetch layer renders a page whose markup still holds template
    placeholders, because it cannot know whether that matters. Here the rows
    from both versions can be counted, which settles it — and the answer is
    written onto the profile so the rest of the site skips the browser.

    It is worth measuring rather than assuming. Rendering costs 6 to 13 seconds
    a page. On alefbatour.com it was the difference between no hotels and all of
    them; on nahalgasht.com one page went from 19 rows to 1, so rendering was
    not merely wasted but harmful.

    A page fetched without a render, or after the question is settled, passes
    through untouched.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> page = ExtractedPage(url="https://example.com/t/1", html_raw="<body></body>")
        >>> _settle_render_question(page, profile, None) is page
        True
        >>> profile.rendering.render_helps is None
        True
    """
    if not fetched.html_before_render or not fetched.html_raw:
        return fetched

    before = fetched.html_before_render
    plain = extract_page_from_html(fetched.url, before, today=today)
    rendered_rows = len(
        extract_page_from_html(fetched.url, fetched.html_raw, fetched.embedded_json,
                               today=today).rows
    )
    helped = rendered_rows > len(plain.rows)
    # A page that gave nothing either way says nothing about the site. On
    # lachinseir.com a guide page scored 0 against 0 and switched rendering off
    # for the 29 tour pages a rendered sample had just proved it worked on.
    if rendered_rows or plain.rows:
        profile.rendering.render_helps = helped
    profile.rendering.notes = (
        f"rendering a tour page gave {rendered_rows} rows against "
        f"{len(plain.rows)} without it"
    )

    fetched.html_before_render = None
    if helped:
        fetched.notes.append(f"rendering earned its cost here ({rendered_rows} rows)")
        return fetched

    # The plain markup was as good or better, so keep it and stop rendering
    # this site. The browser fetch is already spent; the rest of the site is not.
    fetched.notes.append(
        f"rendering gave {rendered_rows} rows against {len(plain.rows)} plain; "
        "not rendering this site again"
    )
    plain.fetch_method = fetched.fetch_method
    plain.status_code = fetched.status_code
    plain.fetched_at = fetched.fetched_at
    plain.content_hash = fetched.content_hash
    plain.notes = fetched.notes
    plain.html_raw = before
    return plain


async def _tour_json(
    api_url: str, client: Optional[httpx.AsyncClient], timeout: float
) -> tuple[Optional[dict], Optional[str]]:
    """Fetch a tour's JSON, borrowing the caller's client when there is one."""
    if client is not None:
        return await fetch_tour_json(client, api_url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as owned:
        return await fetch_tour_json(owned, api_url)
