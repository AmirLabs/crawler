"""The country a tour goes to, read from where its site files it.

The tour's own words rarely say it. Measured over the 25 tours of the first
client run, 3 titles name a country and 22 name a city — "تور آنتالیا",
"تور کوش آداسی", "تور بدروم" — and the API the three platform sites share
answers ``destination.country: null``. So the country is not in the tour, it is
in the filing around it: the breadcrumb above the page, the category the page
belongs to, and the slug of its own URL. All three are the site saying where it
keeps this tour, which is the same answer written three ways.

The table below is a closed list of countries, the way ``_IRANIAN_AIRPORTS`` in
:mod:`~crawler.extraction.readers.places` is a closed list of airports: a fact
about the world, not a rule about a site. It is needed because a category path
is ``turkiye/antalya`` on one site and ``europe-tours/france-tours`` on another,
and only a country list tells which segment is the country. It stays small on
purpose — we rejected a place library for *cities* after measuring that only 13
of 45 place mentions were in it, but countries are few, they do not change, and
the agencies sell about sixty of them.

Both names come back together. The latin one is free — it is the slug the site
already put in the link — and it fills ``destination_country_name_en``, which
the client's second structure asks for and nothing else in the crawl can
answer.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional
from urllib.parse import unquote, urlparse

from selectolax.parser import HTMLParser

from crawler.extraction.readers.places import is_a_departure_city


class Country(NamedTuple):
    """A country under both names: as a page writes it, and as a slug spells it."""

    name: str
    latin: str


# Persian name -> the latin slug these sites use for it. The Persian side is
# what the client's output carries; the latin side is what the URL carries.
COUNTRIES: dict[str, str] = {
    "ترکیه": "turkey",
    "امارات متحده عربی": "emirates",
    "امارات": "emirates",
    "گرجستان": "georgia",
    "ارمنستان": "armenia",
    "آذربایجان": "azerbaijan",
    "روسیه": "russia",
    "تایلند": "thailand",
    "مالزی": "malaysia",
    "اندونزی": "indonesia",
    "ویتنام": "vietnam",
    "چین": "china",
    "هند": "india",
    "سریلانکا": "sri-lanka",
    "سری لانکا": "sri-lanka",
    "سنگاپور": "singapore",
    "فیلیپین": "philippines",
    "کامبوج": "cambodia",
    "لائوس": "laos",
    "میانمار": "myanmar",
    "نپال": "nepal",
    "پاکستان": "pakistan",
    "ژاپن": "japan",
    "کره جنوبی": "south-korea",
    "قبرس شمالی": "north-cyprus",
    "قبرس": "cyprus",
    "عمان": "oman",
    "قطر": "qatar",
    "بحرین": "bahrain",
    "کویت": "kuwait",
    "عربستان": "saudi-arabia",
    "عراق": "iraq",
    "لبنان": "lebanon",
    "اردن": "jordan",
    "سوریه": "syria",
    "مصر": "egypt",
    "تونس": "tunisia",
    "مراکش": "morocco",
    "کنیا": "kenya",
    "تانزانیا": "tanzania",
    "آفریقای جنوبی": "south-africa",
    "موریس": "mauritius",
    "مالدیو": "maldives",
    "سیشل": "seychelles",
    "تاجیکستان": "tajikistan",
    "ازبکستان": "uzbekistan",
    "قزاقستان": "kazakhstan",
    "قرقیزستان": "kyrgyzstan",
    "بلاروس": "belarus",
    "اوکراین": "ukraine",
    "فرانسه": "france",
    "ایتالیا": "italy",
    "اسپانیا": "spain",
    "پرتغال": "portugal",
    "آلمان": "germany",
    "اتریش": "austria",
    "سوئیس": "switzerland",
    "هلند": "netherlands",
    "بلژیک": "belgium",
    "انگلستان": "england",
    "انگلیس": "england",
    "ایرلند": "ireland",
    "یونان": "greece",
    "مجارستان": "hungary",
    "چک": "czech",
    "لهستان": "poland",
    "رومانی": "romania",
    "بلغارستان": "bulgaria",
    "صربستان": "serbia",
    "کرواسی": "croatia",
    "بوسنی": "bosnia",
    "سوئد": "sweden",
    "نروژ": "norway",
    "دانمارک": "denmark",
    "فنلاند": "finland",
    "برزیل": "brazil",
    "آرژانتین": "argentina",
    "مکزیک": "mexico",
    "کوبا": "cuba",
    "کانادا": "canada",
    "استرالیا": "australia",
    "نیوزیلند": "new-zealand",
    "ایران": "iran",
}

# The other spellings a link uses for the same country. A site writes Turkey
# four ways across its own pages — ``turkey``, ``turkiye``, ``turkey-tour``,
# ``تور ترکیه`` — and the last two are handled by stripping, not listed here.
_ALIASES: dict[str, str] = {
    "turkiye": "ترکیه",
    "united-arab-emirates": "امارات",
    "uae": "امارات",
    "holland": "هلند",
    "czechia": "چک",
    "korea": "کره جنوبی",
    "northern-cyprus": "قبرس شمالی",
    "srilanka": "سریلانکا",
    "zanzibar": "تانزانیا",
    "uk": "انگلستان",
    "united-kingdom": "انگلستان",
    "russian": "روسیه",
}

_BY_SLUG: dict[str, str] = {slug: name for name, slug in COUNTRIES.items()}
_BY_SLUG.update(_ALIASES)

# Longest first for the same reason the name pattern is: "north-cyprus" must
# not be read as "cyprus".
_SLUGS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(_BY_SLUG, key=len, reverse=True)
)

# Longest first, so "قبرس شمالی" is not read as "قبرس", and "امارات متحده
# عربی" not as "امارات". The same ordering bug cost three titles in places.py.
_A_COUNTRY = re.compile(
    "(?<![؀-ۿ])(?:"
    + "|".join(sorted(COUNTRIES, key=len, reverse=True))
    + ")(?![؀-ۿ])"
)

# What a site wraps a country slug in when it turns it into a page.
_SLUG_NOISE = re.compile(r"^(?:tour-|tours-)|(?:-tour|-tours|-combo|-combined)$")

# What a page wraps a country name in when it prints it in a menu or a crumb.
_NAME_NOISE = re.compile(r"^(?:تور(?:های)?|سفر)[\s‌]*|[\s‌]*تور$")


def country_named(text: str) -> Optional[Country]:
    """The country a piece of text names, under both of its names.

    Reads a crumb, a menu entry or a title — anything short that a site wrote
    to say where a tour goes. Returns ``None`` rather than a guess when the
    text names a city, which is what most of them name.

    Example:
        >>> country_named("تور ترکیه")
        Country(name='ترکیه', latin='turkey')
        >>> country_named("تور قبرس شمالی").latin
        'north-cyprus'
        >>> country_named("تور آنتالیا") is None
        True
    """
    if not text:
        return None
    found = _A_COUNTRY.search(text)
    if not found:
        return None
    name = found.group(0)
    return Country(name=name, latin=COUNTRIES[name])


def country_slugged(segment: str) -> Optional[Country]:
    """The country a URL segment spells, once the site's wrapping is off.

    A segment carries the country the way that site files it: ``turkey``,
    ``turkey-tour``, ``france-tours``, ``35616-russia``. All four are the same
    fact, so all four are stripped down to it.

    Example:
        >>> country_slugged("france-tours")
        Country(name='فرانسه', latin='france')
        >>> country_slugged("35616-russia").name
        'روسیه'
        >>> country_slugged("6-days-france").latin
        'france'
        >>> country_slugged("antalya") is None
        True
    """
    cleaned = _SLUG_NOISE.sub("", (segment or "").strip("/").lower())
    cleaned = re.sub(r"^\d+-", "", cleaned)
    name = _BY_SLUG.get(cleaned)
    if name is None:
        # The country can also be one word of a longer slug — liyaseir writes
        # "6-days-france". Only a whole word at either end counts, so "france"
        # is found and the "africa" inside "south-africa" is not.
        for slug in _SLUGS_BY_LENGTH:
            if cleaned.startswith(f"{slug}-") or cleaned.endswith(f"-{slug}"):
                name = _BY_SLUG[slug]
                break
    if name is None:
        return None
    return Country(name=name, latin=COUNTRIES[name])


def clean_country_text(text: str) -> str:
    """Take the word "تور" off a crumb or menu entry, leaving the place.

    Example:
        >>> clean_country_text("تورهای ترکیه ")
        'ترکیه'
    """
    return _NAME_NOISE.sub("", (text or "").strip()).strip()


def _crumb_lists(blob: object) -> list[list[dict]]:
    """Every ``BreadcrumbList`` inside one JSON-LD blob, however deep it sits.

    Sites nest it differently — nahalgasht puts it in an ``@graph``, alefbatour
    gives it a script of its own — so the blob is walked rather than indexed.

    Example:
        >>> _crumb_lists({"@graph": [{"@type": "BreadcrumbList",
        ...                           "itemListElement": [{"name": "تور"}]}]})
        [[{'name': 'تور'}]]
    """
    found: list[list[dict]] = []
    if isinstance(blob, dict):
        if blob.get("@type") == "BreadcrumbList":
            items = blob.get("itemListElement")
            if isinstance(items, list):
                found.append([item for item in items if isinstance(item, dict)])
        for value in blob.values():
            found.extend(_crumb_lists(value))
    elif isinstance(blob, list):
        for value in blob:
            found.extend(_crumb_lists(value))
    return found


def read_crumbs(
    embedded_json: Optional[list[dict]] = None,
    tree: Optional[HTMLParser] = None,
) -> list[tuple[str, str]]:
    """The trail above a tour page, as ``(text, link)`` from root to page.

    The JSON-LD trail is read first because it carries the link as well as the
    text, and the link holds the latin name. The markup trail is read only when
    there is no JSON-LD one — bisanseir publishes both and they say the same
    thing.

    Example:
        >>> read_crumbs([{"@type": "BreadcrumbList", "itemListElement": [
        ...     {"name": "تور", "item": "https://x.com/tour"},
        ...     {"name": "تور ترکیه", "item": "https://x.com/tour/turkey"}]}])
        [('تور', 'https://x.com/tour'), ('تور ترکیه', 'https://x.com/tour/turkey')]
    """
    trail: list[tuple[str, str]] = []
    for blob in embedded_json or []:
        for items in _crumb_lists(blob):
            for item in items:
                name = item.get("name") or ""
                link = item.get("item")
                if isinstance(link, dict):
                    link = link.get("@id")
                trail.append((str(name), str(link or "")))
    if trail or tree is None:
        return trail

    for holder in tree.css('[class*="breadcrumb"], [class*="crumb"], [id*="breadcrumb"]'):
        for anchor in holder.css("a"):
            text = (anchor.text() or "").strip()
            if text:
                trail.append((text, anchor.attributes.get("href") or ""))
        if trail:
            break
    return trail


def _country_of(text: str, link: str) -> Optional[Country]:
    """The country one crumb states, preferring the link's spelling of it."""
    named = country_named(clean_country_text(text))
    if named is None:
        return None
    for segment in reversed(urlparse(link).path.split("/")):
        slugged = country_slugged(segment)
        if slugged is not None and slugged.name == named.name:
            return slugged
    return named


def find_country(
    url: str,
    embedded_json: Optional[list[dict]] = None,
    tree: Optional[HTMLParser] = None,
    title: Optional[str] = None,
    city: Optional[str] = None,
) -> Optional[Country]:
    """Where the site files this tour, read in the order the sources are sure.

    The crumb wins because it is the site's own answer for this page. Its own
    Its own URL comes next — five of the ten sites put the country slug in it,
    and lachinseir writes it in Persian. Then the menu, which files the city under
    a country. The title is last and answers least: 3 of 25 tours name their
    country in it, but those three are right and cost nothing.

    Returns ``None`` when nothing states it, so an empty column stays visible
    instead of filling with the country of whatever menu the page carries.

    Example:
        >>> find_country("https://x.com/tour/1", [{"@type": "BreadcrumbList",
        ...     "itemListElement": [{"name": "تور ارمنستان",
        ...                          "item": "https://x.com/tour/armenia"}]}])
        Country(name='ارمنستان', latin='armenia')
        >>> find_country("https://nahalgasht.com/tour/35616-russia/").name
        'روسیه'
        >>> find_country("https://lachinseir.com/%D8%AA%D9%88%D8%B1%D9%87%D8%A7%DB%8C-%D8%AA%D9%88%D9%86%D8%B3").latin
        'tunisia'
        >>> find_country("https://x.com/tour/1", title="تور مشهد", city="مشهد")
        Country(name='ایران', latin='iran')
        >>> find_country("https://x.com/tour/28445-antalya/") is None
        True
    """
    for text, link in read_crumbs(embedded_json, tree):
        found = _country_of(text, link)
        if found is not None:
            return found

    for segment in urlparse(url).path.split("/"):
        found = country_slugged(segment) or country_named(unquote(segment))
        if found is not None:
            return found

    if tree is not None:
        found = country_beside(tree, " ".join(filter(None, (title, city))))
        if found is not None:
            return found

    named = country_named(title or "")
    if named is not None:
        return named

    # A tour whose destination is one of the country's own airport cities is a
    # domestic one, and those cities are Iranian by the same table that reads
    # the origin. Only the destination is read this way, never the title: half
    # the outbound titles name Tehran as where the flight leaves from.
    for word in re.split(r"[^؀-ۿ]+", city or ""):
        if is_a_departure_city(word):
            return Country(name="ایران", latin=COUNTRIES["ایران"])
    return None


# How far up from a city's menu entry the group header can sit. Measured: the
# header is one level up on chakavaktravel (a button over a div of cities) and
# two on radingasht (a country link over its submenu ``ul``). Three is the
# whole depth of a mega menu, and stopping there keeps a page's own headings
# out of the answer.
_MENU_DEPTH = 3

# Shorter than this a menu entry is a word like "تور" or "شب", which matches
# any title by accident.
_SHORTEST_CITY = 3

# A group header is a name, not a paragraph. Past this length the text is the
# menu itself, which names every country the agency sells and so names none.
_HEADER_LENGTH = 40


def _previous_element(node: object) -> object:
    """The node before this one, skipping the whitespace between tags."""
    sibling = getattr(node, "prev", None)
    while sibling is not None and sibling.tag in ("-text", "-undef"):
        sibling = sibling.prev
    return sibling


def country_beside(tree: HTMLParser, names: str) -> Optional[Country]:
    """The country the site's own menu keeps this tour's city under.

    For the sites whose crumb stops at the tour, the answer is still on the
    page: every one of them carries a menu that files cities under countries,
    and it says so in one of two ways. bisanseir links the city at
    ``/tours/oman/muscat`` — the country is the segment above it. chakavaktravel
    and radingasht link it at ``/tours/kusadasi-tours`` and put the country in
    the heading over the group.

    ``names`` is where the tour says it goes — its title and destination joined.
    The menu supplies the vocabulary of cities and the title picks one out of
    it, which is why this still answers on the two pages whose destination
    itself was read wrongly ("اقامت هتل" on bisanseir). The longest matching
    entry wins, so "کوش آداسی" is not read as "آداسی", and a city a tour only
    leaves from is skipped.

    This is the site answering for itself, not a table of ours: an agency that
    moves a city to another country's menu moves this answer with it.

    Example:
        >>> tree = HTMLParser('<ul><li><a href="/tours/oman/muscat">تور مسقط</a>'
        ...                   '</li></ul>')
        >>> country_beside(tree, "تور مسقط ماهان 3 شب")
        Country(name='عمان', latin='oman')
        >>> tree = HTMLParser('<div><button>ترکیه</button>'
        ...                   '<div><a href="/tours/kusadasi-tours">کوش آداسی</a>'
        ...                   '</div></div>')
        >>> country_beside(tree, "تور کوش آداسی | لحظه آخری").latin
        'turkey'
        >>> country_beside(tree, "تور بانکوک") is None
        True
    """
    if not names:
        return None
    matched: list[tuple[int, object]] = []
    for anchor in tree.css("a"):
        city = clean_country_text(anchor.text() or "")
        if len(city) < _SHORTEST_CITY or city not in names:
            continue
        if is_a_departure_city(city):
            continue
        matched.append((len(city), anchor))

    for _, anchor in sorted(matched, key=lambda pair: pair[0], reverse=True):
        path = urlparse(anchor.attributes.get("href") or "").path.split("/")
        for segment in reversed(path[:-1]):
            found = country_slugged(segment)
            if found is not None:
                return found
        node: object = anchor
        for _ in range(_MENU_DEPTH):
            node = getattr(node, "parent", None)
            if node is None:
                break
            header = _previous_element(node)
            text = (header.text() or "").strip() if header is not None else ""
            if 0 < len(text) <= _HEADER_LENGTH:
                found = country_named(clean_country_text(text))
                if found is not None:
                    return found
    return None
