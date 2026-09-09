"""Reading the places a tour goes to out of its own title.

Every tour title measured on the ten target sites names its destination. What
surrounds that name is noise, and the noise falls into four closed vocabularies
— duration, time, airline, marketing — so the destination is what is left after
they are taken away. Searching *for* place names instead would need a list of
every city on earth; subtracting the noise needs four short lists.

Measured over the 23 distinct titles the sites produced (six sites; the other
four yield no title yet because they yield no rows), the rule reads 22. The one
it misses is ``تور نمایشگاه چاپ و تبلیغات گوانگجو`` — an exhibition tour, whose
event name is none of the four noise kinds and so survives untouched.

Two things this module does that a first attempt would not, both of which cost
real accuracy when they were missing:

**Separators are kept, not deleted.** Strip them and two different facts become
the same string::

    کوش آداسی      one place, two words
    بالی اوبود      two places, one word each

The titles they came from were ``تور کوش آداسی 7 روزه`` and ``تور بالی + اوبود``
— the ``+`` was the whole difference. So separators are turned into a marker
early and split on last, rather than removed with the other punctuation.

**The longest noise word goes first.** Removing ``معراج`` before ``معراج ایر``
leaves a bare ``ایر`` behind, which then reads as a destination; that alone
broke three of vanrosh's five titles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from selectolax.parser import HTMLParser

from crawler.extraction.readers.prices import normalize_digits


MONTHS: tuple[str, ...] = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
)
SEASONS: tuple[str, ...] = ("بهار", "تابستان", "پاییز", "پائیز", "زمستان")


AIRLINES: tuple[str, ...] = (
    "ایران ایرتور", "معراج ایر", "کیش ایر", "قشم ایر", "پویا ایر", "ایران ایر",
    "ایرتور", "سپهران", "کاسپین", "ماهان", "معراج", "زاگرس", "وارش", "کارون",
    "تابان", "آسمان", "سپاهان", "چابهار", "اطلس", "آتا", "ترکیش", "قطری",
    # Met while reading the flight blocks, and worth having here too: a carrier
    # left in a title reads as a place. lachinseir sells "تور ... سلام ایر".
    "سلام ایر", "اطلس ایر", "وارش ایر", "قشم ایرلاین",
)

FILLER: tuple[str, ...] = (
    "با پرواز", "پرواز", "نمایشگاه", "ویژه", "روزه", "تور", "شب", "روز", "ماه", "با",
    # How the tour travels, which the title states as readily as where it goes:
    # "تور زمینی تبریز هتل گسترش" gave the destination "زمینی تبریز هتل گسترش".
    # A closed vocabulary like the airlines, and the fact itself is already read
    # into transportation_type, so nothing is lost by taking it out here.
    "هوایی", "زمینی", "قطار", "اتوبوس", "کشتی",
)

_SEPARATOR = re.compile(r"[,\.،؛/\\+\-–—:]+")

_BRACKETED = re.compile(r"[(\[]([^)\]]*)[)\]]")

_MARK = "§"

_SPACE = re.compile(r"[\s‌]+")


@dataclass
class TourPlaces:
    """Where one tour goes, and where it starts.

    ``destinations`` keeps the order the title wrote them in, because a
    multi-city tour reads as a route: "بالی + سنگاپور" is not the same
    advertisement as "سنگاپور + بالی".

    Example:
        >>> places = TourPlaces(destinations=["بالی", "سنگاپور"], origin="تهران")
        >>> places.display
        'بالی + سنگاپور'
    """

    destinations: list[str] = field(default_factory=list)
    origin: Optional[str] = None

    @property
    def display(self) -> Optional[str]:
        """The destinations as one string, for showing rather than searching.

        Example:
            >>> TourPlaces(destinations=["کیش"]).display
            'کیش'
            >>> TourPlaces().display is None
            True
        """
        return " + ".join(self.destinations) if self.destinations else None


_WHOLE_WORD = r"(?<![\w؀-ۿ]){word}(?![\w؀-ۿ])"
_A_MONTH = r"(?<![\w؀-ۿ]){month}(?:\s*ماه)?(?![\w؀-ۿ])"


def _strip_noise(text: str) -> str:
    """Take the four noise vocabularies out, leaving separators as markers.

    Order matters twice over: the longest airline first so ``معراج ایر`` does
    not decay to ``ایر``, and the numbers before the month so ``13 شهریور``
    goes whole rather than leaving a stray day behind.

    Example:
        >>> _strip_noise("تور ازمیر 6 شب - ویژه  شهریور ماه 1405 ( کاسپین )")
        'ازمیر'
        >>> _strip_noise("تور بالی + اوبود 7 شب")
        'بالی§اوبود'
        >>> _strip_noise("تور ترکیه استانبول 4 روزه از مهرآباد")
        'ترکیه استانبول'
        >>> _strip_noise("تور ازمیر ویژه مهرماه")
        'ازمیر'
    """
    out = normalize_digits(strip_leaving_clause(text))
    out = re.sub(r"[A-Za-z]+", " ", out)          # "(VIP Tour)"
    out = re.sub(r"\d+", " ", out)
    out = _SEPARATOR.sub(_MARK, out)
    out = re.sub(r"[()\[\]|]", " ", out)

    # Every vocabulary is taken out on word boundaries. Without them a month
    # name is removed from inside a word: "از مهرآباد" became "از آباد", and
    # nahalgasht's destination read "ترکیه استانبول از آباد". A month may still
    # carry "ماه" behind it, glued or spaced, because that is how these titles
    # write it — "مهرماه" goes whole, "مهرآباد" is left alone.
    for word in sorted(AIRLINES, key=len, reverse=True):
        out = re.sub(_WHOLE_WORD.format(word=re.escape(word)), " ", out)
    for month in MONTHS:
        out = re.sub(_A_MONTH.format(month=month), " ", out)
    for season in SEASONS:
        out = re.sub(_WHOLE_WORD.format(word=season), " ", out)
    for word in sorted(FILLER, key=len, reverse=True):
        out = re.sub(rf"(?<![\w؀-ۿ]){re.escape(word)}(?![\w؀-ۿ])",
                     " ", out)

    # "و" joins two places as often as it joins two nouns, so it becomes a
    # marker like any other separator — but only now, after "6 شب و 7 روز" has
    # already lost the words that made it a duration.
    out = re.sub(r"(?<![\w؀-ۿ])و(?![\w؀-ۿ])", _MARK, out)

    out = _SPACE.sub(" ", out)
    out = re.sub(rf"\s*{_MARK}\s*", _MARK, out)
    return out.strip().strip(_MARK)


def _split(text: str) -> list[str]:
    """The places in a stripped string, in the order they were written.

    Example:
        >>> _split("مسکو§سنت پترزبورگ")
        ['مسکو', 'سنت پترزبورگ']
        >>> _split("کوش آداسی")
        ['کوش آداسی']
    """
    return [part.strip() for part in text.split(_MARK) if part.strip()]


def _inner_groups(title: str) -> list[str]:
    """The bracketed parts of a title, pipes included.

    A pipe is a separator rather than a pair on vanrosh, which leaves one
    dangling, so everything after the first pipe counts as inside.

    Example:
        >>> _inner_groups("تور مالزی (کوالالامپور و لنکاوی) شهریور 1405")
        ['کوالالامپور و لنکاوی']
        >>> _inner_groups("تور روسیه |مسکو،سنت پترزبورگ| تابستان 1405 |")
        ['مسکو،سنت پترزبورگ', ' تابستان 1405 ', '']
    """
    groups = _BRACKETED.findall(title)
    if "|" in title:
        groups.extend(title.split("|")[1:])
    return groups


def read_title_places(title: str) -> TourPlaces:
    """Read the destinations a title names, or none if it names nothing.

    A bracket holds four different things across these sites — the city list,
    the airline, the night count, the marketing line — so it cannot be trusted
    on sight. It is trusted only once the other three are stripped and
    something is still standing: at that point the name *outside* the bracket
    is the country and the ones inside are the cities the tour actually sleeps
    in, which is the grain the client's hotel list needs.

    Returns an empty result rather than a guess when nothing survives, so the
    fill rate shows the gap instead of the output carrying an invention.
                
    Example:
        >>> read_title_places("تور بالی + اوبود 7 شب -ویژه شهریور 1405 ( ایران ایرتور )")
        TourPlaces(destinations=['بالی', 'اوبود'], origin=None)
        >>> read_title_places("تور مالزی (کوالالامپور و لنکاوی) شهریور 1405")
        TourPlaces(destinations=['کوالالامپور', 'لنکاوی'], origin=None)
        >>> read_title_places("تور کوش آداسی 7 روزه با پرواز کاسپین")
        TourPlaces(destinations=['کوش آداسی'], origin=None)
        >>> read_title_places("تور تابستان 1405")
        TourPlaces(destinations=[], origin=None)
    """
    if not title or not title.strip():
        return TourPlaces()

    for group in _inner_groups(title):
        inside = _split(_strip_noise(group))
        if inside:
            return TourPlaces(destinations=inside)

    return TourPlaces(destinations=_split(_strip_noise(title)))


# The airports a tour can leave Iran from, and the city each one serves. A
# closed table of facts about the country, not a rule about any site: the seven
# cities below are every one of these agencies' departure points, and a table
# keyed on them keeps working when a site redraws its pages.
#
# It is here because a page names the airport far more often than the city.
# nahalgasht writes "پرواز از فرودگاه مهرآباد" and never says Tehran; vanrosh
# and tahagasht write the IATA code beside the city; chakavaktravel prints the
# city and the airport back to back. One table reads all four.
_IRANIAN_AIRPORTS: dict[str, str] = {
    "امام خمینی": "تهران",
    "مهرآباد": "تهران",
    "IKA": "تهران",
    "THR": "تهران",
    "مشهد": "مشهد",
    "MHD": "مشهد",
    "شیراز": "شیراز",
    "SYZ": "شیراز",
    "اصفهان": "اصفهان",
    "IFN": "اصفهان",
    "تبریز": "تبریز",
    "TBZ": "تبریز",
    "کیش": "کیش",
    "KIH": "کیش",
    # The rest of the country's airports. Added after a run read nahalgasht's
    # "تور هوایی مشهد از اهواز" as a destination of "مشهد از اهواز": the clause
    # is only taken off a title where the place in it is known to be a place a
    # tour leaves from, and the six cities above were not the whole list.
    "اهواز": "اهواز",
    "AWZ": "اهواز",
    "کرمان": "کرمان",
    "KER": "کرمان",
    "رشت": "رشت",
    "RAS": "رشت",
    "ارومیه": "ارومیه",
    "OMH": "ارومیه",
    "بندرعباس": "بندرعباس",
    "BND": "بندرعباس",
    "یزد": "یزد",
    "AZD": "یزد",
    "کرمانشاه": "کرمانشاه",
    "KSH": "کرمانشاه",
    "ساری": "ساری",
    "SRY": "ساری",
    "قشم": "قشم",
    "GSM": "قشم",
    "اردبیل": "اردبیل",
    "ADU": "اردبیل",
    "همدان": "همدان",
    "HDM": "همدان",
    "بوشهر": "بوشهر",
    "BUZ": "بوشهر",
    "زاهدان": "زاهدان",
    "ZAH": "زاهدان",
    "کرج": "کرج",
    "قم": "قم",
    "اصفهان ": "اصفهان",
}

_ORIGIN_CITIES = "|".join(
    sorted(set(_IRANIAN_AIRPORTS.values()), key=len, reverse=True)
)
_GAP_ = r"[\s‌]*"

# The airport is the one the tour leaves from when the page says so in one of
# three ways. Naming it anywhere on the page is not enough: lachinseir offers
# "خدمات رایگان CIP فرودگاه امام خمینی", which is a lounge and not a departure,
# and reading that as the origin would put a city on a page that never stated
# one.
_LEAVES_FROM = re.compile(r"از" + _GAP_ + r"(?:فرودگاه" + _GAP_ + r")?$")
_BESIDE_CITY = re.compile(
    r"(?:" + _ORIGIN_CITIES + r")" + _GAP_
    + r"(?:فرودگاه" + _GAP_ + r")?(?:بین" + _GAP_ + r"المللی" + _GAP_ + r")?$"
)
_AIRPORT_CODE = re.compile(r"\(?\b(?:IKA|THR|MHD|SYZ|IFN|TBZ|KIH)\b\)?")

# How much text on either side of the airport is read to judge it. Wide enough
# for "تهران , فرودگاه بین‌المللی امام خمینی", narrow enough that the sentence
# before it is somebody else's.
_ORIGIN_WINDOW = 30


_DEPARTURE_CITIES = frozenset(_IRANIAN_AIRPORTS.values())


def is_a_departure_city(name: str) -> bool:
    """Whether a place is one of the cities these tours leave Iran from.

    Public because the destination rules need the same answer the origin rule
    does — a leg that starts at home is not where the tour goes.

    Example:
        >>> is_a_departure_city("تهران"), is_a_departure_city("بانکوک")
        (True, False)
    """
    return name in _DEPARTURE_CITIES


def find_origin(text: str) -> Optional[str]:
    """The city a tour leaves from, where the page says which airport it uses.

    Returns ``None`` rather than Tehran when the page is silent. Most of these
    tours do leave Tehran, but a default would be invisible in the fill rate
    and wrong the day an agency sells a Mashhad departure.

    Example:
        >>> find_origin("پرواز از فرودگاه مهرآباد میباشد.")
        'تهران'
        >>> find_origin("تهران , فرودگاه بین‌المللی امام خمینی IKA مدت پرواز")
        'تهران'
        >>> find_origin("همراه با خدمات رایگان CIP فرودگاه امام خمینی") is None
        True
        >>> find_origin("ترانسفر فرودگاهی و بیمه مسافرتی") is None
        True
    """
    for airport, city in _IRANIAN_AIRPORTS.items():
        for match in re.finditer(re.escape(airport), text):
            before = text[max(0, match.start() - _ORIGIN_WINDOW): match.start()]
            after = text[match.end(): match.end() + 14]
            if _LEAVES_FROM.search(before):
                return city
            if _AIRPORT_CODE.search(after) or _AIRPORT_CODE.search(before[-8:]):
                return city
            if _BESIDE_CITY.search(before):
                return city
    return None


# "تور استانبول از مهرآباد" names where the tour goes and where it leaves from,
# and only the first is the destination. Built from the table above, so it
# covers the airport and the city alike and grows with it. Anchored on "از" and
# on a word boundary: a place whose own name begins with one of these words is
# not an origin clause.
_LEAVING_CLAUSE = re.compile(
    r"(?<![\w؀-ۿ])از[\s‌]*(?:فرودگاه[\s‌]*)?(?:"
    + "|".join(
        sorted(
            {re.escape(k) for k in _IRANIAN_AIRPORTS}
            | {re.escape(v) for v in _IRANIAN_AIRPORTS.values()},
            key=len,
            reverse=True,
        )
    )
    + r")(?![\w؀-ۿ])"
)


def strip_leaving_clause(title: str) -> str:
    """A title with the "from <Iranian airport>" part taken off.

    That clause says where the tour starts, and leaving it in put the airport
    in the destination: nahalgasht sells "تور ترکیه استانبول 4 روزه از مهرآباد"
    and the destination read "ترکیه استانبول از مهرآباد".

    Only the places the origin table already knows are stripped, so a foreign
    city is never mistaken for a departure point.

    Example:
        >>> strip_leaving_clause("تور ترکیه استانبول 4 روزه از مهرآباد")
        'تور ترکیه استانبول 4 روزه'
        >>> strip_leaving_clause("تور مشهد از شیراز")
        'تور مشهد'
        >>> strip_leaving_clause("تور استانبول 4 روزه")
        'تور استانبول 4 روزه'
    """
    return _SPACE.sub(" ", _LEAVING_CLAUSE.sub(" ", title)).strip()


# What marks an element as naming an airport: the word itself, or a three-letter
# code standing alone. Both are needed — chakavaktravel writes only the word,
# tahagasht writes only the code beside the city, vanrosh writes both.
_AIRPORT_MARKER = re.compile(r"فرودگاه|(?<![A-Za-z])[A-Z]{3}(?![A-Za-z])")

# A flight line is short. Anything longer is a paragraph that mentions an
# airport, and the city beside it belongs to a sentence, not to a route.
_MAX_FLIGHT_CHARS = 220

# How far the comma that introduces an airport may sit from the airport itself.
# Wide enough for "ازمیر , عدنان مندرس ADB", where the airport's own name comes
# between; narrow enough to refuse a comma that belongs to another sentence.
_COMMA_TO_AIRPORT = 26

# Three-letter words that are not airports. A meal plan in brackets reads
# exactly like a code beside a city — vanrosh prints "هتل 4 ستاره BABAYLON
# (ALL)" — and taking it turned a hotel into a destination.
_NOT_AN_AIRPORT = {"ALL", "CIP", "VIP", "PDF", "USD", "EUR", "IRR", "IRT",
                   "NEW", "TOP", "SPA", "THE", "AND", "FOR", "NAN"}

# A city writes its own name and nothing else: Persian letters and spaces, no
# digit and no Latin. That one condition is what tells "تهران (IKA)" from
# "هتل 4 ستاره BABAYLON (ALL)", and it needs no list of hotels to do it.
_ONLY_A_NAME = re.compile(r"^[آ-ی][آ-ی‌\s]{1,24}$")

# "باکو (GYD)" — the city and its code in one element, which is how a flight
# table headed by times writes it.
_CITY_THEN_CODE = re.compile(r"^\s*([^()]{2,34}?)\s*\(([A-Z]{3})\)\s*$")

# "ازمیر , عدنان مندرس ADB" and "... ساعت : 22:00 تهران , فرودگاه ... IKA".
# The city is what ends the part before the comma: everything after it names
# the airport, and "عدنان مندرس" is an airport's name, not a city's.
_TRAILING_WORDS = re.compile(r"([آ-ی][آ-ی‌]*(?:\s+[آ-ی][آ-ی‌]*)?)\s*$")


def _cities_beside(own: str, previous: Optional[str]) -> list[str]:
    """Every city the airports on one line belong to, by whichever shape it uses.

    A line may carry both ends of a flight — vanrosh prints the outbound and
    the return in one paragraph — so every marker on it is read, not the first.

    Example:
        >>> _cities_beside("باکو (GYD)", None)
        ['باکو']
        >>> _cities_beside("ساعت : 22:00 تهران , فرودگاه امام خمینی IKA"
        ...                " مدت پرواز : 03:30 ازمیر , عدنان مندرس ADB", None)
        ['تهران', 'ازمیر']
        >>> _cities_beside("فرودگاه بین المللی کوالالامپور", "کوالالامپور")
        ['کوالالامپور']
        >>> _cities_beside("هتل 4 ستاره BABAYLON (ALL)", None)
        []
    """
    match = _CITY_THEN_CODE.match(own)
    if match and match.group(2) not in _NOT_AN_AIRPORT:
        found = match.group(1).strip()
        if _ONLY_A_NAME.match(found) and "فرودگاه" not in found:
            return [found]

    # The city sits just before the comma that introduces the airport. The two
    # have to be close: liyaseir writes "تحویل اتاق‌ها، ترانسفر زمینی ... به
    # فرودگاه" in its itinerary, where the comma is a sentence away and the
    # words before it are not a place at all.
    cities: list[str] = []
    for marker in _AIRPORT_MARKER.finditer(own):
        if marker.group(0) in _NOT_AN_AIRPORT:
            continue
        head = own[: marker.start()]
        comma = max(head.rfind(","), head.rfind("،"))
        if comma < 0 or len(head) - comma > _COMMA_TO_AIRPORT:
            continue
        words = _TRAILING_WORDS.search(head[:comma])
        if words is None:
            continue
        city = words.group(1).strip()
        if _ONLY_A_NAME.match(city) and "فرودگاه" not in city and city not in cities:
            cities.append(city)
    if cities:
        return cities

    # The previous element is only the city when this one is the airport and
    # nothing else. nahalgasht writes "استفاده از سالن CIP لانژ فرودگاه
    # مهرآباد" in the middle of its services, and the element before that is
    # "بیمه مسافرتی" — a sentence away from being a city.
    if previous and own.startswith("فرودگاه") and _ONLY_A_NAME.match(previous.strip()):
        return [previous.strip()]
    return []


def read_route(html: str) -> list[str]:
    """Every city the page's flight lines name, in the order they are flown.

    Read from the markup rather than from the page's text. Flattened, a flight
    box reads "تهرانتهرانفرودگاه امام خمینیبار مجاز: 30اکونومیکوالالامپور" and
    the city cannot be told from what surrounds it; in the markup each part sits
    in its own element.

    Example:
        >>> read_route('<body><ul><li>تهران</li><li>فرودگاه امام خمینی</li></ul>'
        ...            '<ul><li>کوالالامپور</li>'
        ...            '<li>فرودگاه بین المللی کوالالامپور</li></ul></body>')
        ['تهران', 'کوالالامپور']
    """
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript"):
        node.decompose()

    route: list[str] = []
    for node in tree.css("*"):
        own = " ".join((node.text(deep=False) or "").split())
        if not own or len(own) > _MAX_FLIGHT_CHARS:
            continue
        if not _AIRPORT_MARKER.search(own):
            continue
        previous = node.prev
        while previous is not None and previous.tag in ("-text", "-undef"):
            previous = previous.prev
        earlier = (
            " ".join((previous.text(deep=True) or "").split())
            if previous is not None else None
        )
        for city in _cities_beside(own, earlier):
            if city not in route:
                route.append(city)
    return route


def find_destination(html: str) -> Optional[str]:
    """Where the page's own flight lines say the tour goes.

    The route names both ends and the airport table tells them apart: the leg
    that does not leave from an Iranian airport is the one abroad. That is what
    makes this safe without trusting the order the page prints them in.

    Returns ``None`` where every city on the route is Iranian — a domestic tour
    cannot be told apart this way, and guessing would be worse than the title
    rule the caller falls back to.

    Example:
        >>> find_destination('<body><ul><li>تهران</li><li>فرودگاه امام خمینی</li></ul>'
        ...                  '<ul><li>کوالالامپور</li>'
        ...                  '<li>فرودگاه بین المللی کوالالامپور</li></ul></body>')
        'کوالالامپور'
    """
    abroad = [city for city in read_route(html) if not is_a_departure_city(city)]
    return abroad[0] if abroad else None


# "پرواز از بانکوک به کوالالامپور" — a leg the tour flies between its stops.
# The connective is kept out of the name: liyaseir writes "پرواز از تهران و
# ورود به سئول", where "و ورود" belongs to the sentence and not to the city.
_FLIGHT_LEG = re.compile(
    r"پرواز[\s‌]*از[\s‌]+([آ-ی][آ-ی‌\s]{1,18}?)"
    r"[\s‌]+(?:و[\s‌]+\S+[\s‌]+)?(?:به|تا)[\s‌]+(?:سمت[\s‌]+)?"
    r"([آ-ی][آ-ی‌\s]{1,18}?)(?=[\s،,.]|$)"
)


def read_flight_legs(text: str) -> list[str]:
    """Every city the page says the tour flies between, in the order flown.

    This is where a tour that never names its destination still says it.
    lachinseir sells "تور دور آسیا ویژه نوروز" — a title with no place in it at
    all — and then lists "پرواز از بانکوک به کوالالامپور ، پرواز از کوالالامپور
    به بالی". Those three cities are the tour.

    Read from the page's own words rather than from a flight box, because the
    sites that write this have no flight box; the ones that do are answered by
    :func:`read_route` before this is reached.

    Example:
        >>> read_flight_legs("پرواز از بانکوک به کوالالامپور ، "
        ...                  "پرواز از کوالالامپور به بالی")
        ['بانکوک', 'کوالالامپور', 'بالی']
        >>> read_flight_legs("پرواز از تهران و ورود به سئول")
        ['تهران', 'سئول']
        >>> read_flight_legs("ترانسفر فرودگاهی و بیمه مسافرتی")
        []
    """
    cities: list[str] = []
    for match in _FLIGHT_LEG.finditer(text):
        for city in (match.group(1).strip(), match.group(2).strip()):
            if _ONLY_A_NAME.match(city) and city not in cities:
                cities.append(city)
    return cities


# Where a hotel's name starts inside a destination. A place is never called
# "هتل", so once that word appears the rest of the string belongs to the hotel
# — and this needs no help from the row, which is what makes it work where
# matching the name did not: nahalgasht's title says "هتل هما" and its row says
# "هتل هما 1", "هتل مجلل درویشی" against "هتل درویشی".
_HOTEL_STARTS = re.compile(r"[\s§+]*(?:هتل|اقامتگاه|هاستل|مهمانپذیر)\b.*$")


def without_a_hotel_word(display: Optional[str]) -> Optional[str]:
    """A destination cut short at the word that starts a hotel's name.

    Example:
        >>> without_a_hotel_word("مشهد هتل مجلل درویشی")
        'مشهد'
        >>> without_a_hotel_word("مشهد هتل هما")
        'مشهد'
        >>> without_a_hotel_word("کوالالامپور")
        'کوالالامپور'
        >>> without_a_hotel_word("هتل گسترش") is None
        True
    """
    if not display:
        return display
    return _SPACE.sub(" ", _HOTEL_STARTS.sub("", display)).strip() or None


def without_the_hotel(display: Optional[str], hotel: Optional[str]) -> Optional[str]:
    """A destination with the hotel's own name taken out of it.

    Some titles name the hotel as well as the place — "تور زمینی تبریز هتل
    گسترش" — and subtracting noise cannot know which words are the hotel's. It
    does not have to: the page states the hotel elsewhere and step 3 has already
    read it, so the name can simply be removed from what is left.

    This is the one piece of noise that no vocabulary could ever cover, because
    hotels are named after everything.

    Example:
        >>> without_the_hotel("تبریز هتل گسترش", "هتل گسترش")
        'تبریز'
        >>> without_the_hotel("مشهد هتل مجلل درویشی", "هتل مجلل درویشی")
        'مشهد'
        >>> without_the_hotel("کوالالامپور", "هتل ASTO")
        'کوالالامپور'
        >>> without_the_hotel("هتل گسترش", "هتل گسترش") is None
        True
    """
    if not display or not hotel:
        return display
    left = display
    for name in (hotel, *hotel.split(" + ")):
        if name and name in left:
            left = left.replace(name, " ")
    left = _SPACE.sub(" ", left).strip(" §" + "".join("+-,،"))
    return left or None


# The letters that are written two ways in Persian, and how they are folded
# before two place names are compared. A page writes "اصفهان" and a menu writes
# "أصفهان"; a plain == says they are different cities.
_SAME_LETTER = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ي": "ی", "ى": "ی",
    "ك": "ک",
    "ۀ": "ه", "ة": "ه",
    "ؤ": "و",
    "‌": " ",  # ZWNJ: "بندر عباس" and "بندرعباس" are one place
    "ً": "", "ٌ": "", "ٍ": "", "َ": "",
    "ُ": "", "ِ": "", "ّ": "", "ْ": "",
})


def fold_name(name: str) -> str:
    """A place name reduced to the form two spellings of it share.

    Only for comparing, never for output — what the client is sent is what the
    page wrote.

    Example:
        >>> fold_name("أصفهان"), fold_name(" بندر‌عباس ")
        ('اصفهان', 'بندر عباس')
        >>> fold_name("كيش") == fold_name("کیش")
        True
    """
    folded = (name or "").translate(_SAME_LETTER)
    return " ".join(folded.split())


# The cities of Iran a tour can be sold to. Wider than the airport table above,
# because a domestic tour goes to places with no airport — Kalardasht is sold
# by road — and this list answers a different question: not "can a tour leave
# from here" but "is this place inside the country".
#
# A closed list of the country's own cities, the way the airport table is: it
# is a fact about Iran, not a rule about a site. It only has to hold the places
# these ten agencies actually sell, and it is only consulted when the country
# could not be read at all.
_IRANIAN_CITIES: frozenset[str] = frozenset(
    fold_name(city) for city in (
        *_IRANIAN_AIRPORTS.values(),
        "انزلی", "بندر انزلی", "رامسر", "چالوس", "نوشهر", "کلاردشت", "متل قو",
        "سرعین", "ماسال", "لاهیجان", "گرگان", "بابلسر", "محمودآباد", "فریدون کنار",
        "کاشان", "شاهرود", "سمنان", "دامغان", "طبس", "خور", "مصر", "گرمه",
        "خرم آباد", "سنندج", "سقز", "مریوان", "ایلام", "یاسوج", "شهرکرد",
        "بیرجند", "زابل", "چابهار", "کنارک", "بندر لنگه", "هرمز", "هنگام",
        "خارک", "عسلویه", "دزفول", "شوش", "شوشتر", "آبادان", "خرمشهر",
        "ماهشهر", "اراک", "خمین", "ساوه", "زنجان", "قزوین", "الموت",
        "خلخال", "پارس آباد", "مشکین شهر", "بجنورد", "شیروان", "قوچان",
        "نیشابور", "سبزوار", "تربت حیدریه", "کلات", "مراغه", "مرند", "جلفا",
        "خوی", "ماکو", "سلماس", "میاندوآب", "بوکان", "مهاباد", "پیرانشهر",
        "نجف آباد", "خمینی شهر", "شاهین شهر", "کاشمر", "فردوس", "رفسنجان",
        "بم", "جیرفت", "سیرجان", "زرند", "شهربابک", "مرودشت", "کازرون",
        "لار", "جهرم", "فسا", "داراب", "نی ریز", "آباده", "اقلید",
        "بندر ترکمن", "کردکوی", "علی آباد", "مینودشت", "آزادشهر",
        "تنکابن", "عباس آباد", "سلمان شهر", "رویان", "آمل", "بابل", "قائم شهر",
        "بهشهر", "نکا", "جویبار", "سوادکوه", "پل سفید", "فیروزکوه", "دماوند",
        "طالقان", "شهرکرد", "سمیرم", "ابیانه", "نطنز", "اردستان", "میمه",
    )
)


def is_an_iranian_city(name: str) -> bool:
    """Whether a place is a city inside Iran.

    Answers "does this tour stay in the country" where the country itself could
    not be read. A place the list does not know is not called foreign here —
    the caller decides what to do with a no, and the list is short by design.

    Example:
        >>> is_an_iranian_city("مشهد"), is_an_iranian_city("كيش")
        (True, True)
        >>> is_an_iranian_city("آنتالیا")
        False
    """
    return fold_name(name) in _IRANIAN_CITIES


# What the client's output calls these cities in latin letters. Only the ones a
# tour actually leaves from: origin_city_name_en has no other source, and a
# destination's latin name comes from the site's own slug instead.
CITY_LATIN: dict[str, str] = {
    "تهران": "tehran", "مشهد": "mashhad", "شیراز": "shiraz",
    "اصفهان": "isfahan", "تبریز": "tabriz", "کیش": "kish", "قشم": "qeshm",
    "اهواز": "ahvaz", "کرمان": "kerman", "رشت": "rasht", "ارومیه": "urmia",
    "بندرعباس": "bandar-abbas", "یزد": "yazd", "کرمانشاه": "kermanshah",
    "ساری": "sari", "اردبیل": "ardabil", "همدان": "hamedan",
    "بوشهر": "bushehr", "زاهدان": "zahedan", "کرج": "karaj", "قم": "qom",
}

_LATIN_BY_FOLDED: dict[str, str] = {
    fold_name(city): latin for city, latin in CITY_LATIN.items()
}


def latin_city(name: str) -> Optional[str]:
    """The latin spelling of a city a tour leaves from, if we know it.

    Example:
        >>> latin_city("تهران"), latin_city("كيش")
        ('tehran', 'kish')
        >>> latin_city("بانکوک") is None
        True
    """
    return _LATIN_BY_FOLDED.get(fold_name(name or ""))
