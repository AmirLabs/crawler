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
)

FILLER: tuple[str, ...] = (
    "با پرواز", "پرواز", "نمایشگاه", "ویژه", "روزه", "تور", "شب", "روز", "ماه", "با",
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
    """
    out = normalize_digits(text)
    out = re.sub(r"[A-Za-z]+", " ", out)          # "(VIP Tour)"
    out = re.sub(r"\d+", " ", out)
    out = _SEPARATOR.sub(_MARK, out)
    out = re.sub(r"[()\[\]|]", " ", out)

    for word in sorted(AIRLINES, key=len, reverse=True):
        out = re.sub(re.escape(word), " ", out)
    for month in MONTHS:
        out = re.sub(month, " ", out)
    for season in SEASONS:
        out = re.sub(season, " ", out)
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
