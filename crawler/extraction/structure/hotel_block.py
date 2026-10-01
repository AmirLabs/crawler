"""Reading the hotel a set of rows belongs to, from the block that encloses them.

Some of what an offer needs is not printed inside the offer. bisanseir.com
lists a hotel once and then a table of room prices beneath it; nahalgasht.com
does the same with a card. The rows carry the prices, and the hotel's name,
star rating and board type sit one or more levels above them::

    <div class="hotel">
        <a>The Train Resort</a>              ← name
        <span>خدمات:</span><span>BB</span>   ← board
        <div class="hotel__beds">
            <div class="hotel__bed">دو تخته 82,000,000 تومان</div>   ← a row
            <div class="hotel__bed">یک تخته 82,000,000 تومان</div>   ← a row
        </div>
    </div>

Two things make this readable without a per-site rule:

* **Which block.** Walking up from the rows, the first ancestor that adds text
  of its own — text the rows do not already contain — is the block describing
  them. That lands on ``div.hotel`` for bisanseir.com and ``div.reserve`` for
  nahalgasht.com, two different depths reached by one rule.
* **Which text is the name.** Within that block, the hotel's name comes first,
  ahead of the labels. So the first text that is not a price, a room type or a
  label the other readers already recognise is the name.

Everything found here belongs to *every* row inside the block, which is what
makes it safe to hand down: the rows are the room choices for this one hotel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from selectolax.parser import HTMLParser, Node

from crawler.extraction.structure.html_rows import RowContainer, _element_path
from crawler.extraction.readers.labels import (
    BOARD_CODES,
    detect_board_type,
    detect_nights,
    detect_room_type,
    detect_star_rating,
)
from crawler.extraction.readers.prices import CURRENCY_WORDS, find_prices

# How far above the rows to look before giving up. Beyond this the ancestors
# hold the whole page, and their first text is a menu item.
_MAX_LEVELS = 5

# A name shorter than this is a fragment; longer than this is a sentence. The
# upper bound has to clear the longest real name on these sites, which is
# "Park Inn by Radisson Pulkovskaya Hotel & Conference Centre St Petersburg"
# at 71 characters.
_MIN_NAME_CHARS = 3
_MAX_NAME_CHARS = 80

# How much a hotel block may say beyond its rows. Measured again once blocks
# naming several hotels were being read: a real block runs 58 to 281 characters
# — one to three names with their labels — while the page sections that merely
# mention a hotel start at 325 and climb to 3429. The gap between 281 and 325 is
# empty, so the cap sits in it.
#
# A stay of four or more hotels would pass 300 and be missed. None appears on
# the target sites today, and a cap that stayed ahead of that would reach into
# the sections, so this is a known limit rather than an oversight.
_MAX_BLOCK_CHARS = 300

# Text that sits beside a hotel name and would otherwise be mistaken for one.
# This list is a guard, not the mechanism: the name is found by position, and
# these only matter on a page that prints a label before the name.
_NOT_A_NAME = (
    "موقعیت مکانی",
    "خدمات",
    "رزرو",
    "مشاهده",
    "دانلود",
    "جزئیات",
    "پیشنهاد ویژه",
    "تکمیل ظرفیت",
    "امتیاز",
    # "قیمتهای هتل" closes a bisanseir block and names a hotel word, so it
    # survives every other test.
    "قیمت",
    "خرید",
    # The meal wording arrives split from its code — "با صبحانه" in one element
    # and "(BB)" in the next — so detect_board_type cannot see it whole.
    "صبحانه",
    # lachinseir closes its hotel list with the payment terms and the currency
    # the prices are quoted in, and both read as two more hotels.
    "شرایط",
    "اقساط",
)

_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# What makes an ancestor a hotel block rather than merely a block. Without this
# test the walk stops at whichever ancestor happens to add a word, and the
# first fragment inside it becomes a "hotel name" — a page listing departure
# dates yielded "رفت 07 شهریور" and one with a read-more heading yielded
# "بیشتر بخوانیــد". Both look like data and are not, which is worse than
# reporting nothing, so a block has to mention lodging to qualify.
# What is left of a fragment once its digits are gone, when the fragment was
# never a name: "3 ستاره", "7 شب اقامت", "6 شب و 7 روز". Matched whole, so
# "هتل پارسه 3 ستاره" keeps its name and is kept.
_ONLY_A_DURATION = re.compile(r"(?:ستاره|شب|روز|اقامت|و)*")

# What a site prints *about* a hotel in the same breath as its name: the grade
# and the meal plan. Both already have a column of their own, so carrying them
# in the name says the same thing twice — vanrosh writes
# "هتل 4 ستاره (ALL) BABAYLON" where the name is BABAYLON.
_A_GRADE = re.compile(r"[\d۰-۹٠-٩]+" + r"[\s\u200c]*" + r"ستاره")
_A_BOARD = re.compile(
    rf"\(?\s*(?<![A-Za-z])(?:{BOARD_CODES})(?![A-Za-z])\s*\)?", re.I,
)
# What is left when a fragment described a hotel without ever naming one.
_NAMES_NOTHING = re.compile(r"(?:هتل|اقامتگاه|هاستل|مهمانپذیر|ویلا|hotel|resort|hostel)*", re.I)

_HOTEL_WORDS = (
    "هتل",
    "اقامتگاه",
    "هاستل",
    "مهمانپذیر",
    "ویلا",
    "hotel",
    "resort",
    "residence",
    "hostel",
    "apartment",
)


def _clean(text: Optional[str]) -> str:
    """Collapse whitespace, including the double spaces these pages leave in.

    Example:
        >>> _clean("  هتل   صفی خانی ")
        'هتل صفی خانی'
    """
    return " ".join((text or "").split())


def _extra_text(node: Node, row_texts: list[str]) -> str:
    """What ``node`` says beyond what its rows already said.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser('<div><b>The Train Resort</b><i>دو تخته</i></div>')
        >>> _extra_text(tree.css_first("div"), ["دو تخته"])
        'The Train Resort'
    """
    text = _clean(node.text(deep=True))
    for row_text in row_texts:
        if row_text:
            text = text.replace(row_text, " ", 1)
    return _clean(text)


def _describes_a_hotel(text: str) -> bool:
    """Whether a block's own text is about lodging at all.

    A star rating or a board code counts as well as the word itself: some
    pages name the hotel in Latin script and rate it in Persian, so requiring
    the word "هتل" alone would miss them.

    Example:
        >>> _describes_a_hotel("هتل صفی خانی موقعیت مکانی 2 شب")
        True
        >>> _describes_a_hotel("The Train Resort خدمات: BB")
        True
        >>> _describes_a_hotel("انتخاب تاریخ تور")
        False
        >>> _describes_a_hotel("2 شب 3 شب")
        False
    """
    low = text.lower()
    if any(word in low for word in _HOTEL_WORDS):
        return True
    return detect_star_rating(text) is not None or detect_board_type(text) is not None


def _looks_like_a_name(text: str) -> bool:
    """Whether a fragment could be a hotel's name rather than a label.

    Example:
        >>> _looks_like_a_name("The Train Resort")
        True
        >>> _looks_like_a_name("خدمات:")
        False
        >>> _looks_like_a_name("دو تخته")
        False
        >>> _looks_like_a_name("82,000,000 تومان")
        False
        >>> _looks_like_a_name("BB با صبحانه")
        False
        >>> _looks_like_a_name("تور دبی")
        False
        >>> _looks_like_a_name("7 شب اقامت")
        False
        >>> _looks_like_a_name("شهر: کوالالامپور")
        False
        >>> _looks_like_a_name("هتل ASTO")
        True
        >>> _looks_like_a_name("دلار")
        False
        >>> _looks_like_a_name("هتل 4 ستاره")
        False
        >>> _looks_like_a_name("هتل 4 ستاره (ALL) BABAYLON")
        True
    """
    if not (_MIN_NAME_CHARS <= len(text) <= _MAX_NAME_CHARS):
        return False
    if not _HAS_LETTER.search(text):
        return False
    if any(label in text for label in _NOT_A_NAME):
        return False
    if find_prices(text) or detect_room_type(text):
        return False
    # A board code standing on its own is a label. It only has to be excluded
    # now that every name in a block is collected rather than the first one:
    # "BB با صبحانه" is printed after each hotel, and the old walk stopped
    # before ever reaching it.
    if detect_board_type(text) is not None and not any(
        word in text.lower() for word in _HOTEL_WORDS
    ):
        return False
    # One lowercase word naming no lodging is a marker, not a hotel. bisanseir
    # prints "land" beside every ground tour. Hotels on these sites are always
    # several words or carry the word itself, so nothing real is lost.
    if len(text.split()) == 1 and text.islower() and not any(
        word in text.lower() for word in _HOTEL_WORDS
    ):
        return False
    # A tour is not a hotel. tahagasht heads its page "تور دبی هتل آدرس بیچ
    # ریزورت", which names lodging and so passes every test above, and the
    # first thing printed under it is the tour's own name.
    if text.split()[0] == "تور":
        return False
    # A colon separates a caption from its value. chakavaktravel prints
    # "شهر: کوالالامپور" beside the hotel it belongs to, and that read as a
    # second hotel and was joined to the first. A hotel does not punctuate
    # its own name.
    if ":" in text or "：" in text:
        return False
    # A currency standing on its own is what the prices are quoted in, not a
    # hotel. It survives every other test because it carries no digit.
    if text.strip() in CURRENCY_WORDS:
        return False
    # A fragment that is nothing but a rating or a duration is a label, but one
    # that carries them alongside a name ("هتل پارسه 3 ستاره") is still a name.
    # "7 شب اقامت" is the whole of the stay's length and none of its name.
    stripped = re.sub(r"[\d۰-۹٠-٩\s:،,.]+", "", text)
    if _ONLY_A_DURATION.fullmatch(stripped):
        return False
    # A fragment that grades a hotel without naming one is not a name.
    # lachinseir prints "هتل 4 ستاره" where it has no name to print, and
    # sending that on says the tour sleeps at a hotel called "هتل 4 ستاره".
    if not _NAMES_NOTHING.fullmatch(re.sub(r"[^\w]|[\d۰-۹٠-٩_]", "", name_only(text))):
        return True
    return False


def name_only(text: str) -> str:
    """A hotel's name with the grade and meal plan printed beside it removed.

    Both have a column of their own, so keeping them here repeats them and
    hides the name behind them. What is not a grade or a plan is left exactly
    as the page wrote it.

    Example:
        >>> name_only("هتل 4 ستاره (ALL) BABAYLON")
        'هتل BABAYLON'
        >>> name_only("هتل 3 ستاره KEMER STAR (B.B)")
        'هتل KEMER STAR'
        >>> name_only("هتل 4 ستاره")
        'هتل'
        >>> name_only("Kemer Star Hotel")
        'Kemer Star Hotel'
    """
    return _clean(_A_BOARD.sub(" ", _A_GRADE.sub(" ", text)))


@dataclass
class HotelBlock:
    """What the block above a set of rows says about their hotel.

    Every field is optional: a page may name a hotel without rating it, or
    rate it without naming it, and liyaseir.com does the second.

    Example:
        >>> HotelBlock(name="The Train Resort", board_type="BB", path="div.hotel")
        HotelBlock(name='The Train Resort', star_rating=None, board_type='BB', boards=(), stars=(), each_nights=(), cities=(), nights=None, image_url=None, path='div.hotel')
    """

    name: Optional[str] = None
    star_rating: Optional[int] = None
    board_type: Optional[str] = None
    # One board per name, where the block gives each hotel its own. Empty where
    # it states a single board for the stay, which is the common case.
    boards: tuple[Optional[str], ...] = ()
    # The star and the nights of each of those names, where the block prints
    # one against each hotel rather than one for the whole stay.
    stars: tuple[Optional[int], ...] = ()
    each_nights: tuple[Optional[int], ...] = ()
    cities: tuple[Optional[str], ...] = ()
    nights: Optional[int] = None
    image_url: Optional[str] = None
    path: Optional[str] = None


def find_hotel_block(container: RowContainer) -> Optional[Node]:
    """Find the ancestor that describes the rows' hotel.

    Walking stops at the first ancestor whose own text is about lodging, not
    merely the first that says anything: a block adding only "2 شب 3 شب" is
    still part of the price table, and stopping there misses the name printed
    one level above it.

    Returns ``None`` when no ancestor within :data:`_MAX_LEVELS` describes a
    hotel, which is the honest answer for a page whose rows stand alone —
    avisatravel.com lists departure dates with no hotel attached to them.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> from crawler.extraction.structure.html_rows import find_row_containers
        >>> tree = HTMLParser(
        ...     '<div class="hotel"><a>The Train Resort</a>'
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 82,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div></div>'
        ... )
        >>> block = find_hotel_block(find_row_containers(tree)[0])
        >>> block.attributes.get("class")
        'hotel'
    """
    if not container.members:
        return None

    row_texts = container.texts()
    node = container.members[0].parent
    describes: Optional[object] = None

    for _ in range(_MAX_LEVELS):
        if node is None or node.tag in ("body", "html", "-undef"):
            break
        extra = _extra_text(node, row_texts)
        if len(extra) > _MAX_BLOCK_CHARS:
            # Ancestors only ever say more, so nothing above this is a hotel
            # block either.
            break
        if _describes_a_hotel(extra):
            if _any_name_in(node, extra):
                return node
            # A level can talk about hotels and name none: nahalgasht's price
            # panel carries a "رزرو با این هتل" button, and stopping there left
            # every package on the page nameless and folded them into one.
            describes = describes or node
        node = node.parent

    return describes


def _surrounding_text(block: Node, row_texts: list[str]) -> str:
    """The block's own text, plus its parent's when the card wraps it.

    Some sites split one hotel across two elements: nahalgasht.com puts the
    name and the location in ``div.reserve`` and the star rating on the
    ``div.hotelcard`` around it. The name has to come from the inner element —
    the outer one starts with a badge, "پیشنهاد ویژه نهال‌گشت", which would be
    read as the hotel — so the block stays where it is and only the ratings
    look one level further out.

    The parent is used only while it stays within :data:`_MAX_BLOCK_CHARS`,
    which is what stops this from climbing into the page.

    The parent's text already contains the block's, so it replaces it rather
    than being added to it.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser(
        ...     '<div class="card"><span>3 ستاره</span>'
        ...     '<div class="inner"><b>هتل پارسه</b>'
        ...     '<div class="bed">دو تخته 5,000,000 تومان</div></div></div>'
        ... )
        >>> inner = tree.css_first("div.inner")
        >>> _surrounding_text(inner, ["دو تخته 5,000,000 تومان"])
        '3 ستارههتل پارسه'
    """
    own = _extra_text(block, row_texts)
    parent = block.parent
    if parent is None or parent.tag in ("body", "html", "-undef"):
        return own

    wider = _extra_text(parent, row_texts)
    return wider if len(wider) <= _MAX_BLOCK_CHARS else own


# Longer than any rating and shorter than the URLs, class lists and JSON blobs
# that make up the rest of a card's attributes.
_MAX_ATTRIBUTE_CHARS = 20

# Never part of a rating widget.
_SKIP_ICON_TAGS = frozenset({'script', 'style', 'noscript', '-text'})


# A rating drawn as icons: how many may stand in a row, and how long a class
# list on one of them can be. Five is the usual scale and seven the widest seen.
_ICON_RUN = (3, 7)


def rating_in_icons(block: Node) -> Optional[int]:
    """A rating drawn as filled and empty icons, read by counting the filled.

    A star widget is a short row of empty elements where the earned ones carry
    a class the rest do not — bisanseir writes five ``<i>`` and marks three of
    them ``fill``. Nothing here is keyed on what that class is called; what is
    read is that a run of look-alike siblings starts with a marked prefix.

    Preferred over :func:`rating_in_attributes` because the icons are what the
    page shows a reader. bisanseir carries ``data-star="سه ستاره"`` on every
    hotel of a page, the same value for all 56 of them including the five star
    ones, while its icons vary from page to page — and on the pages that have
    no such attribute, 877 hotels reached the client with no rating at all.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser('<div><span class="star-rating">'
        ...     '<i class="fill"></i><i class="fill"></i><i class="fill"></i>'
        ...     '<i></i><i></i></span></div>')
        >>> rating_in_icons(tree.css_first("div"))
        3
        >>> whole = HTMLParser('<div><span><i class="on"></i><i class="on"></i>'
        ...     '<i class="on"></i><i class="on"></i><i class="on"></i></span></div>')
        >>> rating_in_icons(whole.css_first("div"))
        5
        >>> none = HTMLParser('<div><span><i></i><i></i></span></div>')
        >>> rating_in_icons(none.css_first("div")) is None
        True
    """
    for parent in [block, *block.css("*")]:
        found = _rating_of_run(parent)
        if found is not None:
            return found
    return None


def _rating_of_run(parent: Node) -> Optional[int]:
    """The rating this one element draws, or ``None`` if it draws none.

    Kept apart from :func:`rating_in_icons` so the walk that collects a block's
    hotels can ask it of each element it passes, instead of rescanning the
    whole block once per hotel.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser('<span><i class="fill"></i><i class="fill"></i>'
        ...                   '<i></i><i></i><i></i></span>')
        >>> _rating_of_run(tree.css_first("span"))
        2
    """
    least, most = _ICON_RUN
    icons = [node for node in parent.iter() if node.tag not in _SKIP_ICON_TAGS]
    if not least <= len(icons) <= most:
        return None
    if len({node.tag for node in icons}) != 1:
        return None
    if any(_clean(node.text(deep=True)) for node in icons):
        return None  # a row of words, not a row of icons
    marks = [(node.attributes.get("class") or "").strip() for node in icons]
    # Only a prefix counts: the earned ones come first, and a row whose marks
    # are scattered is not a rating.
    filled = 0
    for mark in marks:
        if not mark:
            break
        filled += 1
    if filled and all(not mark for mark in marks[filled:]):
        return filled
    return None


def rating_in_attributes(block: Node) -> Optional[int]:
    """A rating the block states in an attribute rather than in its text.

    bisanseir draws its ratings as filled and empty icons and writes the number
    nowhere a reader can see — except on the block itself, as
    ``data-star="سه ستاره"``. Its 1216 hotel rows all carried no rating for it.

    The value is what is read, never the attribute's name, so this is not keyed
    on one site's spelling of ``data-star``. A value only counts if it reads as
    a rating on its own, which a class list or an address does not.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser('<div data-star="سه ستاره"><i></i></div>')
        >>> rating_in_attributes(tree.css_first("div"))
        3
        >>> plain = HTMLParser('<div class="hotel-card" id="h5"></div>')
        >>> rating_in_attributes(plain.css_first("div")) is None
        True
    """
    for node in [block, *block.css("*")]:
        for value in (node.attributes or {}).values():
            if not value or len(value) > _MAX_ATTRIBUTE_CHARS:
                continue
            found = detect_star_rating(value)
            if found is not None:
                return found
    return None


# An img whose src is a placeholder until the page scrolls. The address the
# browser will swap in is parked in one of these.
_LAZY_SOURCES = ("data-src", "data-lazy-src", "data-original", "data-lazy")

# How far above the block to look. nahalgasht keeps the photograph beside the
# text rather than inside it, so its hotel is one level up; going further
# reaches the page's own banner, which is a picture of the destination.
_IMAGE_LEVELS = 1


def _names_a_file(address: str) -> bool:
    """Whether an address ends in something, rather than in a directory.

    bisanseir writes ``src="/media/"`` for the 16 of its 45 hotels that have
    no photograph, and that resolves to a folder which answers 403. An address
    naming no file is the site saying it has no picture.

    Example:
        >>> _names_a_file("/media/hotel/2026/07/25/grand.jpg")
        True
        >>> _names_a_file("/media/")
        False
    """
    path = address.split("?")[0].split("#")[0]
    return bool(path) and not path.endswith("/")


def _address(image: Node) -> Optional[str]:
    """Where an img really points, past any placeholder it was served with.

    Example:
        >>> tree = HTMLParser('<img src="data:image/svg+xml;base64,AAA"'
        ...                   ' data-src="/photos/oz-firat.jpg">')
        >>> _address(tree.css_first("img"))
        '/photos/oz-firat.jpg'
        >>> _address(HTMLParser('<img src="/media/">').css_first("img")) is None
        True
    """
    attributes = image.attributes or {}
    for key in ("src", *_LAZY_SOURCES):
        value = (attributes.get(key) or "").strip()
        if value and not value.startswith("data:") and _names_a_file(value):
            return value
    return None


def image_in_block(block: Node) -> Optional[str]:
    """The photograph this block shows of its hotel, if it shows one.

    Taken from the block rather than matched against the page by name, because
    a name match is not safe: vanrosh.com labels the picture beside "BODRUM
    PARK RESORT" as "SUNBAY PARK", and handing over the wrong building reads
    as a right answer. Scoped this way it is the site itself saying which
    hotel the picture belongs to.

    Above the block, a picture is taken only where there is exactly one — that
    is what makes it this hotel's. bisanseir gives the 16 of its 45 hotels
    that have no photograph an empty ``src``, and its hotels are siblings, so
    reaching up without this returned the neighbour's building for all of
    them: thirteen hotels shown as The Grand Day Night.

    Example:
        >>> tree = HTMLParser('<div class="hotel"><span>هتل آمون</span>'
        ...                   '<img src="/photos/amon.jpg"></div>')
        >>> image_in_block(tree.css_first("div.hotel"))
        '/photos/amon.jpg'
        >>> beside = HTMLParser('<div class="row"><div class="hotel">a</div>'
        ...                     '<img src="/photos/one.jpg"></div>')
        >>> image_in_block(beside.css_first("div.hotel"))
        '/photos/one.jpg'
        >>> crowd = HTMLParser('<div class="row"><div class="hotel">a</div>'
        ...                    '<img src="/a.jpg"><img src="/b.jpg"></div>')
        >>> image_in_block(crowd.css_first("div.hotel"))
        >>> image_in_block(HTMLParser('<div>no picture</div>').css_first("div"))
    """
    node, level = block, 0
    while node is not None and level <= _IMAGE_LEVELS:
        found = [a for a in (_address(i) for i in node.css("img")) if a]
        if found and (level == 0 or len(found) == 1):
            return found[0]
        node, level = node.parent, level + 1
    return None


def _any_name_in(block, extra: str) -> bool:
    """Whether this level names a hotel, rather than only mentioning one.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> tree = HTMLParser('<div><p>Lazdana Hotel</p></div>')
        >>> _any_name_in(tree.css_first("div"), "Lazdana Hotel")
        True
        >>> tree = HTMLParser('<div><p>رزرو با این هتل</p></div>')
        >>> _any_name_in(tree.css_first("div"), "رزرو با این هتل")
        False
    """
    for node in block.css("*"):
        own = _clean(node.text(deep=False))
        if own and own in extra and _looks_like_a_name(own):
            return True
    return False


def read_hotel_block(container: RowContainer) -> HotelBlock:
    """Read the hotel facts that belong to every row in ``container``.

    ``name`` holds every hotel the block names, joined with ``+``. Most blocks
    name one. A tour that moves between cities names two — bisanseir.com sells
    Moscow and St Petersburg under a single price — and both belong to the offer.

    The rating, board and duration are read from the block's own text, so a
    rating printed as a badge beside the name is picked up as readily as one
    written into it.

    Example:
        >>> from selectolax.parser import HTMLParser
        >>> from crawler.extraction.structure.html_rows import find_row_containers
        >>> tree = HTMLParser(
        ...     '<div class="hotel"><a>The Train Resort</a>'
        ...     '<span>3 ستاره</span><span>خدمات:</span><span>BB</span>'
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 82,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div></div>'
        ... )
        >>> found = read_hotel_block(find_row_containers(tree)[0])
        >>> found.name, found.star_rating, found.board_type
        ('The Train Resort', 3, 'BB')
    """
    block = find_hotel_block(container)
    if block is None:
        return HotelBlock()

    row_texts = container.texts()
    extra = _extra_text(block, row_texts)

    # Every name, not just the first: a tour that moves city sleeps in two
    # hotels on one price, and naming only the first hides half the stay.
    #
    # But two names printed back to back are one hotel written twice, not two
    # hotels. alefbatour puts the Persian and Latin names in adjacent spans —
    # "هتل باکسوز ارمنستان" then "BAXOS YEREVAN" — while bisanseir separates
    # real second hotels with their own labels. So a name only counts as a new
    # hotel when something that is not a name came between.
    names: list[str] = []
    # The board each of those names was followed by. lachinseir prints one per
    # hotel — "Seven Sea Hotel Da nang B.B", then "Bhaya halong cruises F.B" —
    # and giving the stay one board sent the second hotel's as the first's and
    # left the rest empty.
    boards: list[Optional[str]] = []
    # The same, for what else a multi-city block prints against each hotel.
    stars: list[Optional[int]] = []
    each_nights: list[Optional[int]] = []
    # Where each of them stands. A card that names its hotels in one element
    # writes the city in another — nahalgasht prints "دوشنبه" and "پنجکنت"
    # beside the hotels of a Tajikistan tour — and that is the only place the
    # city of a stay that moves is stated at all.
    cities: list[Optional[str]] = []
    # A block lists its hotels in one kind of element, so a name printed in a
    # different one is something else that reads like a name. nahalgasht writes
    # each hotel in a <p> and the city it stands in in a <span> beside it, and
    # the cities were coming out as hotels of their own.
    hotel_tag: Optional[str] = None
    open_run = False
    for node in block.css("*"):
        own = _clean(node.text(deep=False))
        if not own:
            # A rating drawn as icons says nothing in text, and sits after the
            # hotel it belongs to. bisanseir gives each hotel of a two-city
            # stay its own widget, and the second hotel had no grade at all.
            drawn = _rating_of_run(node)
            if drawn is not None and names and stars[-1] is None:
                stars[-1] = drawn
            continue
        if own not in extra:
            continue  # belongs to a row, not to the block
        a_name = _looks_like_a_name(own)
        if a_name and hotel_tag is None:
            hotel_tag = node.tag
        if a_name and node.tag == hotel_tag:
            if not open_run and name_only(own) not in names:
                names.append(name_only(own))
                boards.append(None)
                stars.append(None)
                each_nights.append(None)
                cities.append(None)
            open_run = True
            continue
        if not a_name:
            open_run = False
        if not names:
            continue
        if boards[-1] is None:
            boards[-1] = detect_board_type(own)
        if stars[-1] is None:
            stars[-1] = detect_star_rating(own)
        if each_nights[-1] is None:
            each_nights[-1] = detect_nights(own)
        if a_name and cities[-1] is None and detect_nights(own) is None:
            cities[-1] = own
    name = " + ".join(names) if names else None

    # The name must come from the block itself, but a rating or a board code
    # is often printed on the card around it, so those look a little wider.
    around = _surrounding_text(block, row_texts)

    return HotelBlock(
        name=name,
        star_rating=(detect_star_rating(around) or rating_in_icons(block)
                     or rating_in_attributes(block)),
        board_type=detect_board_type(around),
        boards=tuple(boards) if any(boards) else (),
        stars=tuple(stars) if len(names) > 1 and any(stars) else (),
        each_nights=tuple(each_nights) if len(names) > 1 and any(each_nights) else (),
        cities=tuple(cities) if any(cities) else (),
        nights=detect_nights(around),
        image_url=image_in_block(block),
        path=_element_path(block),
    )
