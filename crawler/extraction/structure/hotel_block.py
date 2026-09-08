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

from selectolax.parser import Node

from crawler.extraction.structure.html_rows import RowContainer, _element_path
from crawler.extraction.readers.labels import (
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
    r"\(?\s*(?<![A-Za-z])(?:B\.?B|H\.?B|F\.?B|U\.?A\.?L|A\.?L\.?L|R\.?O)(?![A-Za-z])\s*\)?",
    re.I,
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
        HotelBlock(name='The Train Resort', star_rating=None, board_type='BB', nights=None, path='div.hotel')
    """

    name: Optional[str] = None
    star_rating: Optional[int] = None
    board_type: Optional[str] = None
    nights: Optional[int] = None
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

    for _ in range(_MAX_LEVELS):
        if node is None or node.tag in ("body", "html", "-undef"):
            return None
        extra = _extra_text(node, row_texts)
        if len(extra) > _MAX_BLOCK_CHARS:
            # Ancestors only ever say more, so nothing above this is a hotel
            # block either.
            return None
        if _describes_a_hotel(extra):
            return node
        node = node.parent

    return None


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
    open_run = False
    for node in block.css("*"):
        own = _clean(node.text(deep=False))
        if not own or own not in extra:
            continue  # belongs to a row, not to the block
        if not _looks_like_a_name(own):
            open_run = False
            continue
        if not open_run and name_only(own) not in names:
            names.append(name_only(own))
        open_run = True
    name = " + ".join(names) if names else None

    # The name must come from the block itself, but a rating or a board code
    # is often printed on the card around it, so those look a little wider.
    around = _surrounding_text(block, row_texts)

    return HotelBlock(
        name=name,
        star_rating=detect_star_rating(around),
        board_type=detect_board_type(around),
        nights=detect_nights(around),
        path=_element_path(block),
    )
