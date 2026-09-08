"""Reading the short labels a Persian tour page prints beside its prices.

A row rarely says only a number. It says what the number is for::

    دو تخته 92,000,000 تومان + 1,090 یورو
    هتل 3 ستاره (B.B) FORUM PARK
    2 ستارههتل پارسه ... 2 شب 3 شب

This module turns those labels into the controlled values the schema uses —
:data:`~crawler.schema.output_models.ROOM_DOUBLE` rather than the literal text
"دو تخته" — so the same offer reads the same way whichever site it came from.

Two habits of the real pages shape every pattern here:

* **Spaces are optional.** vanrosh.com renders ``دوتخته114٬690٬000 تومان`` and
  nahalgasht.com renders ``2 ستارههتل پارسه``: the label runs straight into the
  next value because the markup put them in adjacent tags. Matching on whole
  words alone would find neither.
* **Digits come in three scripts**, so text is normalised through
  :func:`~crawler.extraction.readers.prices.normalize_digits` before any number is
  read.

Every function returns ``None`` when nothing matches, following
:func:`crawler.schema.output_models.normalize_transport`: an unknown value is
better left empty, where the fill rate will show it, than guessed.
"""

from __future__ import annotations

import re
from typing import Optional

from crawler.extraction.readers.prices import normalize_digits
from crawler.schema.output_models import (
    ROOM_CHILD_NO_BED,
    ROOM_CHILD_WITH_BED,
    ROOM_DOUBLE,
    ROOM_INFANT,
    ROOM_SINGLE,
    ROOM_TRIPLE,
)

# Zero-width non-joiner, which Persian typing puts inside compound words. It is
# invisible, so a pattern written without it silently fails to match.
# Headings a site prints above a strip of *other* tours. A block under one of
# these is an advertisement, not this page's offers, and reading it records
# another tour's prices as this one's — the worst thing the extractor can do,
# because a wrong row is trusted while an empty one is visible.
#
# Note what is deliberately absent: "سایر تاریخ های برگزاری". alefbatour prints
# that above the *same* tour's other departures, which are genuine offers. The
# word "سایر" alone would sink them, so the phrases below all name tours rather
# than dates.
#
# Lives here rather than in structure/ because two steps read it: the row
# scorer, which refuses such a block, and main_text, which cuts it out before
# the page is flattened into one string.
OTHER_TOURS_HEADINGS: tuple[str, ...] = (
    "تورهای پیشنهادی",
    "تور های پیشنهادی",
    "سایر پیشنهادات",
    "پیشنهادهای مشابه",
    "تورهای مشابه",
    "تور های مشابه",
    "تورهای مرتبط",
    "دیگر تورها",
    "تورهای دیگر",
)

def names_other_tours(text: Optional[str]) -> Optional[str]:
    """The phrase in ``text`` that marks it as naming somebody else's tours.

    Two steps ask this, which is why it sits beside the vocabulary rather than
    inside either of them: the row scorer, about the heading above a block, and
    ``main_text``, about the text of one element while the tree still stands.

    Example:
        >>> names_other_tours("سایر تورهای پیشنهادی")
        'تورهای پیشنهادی'
        >>> names_other_tours("سایر تاریخ های برگزاری") is None
        True
        >>> names_other_tours(None) is None
        True
    """
    if not text:
        return None
    return next((phrase for phrase in OTHER_TOURS_HEADINGS if phrase in text), None)


_ZWNJ = "‌"

# Optional separator between the parts of a label: a space, a ZWNJ, or nothing.
_GAP = r"[\s" + _ZWNJ + r"]*"

# Room labels, longest and most specific first so that "کودک بدون تخت" is never
# read as the "کودک" of "کودک با تخت". Each entry allows both the spelled-out
# number and the digit, since sites mix them freely.
# An age band standing in for the room label. liyaseir prices by who sleeps in
# the bed rather than by what the bed is: "4 الی 12 سال با تخت" instead of
# "کودک با تخت". The ages say which child and the bed word says which price, so
# the bed word is what maps and the ages only mark the shape. Tried first,
# before the plain labels, because they are the longer match.
_DIGIT = r"[\d۰-۹٠-٩]"
_AGE_BAND = (
    r"(?:زیر" + _GAP + r")?" + _DIGIT + r"{1,2}"
    + r"(?:" + _GAP + r"(?:الی|تا)" + _GAP + _DIGIT + r"{1,2})?" + _GAP + r"سال"
)

_ROOM_PATTERNS: tuple[tuple[str, str], ...] = (
    (_AGE_BAND + _GAP + r"بدون" + _GAP + r"تخت", ROOM_CHILD_NO_BED),
    (_AGE_BAND + _GAP + r"با" + _GAP + r"تخت", ROOM_CHILD_WITH_BED),
    # "زیر 2 سال" is the infant band. One or two years only: a larger number
    # under the same wording is a child, and reading it as an infant would put
    # a child's fare in the infant column.
    (r"زیر" + _GAP + r"[۱۲١٢12]" + _GAP + r"سال", ROOM_INFANT),
    (r"کودک" + _GAP + r"بدون" + _GAP + r"تخت", ROOM_CHILD_NO_BED),
    (r"کودک" + _GAP + r"با" + _GAP + r"تخت", ROOM_CHILD_WITH_BED),
    (r"(?:نوزاد|اینفنت|انفانت)", ROOM_INFANT),
    (r"(?:سه|۳|3)" + _GAP + r"تخته", ROOM_TRIPLE),
    (r"(?:دو|۲|2)" + _GAP + r"تخته", ROOM_DOUBLE),
    # "تک تخته" and "تک نفره" are as common as "یک تخته": lachinseir writes
    # only the first, and single_price sat at 0% on every page of it while the
    # label was printed in plain sight.
    (r"(?:یک|تک|۱|1)" + _GAP + r"(?:تخته|نفره)", ROOM_SINGLE),
    # The English words written in Persian letters, which is how a table names
    # its columns: tahagasht heads them "اتاق دبل" and "اتاق سینگل" rather than
    # "دو تخته". Kept after the counted forms so "دو تخته" still wins where
    # both appear.
    (r"(?:تریپل|تريپل)", ROOM_TRIPLE),
    (r"(?:دبل|دابل)", ROOM_DOUBLE),
    (r"(?:سینگل|سينگل)", ROOM_SINGLE),
)

_ROOM_RE: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), value) for pattern, value in _ROOM_PATTERNS
)

# "3 ستاره" / "۴ ستاره" / "هتل: 4 ستاره", and the "5*" a rates table puts in
# its grade column. The digit may be attached to the word.
_STAR_RE = re.compile(r"(\d)" + _GAP + r"(?:ستاره|\*)")

# "7 شب" / "2 شب 3 شب" / "مدت تور: 7 شب و 8 روز". Nights, not days: the sites
# quote both and the schema stores nights.
_NIGHTS_RE = re.compile(r"(\d{1,2})" + _GAP + r"شب")

# "12روزه" / "8 روز". Only read when no night count was printed: a tour sold as
# N days is spent N-1 nights, and the sites that state days rarely also state
# nights.
_DAYS_RE = re.compile(r"(\d{1,2})" + _GAP + r"روز")

# Board codes as printed: "(B.B)", "خدمات:BB", "HB", "ALL". The dotted form is
# common enough that stripping dots is part of reading it.
_BOARD_RE = re.compile(
    r"(?<![A-Za-z])(B\.?B|H\.?B|F\.?B|U\.?A\.?L|A\.?L\.?L|R\.?O)(?![A-Za-z])",
    re.I,
)

# The same plans written out in Persian. Read because a site that spells the
# plan never prints the code beside it: tahagasht writes "بدون پذیرایی" and
# nothing else, which left board_type empty and, worse, read as a hotel name.
_BOARD_WORDS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"بدون" + _GAP + r"(?:پذیرایی|وعده|صبحانه)"), "RO"),
    (re.compile(r"تمام" + _GAP + r"پانسیون|فول" + _GAP + r"بورد"), "FB"),
    (re.compile(r"نیم" + _GAP + r"پانسیون"), "HB"),
    (re.compile(r"همه" + _GAP + r"چیز" + _GAP + r"شامل|آل" + _GAP + r"اینکلوسیو"), "ALL"),
    (re.compile(r"صبحانه"), "BB"),
)


def detect_room_type(text: str) -> Optional[str]:
    """Read which occupancy a row is priced for.

    Example:
        >>> detect_room_type("دو تخته 92,000,000 تومان")
        'double'
        >>> detect_room_type("کودک بدون تخت91٬490٬000 تومان")
        'child_no_bed'
        >>> detect_room_type("4 الی 12 سال با تخت 2690 دلار")
        'child_with_bed'
        >>> detect_room_type("2 الی 4 سال بدون تخت 1690 دلار")
        'child_no_bed'
        >>> detect_room_type("زیر 2 سال 190 دلار")
        'infant'
        >>> detect_room_type("زیر 12 سال با تخت")
        'child_with_bed'
        >>> detect_room_type("تک تخته2,260دلار")
        'single'
        >>> detect_room_type("رفت 03 شهریور 98,000,000 تومان") is None
        True
    """
    for pattern, value in _ROOM_RE:
        if pattern.search(text):
            return value
    return None


def detect_star_rating(text: str) -> Optional[int]:
    """Read a hotel's star count.

    Ratings outside 1..5 are rejected rather than returned: a "7 ستاره" on
    these pages is marketing copy, and a 0 is a template that never filled in.

    Example:
        >>> detect_star_rating("هتل 3 ستاره (B.B) FORUM PARK")
        3
        >>> detect_star_rating("۴ ستاره")
        4
        >>> detect_star_rating("هتل پارسه") is None
        True
    """
    match = _STAR_RE.search(normalize_digits(text))
    if not match:
        return None
    stars = int(match.group(1))
    return stars if 1 <= stars <= 5 else None


def detect_nights(text: str) -> Optional[int]:
    """Read how many nights a stretch of text describes.

    Nights are preferred over days wherever both are printed — "7 شب و 8 روز"
    is seven nights — and a day count is converted only when no night count is
    given, as alefbatour.com's "تور 12روزه" does.

    The first count wins. nahalgasht.com prints "2 شب 3 شب" on one card, where
    the two are separate options rather than a range, and taking the first
    keeps the row tied to the price printed beside it.

    Example:
        >>> detect_nights("مدت تور: 7 شب و 8 روز")
        7
        >>> detect_nights("2 شب 3 شب")
        2
        >>> detect_nights("تور 12روزه آفریقای جنوبی")
        11
        >>> detect_nights("دو تخته 92,000,000 تومان") is None
        True
    """
    normalized = normalize_digits(text)

    match = _NIGHTS_RE.search(normalized)
    if match:
        nights = int(match.group(1))
        return nights if 1 <= nights <= 60 else None

    match = _DAYS_RE.search(normalized)
    if match:
        nights = int(match.group(1)) - 1
        return nights if 1 <= nights <= 60 else None

    return None


def detect_board_type(text: str) -> Optional[str]:
    """Read the meal plan code, normalised to letters only.

    Example:
        >>> detect_board_type("هتل 3 ستاره (B.B) FORUM PARK")
        'BB'
        >>> detect_board_type("خدمات:HB")
        'HB'
        >>> detect_board_type("بدون پذیرایی")
        'RO'
        >>> detect_board_type("نیم پانسیون")
        'HB'
        >>> detect_board_type("هتل کورت") is None
        True
    """
    match = _BOARD_RE.search(text)
    if match:
        return match.group(1).replace(".", "").upper()
    for pattern, value in _BOARD_WORDS:
        if pattern.search(text):
            return value
    return None
