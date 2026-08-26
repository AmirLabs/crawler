"""Reading the Jalali dates a Persian tour page prints.

The target sites write departure dates in a handful of shapes::

    01 شهریور ساعت 11:00        alefbatour.com
    رفت 07 شهریور ساعت : 22:30   avisatravel.com
    21 آذر   25 آذر   28 آذر     bisanseir.com
    تاریخ رفت: مرداد ماه         nahalgasht.com
    1405/06/28                   in page text on several sites

Three facts about them drive everything here:

* **Most carry no year.** "01 شهریور" is a departure, and which year it belongs
  to has to be inferred. Tours are sold ahead of time, so the reading that a
  date already weeks past means next year is the one that matches the pages.
* **Some carry no day.** nahalgasht.com states only "مرداد ماه" for a card
  covering a whole month. That is a real, partial answer, so it is returned as
  one — with ``day`` left empty rather than invented.
* **Digits come in three scripts**, so text is normalised through
  :func:`~crawler.extraction.readers.prices.normalize_digits` first.

Conversion to Gregorian is done here rather than by a library so the project
keeps its short dependency list; the two algorithms below are the standard
ones and are checked against known dates in the tests.

Following the rest of extraction, ``raw`` is always kept: a date that converts
wrongly can be traced and redone without re-crawling, exactly as
:class:`crawler.schema.output_models.TourDate` expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from crawler.extraction.readers.prices import normalize_digits

# Jalali month names in order, so the index gives the month number. Spellings
# with an Arabic yeh are included because some sites type them that way.
_MONTH_NAMES: tuple[tuple[str, ...], ...] = (
    ("فروردین", "فروردين"),
    ("اردیبهشت", "ارديبهشت"),
    ("خرداد",),
    ("تیر", "تير"),
    ("مرداد", "امرداد"),
    ("شهریور", "شهريور"),
    ("مهر",),
    ("آبان", "ابان"),
    ("آذر", "اذر"),
    ("دی",),
    ("بهمن",),
    ("اسفند",),
)

JALALI_MONTHS: dict[str, int] = {
    name: number
    for number, spellings in enumerate(_MONTH_NAMES, start=1)
    for name in spellings
}

_MONTH_ALTERNATION = "|".join(
    sorted(JALALI_MONTHS, key=len, reverse=True)  # longest first, so "دی" never
)                                                 # wins inside another name

# "01 شهریور", optionally followed by "ماه" and a four-digit year.
_DAY_MONTH_RE = re.compile(
    r"(?P<day>\d{1,2})\s*(?P<month>" + _MONTH_ALTERNATION + r")"
    r"(?:\s*ماه)?(?:\s*(?P<year>\d{4}))?"
)

# "مرداد ماه" with no day at all, as nahalgasht.com prints it.
_MONTH_ONLY_RE = re.compile(
    r"(?<![\d])(?P<month>" + _MONTH_ALTERNATION + r")\s*ماه(?:\s*(?P<year>\d{4}))?"
)

# Days listed in front of a dated one, sharing its month: liyaseir writes
# "12 و 29 مرداد" and vanrosh "9 , 23 مهرماه". Two sites of seven print their
# departures this way, and reading only the dated one lost 9 مهر, 7 آبان and
# 12 مرداد — real departures of tours still on sale.
#
# Anchored to the end of the run so it only ever matches text that runs
# straight into a day-and-month date. A bare number anywhere else on the page
# is a price, a night count or a tour id far more often than a day.
_LEADING_DAYS_RE = re.compile(r"(?:\d{1,2}\s*[،,وo]\s*)+$")
_LEADING_DAY_RE = re.compile(r"\d{1,2}")

# How far back to look for such a run. Four days plus their separators fit;
# more than that and the run would be reaching across a sentence.
_LEADING_DAYS_WINDOW = 24

# "1405/06/28"
_SLASH_RE = re.compile(r"(?P<year>\d{4})\s*/\s*(?P<month>\d{1,2})\s*/\s*(?P<day>\d{1,2})")

# A time right after a date: "ساعت 11:00", "ساعت : 22:30", or a bare "11:00".
_TRAILING_TIME_RE = re.compile(r"^\s*(?:ساعت\s*:?\s*)?(?P<hour>\d{1,2}):(?P<minute>\d{2})")

# How far past a date may fall before it is read as next year instead. Pages
# keep departures visible for a while after they leave, so a small window has
# to stay in the current year.
_PAST_TOLERANCE_DAYS = 45


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Convert a Gregorian date to Jalali.

    Example:
        >>> gregorian_to_jalali(2026, 8, 15)
        (1405, 5, 24)
    """
    g_d_m = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        (365 * gy)
        + ((gy2 + 3) // 4)
        - ((gy2 + 99) // 100)
        + ((gy2 + 399) // 400)
        - 80
        + gd
        + g_d_m[gm - 1]
    )
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + (days // 31)
        jd = 1 + (days % 31)
    else:
        jm = 7 + ((days - 186) // 30)
        jd = 1 + ((days - 186) % 30)
    return jy, jm, jd


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    """Convert a Jalali date to Gregorian.

    Example:
        >>> jalali_to_gregorian(1405, 5, 24)
        (2026, 8, 15)
    """
    jy += 1595
    days = (
        -355668
        + (365 * jy)
        + ((jy // 33) * 8)
        + (((jy % 33) + 3) // 4)
        + jd
    )
    days += (jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186

    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1

    leap = (gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)
    month_lengths = (0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    gm = 0
    while gm < 13 and gd > month_lengths[gm]:
        gd -= month_lengths[gm]
        gm += 1
    return gy, gm, gd


@dataclass
class DateText:
    """One date as a page printed it, before any year has been guessed.

    ``day`` is optional because a page may state only a month, and ``year`` is
    optional because most of them state no year at all. Both stay empty here;
    filling them in is :func:`to_iso`'s job, which is where the guess is made
    and can be reviewed.

    Example:
        >>> DateText(raw="01 شهریور ساعت 11:00", day=1, month=6, year=None, time="11:00", start=0, end=20)
        DateText(raw='01 شهریور ساعت 11:00', day=1, month=6, year=None, time='11:00', start=0, end=20)
    """

    raw: str
    month: int
    day: Optional[int] = None
    year: Optional[int] = None
    time: Optional[str] = None  # "11:00"
    start: int = 0
    end: int = 0


def _read_time_after(text: str, position: int) -> tuple[Optional[str], int]:
    """Read a time immediately following ``position``, if one is there.

    Returns the time and the offset where it ends, so the caller can extend the
    raw text to cover it.

    Example:
        >>> _read_time_after("01 شهریور ساعت 11:00 بعد", 9)
        ('11:00', 20)
    """
    match = _TRAILING_TIME_RE.match(text[position:])
    if not match:
        return None, position
    hour, minute = int(match.group("hour")), int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None, position
    return f"{hour:02d}:{minute:02d}", position + match.end()


def _month_text(parsed: DateText) -> str:
    """The month as the page spelled it, taken off a date already read.

    Example:
        >>> _month_text(find_dates("23 مهرماه")[0])
        'مهرماه'
    """
    return re.sub(r"^\s*\d{1,2}\s*", "", parsed.raw).strip()


def find_dates(text: str) -> list[DateText]:
    """Find every Jalali date in ``text``, in the order printed.

    Day-and-month dates are read first, then the bare days listed in front of
    one, then slash dates, then month-only mentions — and a month-only match is
    dropped when a fuller date already covers that part of the text, so
    "01 شهریور ماه" yields one date rather than two.

    Example:
        >>> find_dates("رفت 07 شهریور ساعت : 22:30 برگشت 16 شهریور ساعت : 02:00")
        [DateText(raw='07 شهریور ساعت : 22:30', month=6, day=7, year=None, time='22:30', start=4, end=26),
         DateText(raw='16 شهریور ساعت : 02:00', month=6, day=16, year=None, time='02:00', start=33, end=55)]
        >>> [d.raw for d in find_dates("25 تیر، 12 و 29 مرداد، 19 شهریور")]
        ['25 تیر', '12 مرداد', '29 مرداد', '19 شهریور']
        >>> [d.raw for d in find_dates("9 , 23 مهرماه | 7 , 21 آبان ماه")]
        ['9 مهرماه', '23 مهرماه', '7 آبان ماه', '21 آبان ماه']
    """
    normalized = normalize_digits(text)
    found: list[DateText] = []
    covered: list[tuple[int, int]] = []

    for match in _DAY_MONTH_RE.finditer(normalized):
        time, end = _read_time_after(normalized, match.end())
        year = int(match.group("year")) if match.group("year") else None
        found.append(
            DateText(
                raw=normalized[match.start():end].strip(),
                month=JALALI_MONTHS[match.group("month")],
                day=int(match.group("day")),
                year=year,
                time=time,
                start=match.start(),
                end=end,
            )
        )
        covered.append((match.start(), end))

    # A run of bare days in front of one of those dates belongs to its month.
    for parsed in list(found):
        if parsed.day is None:
            continue
        window = normalized[max(0, parsed.start - _LEADING_DAYS_WINDOW): parsed.start]
        run = _LEADING_DAYS_RE.search(window)
        if run is None:
            continue
        offset = parsed.start - len(window) + run.start()
        for day in _LEADING_DAY_RE.finditer(run.group(0)):
            number = int(day.group(0))
            if not 1 <= number <= 31:
                continue
            found.append(
                DateText(
                    # The month the site left out is written back in. "9" alone
                    # would reach the client's raw column meaning nothing, and
                    # raw is what a wrong date is traced by.
                    raw=f"{day.group(0)} {_month_text(parsed)}".strip(),
                    month=parsed.month,
                    day=number,
                    year=parsed.year,
                    start=offset + day.start(),
                    end=offset + day.end(),
                )
            )
            covered.append((offset + day.start(), offset + day.end()))

    for match in _SLASH_RE.finditer(normalized):
        month = int(match.group("month"))
        day = int(match.group("day"))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        time, end = _read_time_after(normalized, match.end())
        found.append(
            DateText(
                raw=normalized[match.start():end].strip(),
                month=month,
                day=day,
                year=int(match.group("year")),
                time=time,
                start=match.start(),
                end=end,
            )
        )
        covered.append((match.start(), end))

    for match in _MONTH_ONLY_RE.finditer(normalized):
        if any(start <= match.start() < end for start, end in covered):
            continue  # already part of a fuller date
        found.append(
            DateText(
                raw=normalized[match.start():match.end()].strip(),
                month=JALALI_MONTHS[match.group("month")],
                day=None,
                year=int(match.group("year")) if match.group("year") else None,
                start=match.start(),
                end=match.end(),
            )
        )

    return sorted(found, key=lambda d: d.start)


def to_iso(parsed: DateText, today: Optional[date] = None) -> Optional[str]:
    """Turn a parsed date into an ISO Gregorian string, guessing the year.

    Returns ``None`` when the page gave no day, since a month alone cannot be
    a date and inventing the first of the month would read as fact.

    The year, when missing, is taken to be the current Jalali year unless that
    would put the departure more than ``45`` days in the past, in which case it
    is read as next year. Tour pages advertise upcoming departures and keep
    recent ones visible for a while, so this matches how they are written.

    Example:
        >>> to_iso(DateText(raw="24 مرداد", month=5, day=24), today=date(2026, 8, 1))
        '2026-08-15'
        >>> to_iso(DateText(raw="مرداد ماه", month=5)) is None
        True
    """
    if parsed.day is None:
        return None

    if parsed.year is not None:
        gy, gm, gd = jalali_to_gregorian(parsed.year, parsed.month, parsed.day)
        return f"{gy:04d}-{gm:02d}-{gd:02d}"

    today = today or date.today()
    current_jy, _, _ = gregorian_to_jalali(today.year, today.month, today.day)

    gy, gm, gd = jalali_to_gregorian(current_jy, parsed.month, parsed.day)
    candidate = date(gy, gm, gd)
    if (today - candidate).days > _PAST_TOLERANCE_DAYS:
        gy, gm, gd = jalali_to_gregorian(current_jy + 1, parsed.month, parsed.day)
        candidate = date(gy, gm, gd)

    return candidate.isoformat()
