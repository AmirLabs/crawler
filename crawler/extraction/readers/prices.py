"""Reading prices out of the text a Persian tour site prints.

All seven target sites keep their prices in ordinary markup rather than in
embedded JSON, and none of them mark a price with a usable class name: the same
number sits in ``.price-value`` on one site, ``.text-price`` on the next and
``mdAndDown:w-full`` on a third. What *is* the same everywhere is the text
itself — an amount followed by a currency word.

So a price is recognised by content, not by selector. Three traps found on the
real pages drove the details here:

* **Separators vary.** ``98,000,000`` on most sites, but vanrosh.com writes
  ``114٬690٬000`` with U+066C, the Arabic thousands separator.
* **Digits vary.** tahagasht.com prints ``۴۵۹۰`` in Persian digits.
* **Size proves nothing.** An early rule that looked for large numbers missed
  every price on liyaseir.com, which sells in euros: ``2190 یورو`` is four
  digits. Anchoring on the currency word instead of the magnitude is what makes
  the rule work across all of them.

Currency values are the ``CURRENCY_*`` constants from
:mod:`crawler.schema.output_models`, so the vocabulary is defined once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from crawler.schema.output_models import (
    CURRENCY_AED,
    CURRENCY_EUR,
    CURRENCY_RIAL,
    CURRENCY_TOMAN,
    CURRENCY_USD,
)

# Persian (U+06F0-U+06F9) and Arabic-Indic (U+0660-U+0669) digits, in order, so
# each maps onto its ASCII counterpart by position.
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = str.maketrans(
    _PERSIAN_DIGITS + _ARABIC_DIGITS,
    "0123456789" * 2,
)

# Thousands separators seen on the target sites: ASCII comma, Arabic comma
# (U+060C) and Arabic thousands separator (U+066C). A dot is deliberately not
# in this list — none of the sites group with it, and treating it as a
# separator would turn a decimal into a number a thousand times too large.
_GROUPERS = ",،٬"

# What each site prints, mapped to the currency we store. Toman and Rial differ
# by a factor of ten and are both printed in Persian, so the word decides the
# unit; it is never assumed.
CURRENCY_WORDS: dict[str, str] = {
    "تومان": CURRENCY_TOMAN,
    "تومن": CURRENCY_TOMAN,
    "ریال": CURRENCY_RIAL,
    "ريال": CURRENCY_RIAL,   # Arabic yeh spelling, used by some sites
    "دلار": CURRENCY_USD,
    "یورو": CURRENCY_EUR,
    "يورو": CURRENCY_EUR,
    "درهم": CURRENCY_AED,
}

# An amount followed by a currency word. The amount accepts any digit script
# and any of the groupers above; the currency word is required, which is what
# separates a price from a phone number or a tour id.
# A slash may group thousands too — tahagasht writes "9/990/000 تومان" — but a
# slash also writes a date, and "1405/06/21" must never become a number. The
# two are told apart by the groups: thousands are three digits each, a date's
# month and day are two. Matched first, so the general shape below cannot take
# the last group on its own and call 9/990/000 zero, which is what it did.
_SLASH_GROUPED = re.compile(r"[\d۰-۹٠-٩]{1,3}(?:/[\d۰-۹٠-٩]{3})+")

_PRICE_RE = re.compile(
    r"(?P<amount>" + _SLASH_GROUPED.pattern + r"|"
    r"[\d۰-۹٠-٩][\d۰-۹٠-٩" + _GROUPERS + r".]*)"
    r"\s*(?P<currency>" + "|".join(CURRENCY_WORDS) + r")"
)

# Joins the two halves of a dual-currency price: "98,000,000 تومان + 260 دلار".
# Only "+" counts; a comma or a line break between two prices means two
# separate prices, not one split across currencies.
_PLUS_BETWEEN = re.compile(r"^[\s‌]*\+[\s‌]*$")


def normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)

def parse_amount(text: str) -> Optional[float]:
    """Turn one printed amount into a number, or ``None`` if it is not one.

    Grouping separators are removed; a dot is kept as a decimal point. Returns
    ``None`` rather than guessing when the text does not parse, so a bad value
    shows up as missing instead of hiding as a wrong number.

    Example:
        >>> parse_amount("114٬690٬000")
        114690000.0
        >>> parse_amount("۲۱۹۰")
        2190.0
        >>> parse_amount("9/990/000")
        9990000.0
        >>> parse_amount("1405/06/21") is None
        True
        >>> parse_amount("سه میلیون") is None
        True
    """
    cleaned = normalize_digits(text).strip()
    if _SLASH_GROUPED.fullmatch(cleaned):
        cleaned = cleaned.replace("/", "")
    for grouper in _GROUPERS:
        cleaned = cleaned.replace(grouper, "")
    cleaned = cleaned.replace(" ", "").replace("‌", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class PriceText:
    """One amount-plus-currency found in a piece of text.

    ``raw`` keeps the text exactly as printed so a mis-parsed value can always
    be traced back without re-fetching the page.

    Example:
        >>> PriceText(amount=98000000.0, currency=CURRENCY_TOMAN, raw="98,000,000 تومان", start=0, end=16)
        PriceText(amount=98000000.0, currency='IRT', raw='98,000,000 تومان', start=0, end=16)
    """

    amount: float
    currency: str  # one of the CURRENCY_* constants
    raw: str
    start: int     # character offsets into the text it was found in, so two
    end: int       # halves of a dual-currency price can be seen as adjacent


def find_prices(text: str) -> list[PriceText]:
    """Find every amount-with-currency in ``text``, in the order printed.

    A number without a currency word beside it is ignored: on these pages that
    is a phone number, a tour id or a night count far more often than a price.

    Example:
        >>> find_prices("دو تخته 92,000,000 تومان + 1,090 یورو")
        [PriceText(amount=92000000.0, currency='IRT', raw='92,000,000 تومان', start=8, end=24),
         PriceText(amount=1090.0, currency='EUR', raw='1,090 یورو', start=27, end=37)]
    """
    found: list[PriceText] = []
    for match in _PRICE_RE.finditer(text):
        amount = parse_amount(match.group("amount"))
        if amount is None:
            continue
        found.append(
            PriceText(
                amount=amount,
                currency=CURRENCY_WORDS[match.group("currency")],
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    return found


# The currencies an Iranian agency quotes in its own money. Everything else is
# the foreign half of a price, whichever side of the "+" it was printed on.
_LOCAL_CURRENCIES = frozenset({CURRENCY_TOMAN, CURRENCY_RIAL})


def _local_first(
    first: PriceText, second: PriceText
) -> tuple[PriceText, Optional[PriceText]]:
    """Order one joined pair so the Iranian half leads.

    A pair of two local or two foreign amounts is left in the order the page
    wrote it: there is nothing to swap, and reordering it would only lose the
    page's own meaning.

    Example:
        >>> text = "545 دلار + 119,990,000 تومان"
        >>> a, b = find_prices(text)
        >>> local, foreign = _local_first(a, b)
        >>> local.currency, foreign.currency
        ('IRT', 'USD')
    """
    if second.currency in _LOCAL_CURRENCIES and first.currency not in _LOCAL_CURRENCIES:
        return second, first
    return first, second


def pair_dual_currency(
    text: str, prices: list[PriceText]
) -> list[tuple[PriceText, Optional[PriceText]]]:
    """Group prices joined by ``+`` into local-plus-foreign pairs.

    Persian agencies quote outbound tours as "local part + foreign part" —
    ``98,000,000 تومان + 260 دلار`` is one price with two components, not two
    prices. Splitting them loses the dollars; treating every adjacent pair as
    one would merge a hotel's price with the next hotel's. Only an explicit
    ``+`` between the two joins them.

    Returns one entry per logical price: the local half first, then the foreign
    half when there is one. Which is which is decided by the currency, never by
    the order the page printed them in. alefbatour.com writes the Toman first
    and nahalgasht.com writes the dollars first, and taking the first as local
    put ``545 USD`` in the same output column as ``43,900,000 IRT`` — two
    numbers a customer would compare directly.

    Example:
        >>> text = "98,000,000 تومان + 260 دلار"
        >>> [(a.currency, b.currency if b else None) for a, b in pair_dual_currency(text, find_prices(text))]
        [('IRT', 'USD')]
        >>> flipped = "545 دلار + 119,990,000 تومان"
        >>> [(a.currency, b.currency if b else None) for a, b in pair_dual_currency(flipped, find_prices(flipped))]
        [('IRT', 'USD')]
    """
    pairs: list[tuple[PriceText, Optional[PriceText]]] = []
    skip_next = False

    for i, price in enumerate(prices):
        if skip_next:
            skip_next = False
            continue

        nxt = prices[i + 1] if i + 1 < len(prices) else None
        if nxt is not None and _PLUS_BETWEEN.match(text[price.end:nxt.start]):
            pairs.append(_local_first(price, nxt))
            skip_next = True
        else:
            pairs.append((price, None))

    return pairs
