"""Reading how a tour travels — by air, by train or by road.

Reading the whole page for a transport word does not work, and the way it
fails is instructive. nahalgasht.com's menu links to "تور مشهد هوایی" and
"تور مشهد با قطار" on every page, so whichever word the scan meets first
answers for every tour on the site. Every nahalgasht tour came out as ``air``,
including the ones whose own titles say "زمینی" and "با قطار".

The sites do state it, in two different ways, and this reads them in order of
how directly they answer:

1. **The tour's own name.** "تور زمینی مشهد از تبریز", "تور مشهد ... با قطار".
   Nothing is more specific to this tour than its title.
2. **A labelled field.** nahalgasht writes "حمل و نقل تور: هواپیما",
   bisanseir writes "ایرلاین: سپهران", liyaseir has a "پرواز" row. The site
   has said it plainly; there is nothing to infer.
3. **The journey description.** alefbatour and avisatravel label neither, but
   both describe the trip itself — airports for a flight, "قطار" for a train.
   Counting those words separates the two cleanly: an Alanya flight page names
   an airport six times and a train never, while the Van train page is the
   reverse.

When none of the three answers, the field is left empty. Falling back to the
whole page is exactly what produced the wrong values, and an empty field is
visible in the fill rate while a wrong one is believed.
"""

from __future__ import annotations

import re
from typing import Optional

from selectolax.parser import HTMLParser

from crawler.schema.output_models import (
    TRANSPORT_AIR,
    TRANSPORT_BUS,
    TRANSPORT_TRAIN,
    normalize_transport,
)

# Elements that belong to the site rather than to the tour. The navigation menu
# names every kind of tour an agency sells, which is what poisons a plain scan.
_CHROME_TAGS = "nav,header,footer,aside"
_CHROME_WORDS = ("menu", "nav", "header", "footer", "breadcrumb", "sidebar", "topbar")

# Labels that introduce the answer, with how much text after them to read. The
# value often runs straight into the label with no space — nahalgasht renders
# "حمل و نقل تورهواپیما" — so a fixed window is read rather than a word.
_TRANSPORT_LABELS = ("حمل و نقل تور", "حمل‌ونقل تور", "نوع حمل و نقل", "وسیله نقلیه")
_LABEL_WINDOW = 30

# A labelled airline is an answer in itself: only flights have one.
_AIRLINE_LABELS = ("ایرلاین:", "ایرلاین :", "هواپیمایی:", "شرکت هواپیمایی")

# Words that describe the journey. "پرواز" is deliberately absent: it appears
# in "هزینه پرواز" and "بلیت پرواز" on pages for trains and coaches too, while
# an airport is only ever named by a tour that flies.
_JOURNEY_SIGNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (TRANSPORT_AIR, ("فرودگاه", "ترانسفر فرودگاهی")),
    (TRANSPORT_TRAIN, ("قطار", "ریلی", "راه آهن", "راه‌آهن")),
    (TRANSPORT_BUS, ("اتوبوس", "زمینی", "پایانه")),
)


def main_text(html: str) -> str:
    """The page's own words, with the site's furniture removed.

    Menus, headers and footers list every kind of tour the agency sells, so
    they answer for tours they have nothing to do with.

    Example:
        >>> main_text('<body><nav>تور مشهد با قطار</nav><p>پرواز مستقیم</p></body>')
        'پرواز مستقیم'
    """
    tree = HTMLParser(html)
    for node in tree.css("script,style,noscript"):
        node.decompose()
    for node in tree.css(_CHROME_TAGS):
        node.decompose()
    for node in tree.css("*"):
        attributes = (node.attributes.get("class") or "") + " " + (node.attributes.get("id") or "")
        if any(word in attributes.lower() for word in _CHROME_WORDS):
            node.decompose()
    body = tree.body
    return " ".join((body.text(deep=True) if body else "").split())


def from_label(text: str) -> Optional[str]:
    """Read a transport that the page has labelled as such.

    Example:
        >>> from_label("مدت اقامت2 و 3 شبحمل و نقل تورهواپیما تاریخ رفتمرداد ماه")
        'air'
        >>> from_label("مبدا: تهران مقصد: تفلیس ایرلاین: سپهران")
        'air'
        >>> from_label("تور کیش 3 شب") is None
        True
    """
    for label in _TRANSPORT_LABELS:
        position = text.find(label)
        if position >= 0:
            after = text[position + len(label): position + len(label) + _LABEL_WINDOW]
            found = normalize_transport(after)
            if found:
                return found

    if any(label in text for label in _AIRLINE_LABELS):
        return TRANSPORT_AIR

    return None


def from_journey(text: str) -> Optional[str]:
    """Infer the transport from how the journey itself is described.

    The winner has to be alone: a page that names airports and trains equally
    often has not told us anything, and guessing between them would be the
    same mistake in a smaller place.

    Example:
        >>> from_journey("از فرودگاه بین‌المللی امام خمینی IKA به فرودگاه قاضی پاشا")
        'air'
        >>> from_journey("حرکت با قطار فدک، مدت سفر 01:00 با قطار")
        'train'
        >>> from_journey("تور 7 شب ارمنستان") is None
        True
    """
    counts: dict[str, int] = {}
    for kind, words in _JOURNEY_SIGNS:
        counts[kind] = sum(text.count(word) for word in words)

    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        return None
    if any(kind != best and count >= counts[best] for kind, count in counts.items()):
        return None
    return best


def detect_transport(html: str, tour_name: Optional[str] = None) -> tuple[Optional[str], str]:
    """Decide how a tour travels, and say which source answered.

    The source is returned alongside the value so a wrong answer can be traced
    to the rule that produced it without opening the page again.

    Example:
        >>> detect_transport("<body><p>از فرودگاه امام خمینی</p></body>", "تور آلانیا")
        ('air', 'journey')
        >>> detect_transport("<body></body>", "تور زمینی مشهد از تبریز")
        ('bus', 'tour_name')
        >>> detect_transport("<body><p>تور کیش</p></body>", "تور کیش")
        (None, 'none')
    """
    if tour_name:
        found = normalize_transport(tour_name)
        if found:
            return found, "tour_name"

    text = main_text(html)

    found = from_label(text)
    if found:
        return found, "label"

    found = from_journey(text)
    if found:
        return found, "journey"

    return None, "none"
