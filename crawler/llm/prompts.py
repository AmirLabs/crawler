"""What the model is asked, and the shape it must answer in.

The question is narrow on purpose: name the places a tour goes to, and the
place it starts from. Everything else a title carries — the month, the airline,
the number of nights, the hotel, the star rating — already has a rule that
reads it, and those rules are testable in a way a model's answer is not.

The answer's shape is a Pydantic model rather than a hand-written JSON schema.
Both end up as the same constraint on the API, but a class carries its types
where they can be checked, and a dict of ``{"type": "object", ...}`` hides a
typo until it produces a puzzling reply at run time.

The examples are real titles from the target sites, chosen for the cases a rule
struggles with:

* the destination moves around — "تور آلانیا مرداد 1405" names it second,
  "تور 10 روزه کره جنوبی" names it last;
* a tour often visits several places — "تور بالی + اوبود + سنگاپور";
* an origin appears only when the tour starts outside Tehran;
* an exhibition title names its city inside a sentence and never says "تور".

``PROMPT_VERSION`` is part of the cache key. Change the wording and it has to
change too, or every improvement will keep returning yesterday's answers while
looking like it worked.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

PROMPT_VERSION = "tour-entities-1"


class TourEntity(BaseModel):
    """The places named by one tour title."""

    index: int = Field(description="The number of the title this answers.")
    destinations: list[str] = Field(
        description=(
            "Every city or country the tour visits, in the order written, "
            "spelled exactly as in the title."
        )
    )
    origin: Optional[str] = Field(
        description=(
            "The city the tour departs from, only when the title says so. "
            "Null otherwise."
        )
    )


class TourEntities(BaseModel):
    """One answer per title in the batch."""

    results: list[TourEntity]


SYSTEM_PROMPT = """\
You read the titles of Persian tour packages and name the places involved.

For each numbered title, return:
  - destinations: every city or country the tour goes to, in the order written,
    spelled exactly as in the title. A tour may go to several places.
  - origin: the city the tour starts from, only if the title says it.
    Otherwise null.

Rules:
  - Copy the place names character for character from the title. Do not
    translate them, correct their spelling, or expand them.
  - A hotel name is not a destination. In "تور مشهد هتل سی نور", the
    destination is مشهد, not the hotel.
  - An airline is not a destination. Names in brackets — (کاسپین),
    (ایران ایرتور), (قشم ایر) — are airlines.
  - Months, seasons and years are not destinations: مرداد, شهریور, تابستان,
    نوروز, 1405.
  - "از X" marks the origin, not a destination.
  - An exhibition or congress title still has a destination: the city it is
    held in.
  - If no place is named at all, return an empty list.
"""

# Real titles from the target sites. Each is here for a case the rules find
# hard, not to pad the count.
FEW_SHOT_INPUT = """\
1. تور آلانیا مرداد 1405 (کاسپین)
2. تور 10 روزه کره جنوبی
3. تور بالی + اوبود + سنگاپور + کوالالامپور تابستان و پاییز 1405
4. تور مشهد از تهران هتل سی نور با قطار 5 ستاره
5. تور بدروم غیر مسقیم 6 شب - ویژه مردادو شهریور ماه 1405 ( ایران ایرتور )
6. نمایشگاه موادغذایی پاریس(SIAL)
7. آفر تور شیراز
8. تور زمینی مشهد از تبریز هتل مجلل درویشی\
"""

FEW_SHOT_OUTPUT = """\
{"results": [
 {"index": 1, "destinations": ["آلانیا"], "origin": null},
 {"index": 2, "destinations": ["کره جنوبی"], "origin": null},
 {"index": 3, "destinations": ["بالی", "اوبود", "سنگاپور", "کوالالامپور"], "origin": null},
 {"index": 4, "destinations": ["مشهد"], "origin": "تهران"},
 {"index": 5, "destinations": ["بدروم"], "origin": null},
 {"index": 6, "destinations": ["پاریس"], "origin": null},
 {"index": 7, "destinations": ["شیراز"], "origin": null},
 {"index": 8, "destinations": ["مشهد"], "origin": "تبریز"}
]}\
"""


def build_messages(titles: list[str]) -> list[tuple[str, str]]:
    """Build the messages for one batch of titles.

    Titles are numbered from 1 so the reply can be matched back by index.
    Matching on the text itself would break on the duplicate titles these sites
    publish across sibling pages.

    Example:
        >>> messages = build_messages(["تور کیش 3 شب", "تور دبی از شیراز"])
        >>> len(messages), messages[-1][1].splitlines()
        (4, ['1. تور کیش 3 شب', '2. تور دبی از شیراز'])
    """
    numbered = "\n".join(f"{i}. {title}" for i, title in enumerate(titles, start=1))
    return [
        ("system", SYSTEM_PROMPT),
        ("human", FEW_SHOT_INPUT),
        ("ai", FEW_SHOT_OUTPUT),
        ("human", numbered),
    ]
