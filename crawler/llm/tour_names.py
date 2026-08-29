"""Naming the places a tour goes to, from its title.

The parts are already there: :mod:`crawler.llm.cache` remembers answers,
:mod:`crawler.llm.prompts` holds the question, :mod:`crawler.llm.client` sends
it. This is where they meet, and where two things happen that belong to neither.

**Only new titles are asked about.** Titles repeat far more than tours do —
alefbatour publishes 738 pages for 108 tours — and a daily refresh meets almost
nothing it has not seen. Every title is looked up in the cache first, and a run
where nothing is new costs nothing at all.

**Answers are checked against the title.** The schema guarantees the shape of a
reply, not its truth. A place name that does not appear in the title it came
from was not read out of it, so it is dropped. That single test is what keeps a
plausible invention out of the output, and it costs nothing to apply.
"""

from __future__ import annotations

from typing import Optional

from crawler.config import LLM_MODEL
from crawler.extraction.models import (
    METHOD_LLM,
    ExtractedField,
    ExtractedPage,
    record_attempt,
)
from crawler.extraction.readers.places import TourPlaces
from crawler.extraction.target_fields import PAGE_FIELDS, unresolved
from crawler.llm.cache import AnswerCache
from crawler.llm.client import ask, is_available
from crawler.llm.prompts import PROMPT_VERSION, TourEntities, build_messages
from crawler.utils.logger import get_logger

# How many titles go in one request. The instructions cost about 750 tokens and
# a title about 30, so fifty at a time spreads that cost thinly while keeping
# each reply small enough to arrive whole.
BATCH_SIZE = 50

_LOG = get_logger(__name__)

# Characters ignored when checking that an answer appears in its title. Sites
# write "بالی + اوبود" and "بنتوتا+کندی", and a name may be quoted or
# bracketed, so punctuation and spacing cannot be part of the comparison.
_IGNORED = " \t\n‌-+_()[]،,.:؛?«»\"'"


def _comparable(text: str) -> str:
    """Strip a string down to what a comparison should care about.

    Example:
        >>> _comparable("سنت پترزبورگ")
        'سنتپترزبورگ'
        >>> _comparable("بالی + اوبود") == _comparable("بالی+اوبود")
        True
    """
    return "".join(character for character in text if character not in _IGNORED)


def _verified(title: str, destinations: list[str], origin: Optional[str]) -> TourPlaces:
    """Keep only the places that actually appear in the title.

    The schema cannot check this: a reply naming Dubai for a tour to Kish is
    perfectly well-formed. Since the model is told to copy names character for
    character, anything not present in the title was not read from it.

    Example:
        >>> _verified("تور کیش از اهواز", ["کیش"], "اهواز")
        TourPlaces(destinations=['کیش'], origin='اهواز')
        >>> _verified("تور کیش", ["دبی"], None)
        TourPlaces(destinations=[], origin=None)
    """
    haystack = _comparable(title)
    kept = [
        place
        for place in destinations
        if place and _comparable(place) and _comparable(place) in haystack
    ]
    checked_origin = (
        origin if origin and _comparable(origin) in haystack else None
    )
    return TourPlaces(destinations=kept, origin=checked_origin)


def read_places(
    titles: list[str],
    *,
    cache: Optional[AnswerCache] = None,
    model: str = LLM_MODEL,
    batch_size: int = BATCH_SIZE,
) -> dict[str, TourPlaces]:
    """Read the places out of every title, asking only about unseen ones.

    Returns one entry per distinct title. A title the model could not be asked
    about — no API key, or a failed request — is simply absent from the result,
    so a caller can tell "nothing was found" from "nothing was asked".

    The cache is saved before returning, so a run that is interrupted later
    still keeps what it learned.

    Example:
        >>> found = read_places(["تور کیش از اهواز هتل هلیا"])
        >>> found["تور کیش از اهواز هتل هلیا"]
        TourPlaces(destinations=['کیش'], origin='اهواز')
    """
    cache = cache or AnswerCache()
    unique = list(dict.fromkeys(title for title in titles if title and title.strip()))

    found: dict[str, TourPlaces] = {}
    pending: list[str] = []

    for title in unique:
        stored = cache.get(title, model=model, prompt_version=PROMPT_VERSION)
        if stored is None:
            pending.append(title)
        else:
            found[title] = _verified(
                title, stored.get("destinations", []), stored.get("origin")
            )

    if pending and is_available():
        for start in range(0, len(pending), batch_size):
            batch = pending[start: start + batch_size]
            try:
                reply = ask(build_messages(batch), TourEntities, model=model)
            except Exception as error:
                # One failed batch must not lose the ones that worked, so the
                # failure is reported and the rest of the run continues.
                _LOG.warning(
                    "asking about %d titles failed: %s: %s",
                    len(batch),
                    type(error).__name__,
                    error,
                )
                continue

            for entry in reply.results:
                position = entry.index - 1
                if not 0 <= position < len(batch):
                    continue
                title = batch[position]
                answer = {"destinations": entry.destinations, "origin": entry.origin}
                cache.put(title, answer, model=model, prompt_version=PROMPT_VERSION)
                found[title] = _verified(title, entry.destinations, entry.origin)

        cache.save()

    return found


def fill_page_places(
    pages: list[ExtractedPage],
    *,
    cache: Optional[AnswerCache] = None,
    model: str = LLM_MODEL,
) -> int:
    """Fill ``destination``, ``destinations`` and ``origin`` across many pages.

    This runs as a pass over pages rather than inside the extraction of one,
    because the question is asked in batches: the instructions cost twenty-five
    times what a title does, so one call per page would spend fifty times what
    one call per fifty pages spends for the same answers.

    Pages whose title was never asked about are left untouched — a page with no
    title, or a run with no API key. Where the model was asked and named
    nothing, the attempt is recorded, so a later reader can tell a place that
    was looked for and absent from one nothing has ever looked for.

    A page whose destinations were already read from its own title is skipped
    before the question is even built. :func:`crawler.extraction.readers.places
    .read_title_places` answers 22 of the 23 real titles for nothing, so asking
    about them again would pay for answers we hold and risk replacing a read
    fact with a guess. What reaches the model is the residue.

    Returns how many pages gained a destination, so a caller can report it.

    Example:
        >>> from crawler.extraction.models import ExtractedField, METHOD_HEURISTIC_HTML
        >>> page = ExtractedPage(url="https://x.com/tour/1", page_fields={
        ...     "tour_name": ExtractedField("تور کیش از اهواز هتل هلیا", METHOD_HEURISTIC_HTML),
        ... })
        >>> fill_page_places([page])
        1
        >>> page.page_fields["destinations"].value, page.page_fields["origin"].value
        (['کیش'], 'اهواز')
    """
    titles: list[str] = []
    for page in pages:
        if page.page_fields.get("destinations") is not None:
            continue  # the title already gave it up, for free
        field = page.page_fields.get("tour_name")
        if field is not None and isinstance(field.value, str):
            titles.append(field.value)

    if not titles:
        return 0

    places = read_places(titles, cache=cache, model=model)
    filled = 0

    for page in pages:
        field = page.page_fields.get("tour_name")
        if field is None or not isinstance(field.value, str):
            continue
        found = places.get(field.value)
        if found is None:
            continue  # never asked — leave the page as it was

        if found.destinations:
            page.page_fields["destinations"] = ExtractedField(
                value=found.destinations, method=METHOD_LLM, source_key="tour_name"
            )
            page.page_fields["destination"] = ExtractedField(
                value=found.display, method=METHOD_LLM, source_key="tour_name"
            )
            filled += 1
        if found.origin:
            page.page_fields["origin"] = ExtractedField(
                value=found.origin, method=METHOD_LLM, source_key="tour_name"
            )

        for name in ("destination", "destinations", "origin"):
            if name not in page.page_fields:
                record_attempt(page.page_attempts, name, f"{METHOD_LLM}:tour_name")
        page.unresolved_page_fields = unresolved(page.page_fields, PAGE_FIELDS)

    return filled
