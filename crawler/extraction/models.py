from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


# How a page was fetched. Chosen per-URL from the site profile's recommended
# strategy: cheap HTTP when the page is server-rendered, a real browser only
# when the profiler found the content needs JS to appear.
FETCH_HTTP = "http"
FETCH_PLAYWRIGHT = "playwright"

# How one field's value was resolved, ordered cheapest/most-certain first.
# Mirrors the source ordering in crawler.filtering.models (SOURCE_API before
# SOURCE_SITEMAP before SOURCE_CRAWL): never claim an expensive or uncertain
# method did the work when a cheaper, more certain one already did.
METHOD_EXACT_KEY = "exact_key_match"      # JSON key literally named "title", "price", ...
METHOD_FUZZY_KEY = "fuzzy_key_match"      # JSON key matched after normalizing/synonyms
METHOD_HEURISTIC_HTML = "heuristic_html"  # CSS selector / regex pattern learned across sites
METHOD_LLM = "llm"                        # sent to the LLM because nothing else resolved it


def record_attempt(attempts: dict[str, list[str]], field_name: str, method: str) -> None:
    """Note that ``method`` was tried for ``field_name`` and came back empty.

    Only failures are worth recording: a method that worked is already named on
    the resolved :class:`ExtractedField`. What this adds is the difference
    between a field nothing could find and a field nothing looked for — the
    first needs a different approach, the second just needs a rule.

    ``method`` may be qualified with what was searched, as ``method:scope``,
    so a later stage can be told where the gap is rather than only that one
    exists: ``"heuristic_html:row_text"`` says the row's own text was read.

    Example:
        >>> attempts: dict[str, list[str]] = {}
        >>> record_attempt(attempts, "hotel_name", "heuristic_html:row_text")
        >>> record_attempt(attempts, "hotel_name", "heuristic_html:ancestors")
        >>> attempts
        {'hotel_name': ['heuristic_html:row_text', 'heuristic_html:ancestors']}
    """
    tried = attempts.setdefault(field_name, [])
    if method not in tried:
        tried.append(method)


@dataclass
class ExtractedField:
    """One resolved value for one target schema field, with how we got it.

    ``value`` is left as ``Any`` on purpose: at this stage a price might still
    be the raw string ``"110,000,000"`` rather than a parsed ``float`` — schema
    step 4 owns turning this into a typed :class:`~crawler.schema.output_models.Price`.
    This object's job is only to say what was found and how sure we are of the
    source, not to normalize it.

    Example:
        >>> ExtractedField(value="تور کوالالامپور 7 شب", method=METHOD_EXACT_KEY, source_key="title")
        ExtractedField(value='تور کوالالامپور 7 شب', method='exact_key_match', source_key='title')
    """

    value: Any
    method: str  # one of the METHOD_* constants
    # The key/path/selector that produced it, kept for debugging a wrong mapping
    # without re-fetching the page — e.g. "title", "$.hotels[*].price_1".
    source_key: Optional[str] = None


@dataclass
class ExtractedRow:
    """One repeating unit on the page, with its fields kept together.

    A row is whatever the page repeats — usually a hotel option, a departure
    date, or the combination of the two. Everything inside one row is known to
    belong together, which is what makes it safe to hand to
    :class:`~crawler.schema.output_models.TourOffer`.

    ``source_path`` records which part of the page produced the row, so a row
    set that turns out to be the wrong one (a list of related tours rather than
    this tour's hotels, say) can be identified without re-fetching.

    Example:
        >>> row = ExtractedRow(
        ...     fields={
        ...         "hotel_name": ExtractedField("هتل کورت", METHOD_EXACT_KEY, "$.hotels[*].title"),
        ...         "adult_price": ExtractedField(110000000, METHOD_EXACT_KEY, "$.hotels[*].price_1"),
        ...     },
        ...     unresolved_fields=["board_type"],
        ...     source_path="$.hotels[*]",
        ...     index=0,
        ... )
        >>> row.fields["hotel_name"].value
        'هتل کورت'
    """

    fields: dict[str, ExtractedField] = field(default_factory=dict)
    unresolved_fields: list[str] = field(default_factory=list)

    # For the unresolved fields, what was tried and came back empty. A name
    # absent from here was never looked for, which is a different problem and
    # deserves a different answer — see :func:`record_attempt`.
    attempts: dict[str, list[str]] = field(default_factory=dict)

    # Which repeating structure this row came from, and its position in it.
    source_path: Optional[str] = None  # "$.hotels[*]"
    index: Optional[int] = None        # 0, 1, 2, ... within that structure


@dataclass
class ExtractedPage:
    """Everything step 3 learned about one tour URL.

    Holds the raw material (``html_raw``, ``embedded_json``) alongside the
    page-level fields and the rows resolved from it, and is explicit about what
    was not resolved so the next stage knows exactly what to ask an LLM for.

    Example:
        >>> page = ExtractedPage(
        ...     url="https://avisatravel.com/tour/d/65694/kuala-lumpur",
        ...     fetch_method=FETCH_HTTP,
        ...     status_code=200,
        ...     page_fields={
        ...         "tour_name": ExtractedField(
        ...             value="تور کوالالامپور 7 شب",
        ...             method=METHOD_EXACT_KEY,
        ...             source_key="$.title",
        ...         ),
        ...     },
        ...     unresolved_page_fields=["origin", "destination"],
        ...     rows=[
        ...         ExtractedRow(
        ...             fields={"hotel_name": ExtractedField("هتل کورت", METHOD_EXACT_KEY)},
        ...             unresolved_fields=["board_type"],
        ...         ),
        ...     ],
        ... )
        >>> page.row_count
        1
        >>> round(page.page_fill_rate, 2), round(page.row_fill_rate, 2)
        (0.33, 0.5)
    """

    url: str
    fetch_method: str = FETCH_HTTP
    status_code: Optional[int] = None
    error: Optional[str] = None

    # Raw material kept for traceability and for the (future) heuristic-HTML
    # layer and LLM fallback to work from without re-fetching.
    html_raw: Optional[str] = None
    embedded_json: list[dict] = field(default_factory=list)

    # Facts that hold for the whole page, keyed by the target schema field name
    # (e.g. "tour_name", "origin") so step 4 can read this dict directly.
    page_fields: dict[str, ExtractedField] = field(default_factory=dict)

    # Page-level field names nothing could resolve. Empty does not mean "this
    # page has no more data" — it means every field we looked for was found.
    unresolved_page_fields: list[str] = field(default_factory=list)

    # What was tried for those, and failed. See :func:`record_attempt`.
    page_attempts: dict[str, list[str]] = field(default_factory=dict)

    # The repeating part of the page: one entry per bookable combination found.
    # Empty means no repeating structure was recognised, which on a tour page
    # is itself a signal that the rules did not match this layout.
    rows: list[ExtractedRow] = field(default_factory=list)

    # Offers that were read but have already departed. Kept apart rather than
    # dropped so an empty page can explain itself: "nothing was read" and "four
    # departures were read and all four have left" need different fixes.
    expired_rows: list[ExtractedRow] = field(default_factory=list)

    fetched_at: Optional[str] = None    # ISO timestamp, set by the fetch layer
    content_hash: Optional[str] = None  # skip unchanged pages on a daily refresh
    notes: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def sellable(self) -> bool:
        """Whether this page still offers anything a customer could book.

        What makes a page worth keeping, and the cheapest thing for a daily run
        to test before spending anything further on it.

        Example:
            >>> ExtractedPage(url="https://x.com/t/1").sellable
            False
            >>> ExtractedPage(url="https://x.com/t/1", rows=[ExtractedRow()]).sellable
            True
        """
        return bool(self.rows)

    @property
    def page_fill_rate(self) -> float:
        """Share of the page-level fields that were resolved.

        The denominator is fixed — every tour page is asked for the same
        page-level fields — so this number means the same thing on every page
        of every site and can be compared across them.

        Example:
            >>> ExtractedPage(
            ...     url="https://example.com/tour/1",
            ...     page_fields={"tour_name": ExtractedField(value="x", method=METHOD_EXACT_KEY)},
            ...     unresolved_page_fields=["origin"],
            ... ).page_fill_rate
            0.5
        """
        resolved = len(self.page_fields)
        total = resolved + len(self.unresolved_page_fields)
        return (resolved / total) if total else 0.0

    @property
    def row_fill_rate(self) -> Optional[float]:
        """Share of the row-level fields that were resolved, or ``None``.

        ``None`` means there were no rows to measure, which is not the same as
        rows that came out empty. Reporting it as a zero would read as "the
        rules failed on every row"; reporting it as a one — which an earlier
        combined rate effectively did — would read as success. Neither is true
        of a page that produced no rows at all, and
        :attr:`row_count` is what says so.

        Fields are counted one by one across every row rather than by whole
        rows, so a page whose hotel names all resolved but whose prices all
        failed reads as half-filled rather than fully filled.

        Example:
            >>> ExtractedPage(
            ...     url="https://example.com/tour/1",
            ...     rows=[ExtractedRow(
            ...         fields={"hotel_name": ExtractedField(value="x", method=METHOD_EXACT_KEY)},
            ...         unresolved_fields=["board_type", "price"],
            ...     )],
            ... ).row_fill_rate
            0.3333333333333333
            >>> ExtractedPage(url="https://example.com/tour/1").row_fill_rate is None
            True
        """
        if not self.rows:
            return None

        resolved = sum(len(row.fields) for row in self.rows)
        total = resolved + sum(len(row.unresolved_fields) for row in self.rows)
        return (resolved / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
