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
        >>> round(page.fill_rate, 2)
        0.4
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

    # The repeating part of the page: one entry per bookable combination found.
    # Empty means no repeating structure was recognised, which on a tour page
    # is itself a signal that the rules did not match this layout.
    rows: list[ExtractedRow] = field(default_factory=list)

    fetched_at: Optional[str] = None    # ISO timestamp, set by the fetch layer
    content_hash: Optional[str] = None  # skip unchanged pages on a daily refresh
    notes: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def fill_rate(self) -> float:
        """Share of looked-for fields that were resolved, across page and rows.

        Mirrors :attr:`crawler.filtering.models.UrlDiscovery.keep_rate`: near
        zero means the extraction rules found almost nothing on this page and
        probably do not match its structure; near one means most target fields
        were resolved with confidence.

        Rows are counted field by field rather than as whole units, so a page
        whose hotel names all resolved but whose prices all failed reads as
        half-filled instead of fully filled.

        Example:
            >>> ExtractedPage(
            ...     url="https://example.com/tour/1",
            ...     page_fields={"tour_name": ExtractedField(value="x", method=METHOD_EXACT_KEY)},
            ...     unresolved_page_fields=["origin"],
            ... ).fill_rate
            0.5
        """
        resolved = len(self.page_fields)
        total = resolved + len(self.unresolved_page_fields)

        for row in self.rows:
            resolved += len(row.fields)
            total += len(row.fields) + len(row.unresolved_fields)

        return (resolved / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
