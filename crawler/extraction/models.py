"""Data models for step 3: what one fetched tour page yielded.

Step 3 answers: for this one URL, what did we manage to read with confidence,
and what is still missing? An :class:`ExtractedPage` holds both — the resolved
fields plus an honest list of what nothing could fill — so step 4/5 sends only
the missing pieces to the LLM instead of the whole page.

Design rule, carried over from :mod:`crawler.filtering.models`: a field that
cannot be resolved with confidence stays out of ``fields`` rather than being
filled with a guess. A wrong value hides; a missing one shows up in
``unresolved_fields`` and can be traced through ``method`` for the ones that
did resolve.
"""

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
    # The key/selector that produced it, kept for debugging a wrong mapping
    # without re-fetching the page — e.g. "title" or ".tour-price".
    source_key: Optional[str] = None


@dataclass
class ExtractedPage:
    """Everything step 3 learned about one tour URL.

    Holds the raw material (``html_raw``, ``embedded_json``) alongside the
    fields that were confidently resolved from it, and is explicit about what
    was not resolved so the next stage knows exactly what to ask an LLM for.

    Example:
        >>> page = ExtractedPage(
        ...     url="https://avisatravel.com/tour/d/65694/kuala-lumpur",
        ...     fetch_method=FETCH_HTTP,
        ...     status_code=200,
        ...     embedded_json=[{"title": "تور کوالالامپور 7 شب"}],
        ...     fields={
        ...         "tour_name": ExtractedField(
        ...             value="تور کوالالامپور 7 شب",
        ...             method=METHOD_EXACT_KEY,
        ...             source_key="title",
        ...         ),
        ...     },
        ...     unresolved_fields=["departure_date", "hotel_name"],
        ... )
        >>> page.fill_rate
        0.333...
    """

    url: str
    fetch_method: str = FETCH_HTTP
    status_code: Optional[int] = None
    error: Optional[str] = None

    # Raw material kept for traceability and for the (future) heuristic-HTML
    # layer and LLM fallback to work from without re-fetching.
    html_raw: Optional[str] = None
    embedded_json: list[dict] = field(default_factory=list)

    # What step 3 resolved, keyed by the target schema field name (e.g.
    # "tour_name", "hotel_name") so step 4 can read this dict directly.
    fields: dict[str, ExtractedField] = field(default_factory=dict)

    # Target field names nothing could resolve. Empty does not mean "this page
    # has no more data" — it means every field we looked for was found.
    unresolved_fields: list[str] = field(default_factory=list)

    fetched_at: Optional[str] = None    # ISO timestamp, set by the fetch layer
    content_hash: Optional[str] = None  # skip unchanged pages on a daily refresh
    notes: list[str] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        """Share of looked-for fields that were resolved, as a sanity signal.

        Mirrors :attr:`crawler.filtering.models.UrlDiscovery.keep_rate`: near
        zero means the extraction rules found almost nothing on this page and
        probably do not match its structure; near one means most target
        fields were resolved with confidence.

        Example:
            >>> ExtractedPage(
            ...     url="https://example.com/tour/1",
            ...     fields={"tour_name": ExtractedField(value="x", method=METHOD_EXACT_KEY)},
            ...     unresolved_fields=["hotel_name"],
            ... ).fill_rate
            0.5
        """
        total = len(self.fields) + len(self.unresolved_fields)
        return (len(self.fields) / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
