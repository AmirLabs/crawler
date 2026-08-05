"""Data models for step 2: choosing which URLs are tour pages.

Step 2 answers one question — out of every URL a site has, which ones are tour
detail pages worth opening? These dataclasses hold that answer plus the evidence
behind it, so a bad decision can be inspected instead of guessed at.

The evidence matters as much as the result. A site that yields 3 tour URLs out of
4,000 is almost certainly a broken rule rather than a small site, and that only
shows up if the counts and the rejected samples are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# Where the tour URL list came from. Ordered cheapest first, matching the same
# idea as the extraction ladder: never do expensive work a cheaper source
# already did.
SOURCE_API = "api"              # the site's own JSON API listed them
SOURCE_SITEMAP = "sitemap"      # pulled from sitemap.xml
SOURCE_CRAWL = "crawl"          # followed links, because nothing better existed
SOURCE_NONE = "none"            # nothing worked


@dataclass
class UrlGroup:
    """A set of URLs that share the same path shape.

    Grouping by shape is what makes this work on fifty sites without per-site
    rules. ``/tour/d/65694/kuala-lumpur`` and ``/tour/d/71230/dubai`` collapse to
    one shape, ``/tour/d/<num>/<slug>``, so a whole content type can be judged
    once instead of URL by URL.
    """

    shape: str                                   # "/tour/d/<num>/<slug>"
    urls: list[str] = field(default_factory=list)
    score: float = 0.0
    # Why it scored that way, in plain words — "path holds 'tour'", "looks like
    # a listing". Kept so a wrong pick can be explained without a debugger.
    reasons: list[str] = field(default_factory=list)
    is_selected: bool = False

    @property
    def count(self) -> int:
        return len(self.urls)


@dataclass
class UrlDiscovery:
    """Everything step 2 learned about one site."""

    domain: str
    source: str = SOURCE_NONE

    # The output the next step consumes.
    tour_urls: list[str] = field(default_factory=list)

    # --- evidence ---------------------------------------------------------- #
    total_seen: int = 0          # URLs examined before any filtering
    total_kept: int = 0          # == len(tour_urls)
    groups: list[UrlGroup] = field(default_factory=list)
    rejected_samples: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # True when the URL list is known to be incomplete — a sampled sitemap
    # index, a paged API we stopped early, a crawl that hit its limit. The
    # daily refresh needs to know it has not seen everything.
    is_partial: bool = False

    @property
    def keep_rate(self) -> float:
        """Share of seen URLs that were kept, as a sanity signal.

        Near zero means the rule found almost nothing and probably broke. Near
        one means it kept nearly everything and probably is not filtering.
        """
        return (self.total_kept / self.total_seen) if self.total_seen else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
