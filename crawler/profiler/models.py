"""Data models for the Site Profiler output.

These dataclasses define the *shape* of a site profile. Keeping the schema in
one place means every part of the system agrees on what a "profile" looks like,
and we can serialize it to JSON without guessing field names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class RobotsInfo:
    exists: bool = False
    fetched_url: Optional[str] = None
    allows_crawling: bool = True          # default: assume allowed if no rules found
    crawl_delay: Optional[float] = None   # seconds between requests, if declared
    sitemaps: list[str] = field(default_factory=list)  # sitemaps listed in robots.txt


@dataclass
class SitemapInfo:
    exists: bool = False
    location: Optional[str] = None
    type: Optional[str] = None            # "index" (nested) or "urlset" (flat list)
    url_count: int = 0                    # how many *page* URLs it advertises
    sample_urls: list[str] = field(default_factory=list)
    child_sitemap_count: int = 0          # for an index: how many nested sitemaps
    url_count_is_partial: bool = False    # True if we sampled instead of reading every child


@dataclass
class RenderingInfo:
    type: str = "unknown"                 # "static" | "dynamic" | "unknown"
    js_framework_hint: Optional[str] = None
    links_in_raw_html: int = 0            # links visible without running JS
    links_after_render: Optional[int] = None  # links after running JS (None if not tested)
    needs_playwright: bool = False
    # Whether rendering a *tour page* of this site actually yields more offers.
    # Separate from ``needs_playwright``, which is judged from the home page and
    # answers a different question. ``None`` means it has not been tried yet;
    # step 3 tries it once per site on the first page that looks like it might
    # need it, and follows the answer for the rest.
    #
    # It is worth measuring rather than guessing: rendering costs 6 to 13
    # seconds a page and on nahalgasht.com it produced *fewer* rows than the
    # plain HTML did, while on alefbatour.com it was the difference between no
    # hotels and all of them.
    render_helps: Optional[bool] = None
    notes: Optional[str] = None


@dataclass
class AntiBotInfo:
    cloudflare: bool = False
    captcha_detected: bool = False
    blocks_bots: bool = False             # got 403/429 or a challenge page
    notes: list[str] = field(default_factory=list)


@dataclass
class UrlShapeInfo:
    """Which of this site's URL shapes actually lead to a tour.

    No shape means the same thing on two sites: ``/tours/<slug>`` is a real
    tour on vanrosh.com and a destination guide on tahagasht.com, and
    alefbatour.com publishes both ``/tour/d/<num>/<slug>`` (515 tours) and
    ``/tour/<slug>`` (81 guides — /tour/bodrum, /tour/marmaris). So this is
    measured per site rather than decided in the rules, the same way
    :attr:`RenderingInfo.render_helps` is.

    A site can mean more than one shape at once: vanrosh.com sells tours under
    ``/tours/<slug>`` and ``/tours/<num>`` alike, 93 and 57 URLs, both real.
    Hence lists rather than one winning shape.

    ``tested`` stays False when nothing could be sampled — an unreachable site,
    or a fetch that failed — and the caller then keeps every URL it found. A
    site that answers slowly should lose speed, not its whole tour list.
    """

    tour_shapes: list[str] = field(default_factory=list)
    guide_shapes: list[str] = field(default_factory=list)
    tested: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class TechInfo:
    cms: Optional[str] = None        # "WordPress", "Jomla", ...
    has_json_api: bool = False       # e.g. /wp-json responds with JSON
    json_api_url: Optional[str] = None

@dataclass
class SiteProfile:
    domain: str
    reachable: bool = False
    status_code: Optional[int] = None
    final_url: Optional[str] = None       # after following redirects
    error: Optional[str] = None

    robots: RobotsInfo = field(default_factory=RobotsInfo)
    sitemap: SitemapInfo = field(default_factory=SitemapInfo)
    rendering: RenderingInfo = field(default_factory=RenderingInfo)
    anti_bot: AntiBotInfo = field(default_factory=AntiBotInfo)
    tech: TechInfo = field(default_factory=TechInfo)
    url_shapes: UrlShapeInfo = field(default_factory=UrlShapeInfo)

    recommended_strategy: str = "unknown"  # "sitemap" | "static-crawl" | "dynamic-crawl" | "blocked"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
