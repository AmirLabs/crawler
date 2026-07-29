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
    url_count: int = 0                    # how many page URLs it advertises
    sample_urls: list[str] = field(default_factory=list)


@dataclass
class RenderingInfo:
    type: str = "unknown"                 # "static" | "dynamic" | "unknown"
    js_framework_hint: Optional[str] = None
    links_in_raw_html: int = 0            # links visible without running JS
    links_after_render: Optional[int] = None  # links after running JS (None if not tested)
    needs_playwright: bool = False
    notes: Optional[str] = None


@dataclass
class AntiBotInfo:
    cloudflare: bool = False
    captcha_detected: bool = False
    blocks_bots: bool = False             # got 403/429 or a challenge page
    notes: list[str] = field(default_factory=list)


@dataclass
class TechInfo:
    server: Optional[str] = None
    powered_by: Optional[str] = None
    cms: Optional[str] = None


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

    recommended_strategy: str = "unknown"  # "sitemap" | "static-crawl" | "dynamic-crawl" | "blocked"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
