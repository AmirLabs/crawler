"""Step 2: URL eligibility and normalization rules.

Given a site profile, this module answers: which URLs are tour detail pages worth
opening? It reads the API or sitemap, learns URL shapes, scores which shape is
tours, and returns the list.

The core idea: group URLs by path shape, score each group, pick the winner. This
works across 50 sites without per-site rules because we judge patterns instead of
hardcoding paths.
"""

from __future__ import annotations

import re
import asyncio
from urllib.parse import urlparse, urljoin
from collections import defaultdict
from typing import Optional

import httpx

from crawler.config import DEFAULT_TIMEOUT_SECONDS, USER_AGENT
from crawler.profiler.models import SiteProfile
from crawler.filtering.models import (
    UrlDiscovery,
    UrlGroup,
    SOURCE_API,
    SOURCE_SITEMAP,
    SOURCE_NONE,
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/html,*/*;q=0.9",
}

# Words that often appear in tour detail URLs but rarely in listing pages.
TOUR_WORDS = (
    "tour", "package", "trip", "travel", "product",
    "تور", "پکیج", "سفر",
)

# Path segments that mark a listing or category page rather than one tour.
# Matched against whole segments, not substrings: "page" as a substring would
# also sink "package", which is one of the tour words above.
LISTING_SIGNS = (
    "category", "categories", "archive", "tag", "tags",
    "page", "p", "search", "filter",
    "دسته", "آرشیو", "برچسب",
)

MAX_SITEMAP_URLS = 5000  # stop reading a sitemap after this many
MAX_API_PAGES = 50       # stop paging an API after this many pages
MIN_GROUP_SCORE = 0.5    # a group must score at least this to be selected


# A slug that is really an identifier: "65694", "d-2291", "a1b2c3d4". The test
# is that digits dominate the segment — a title that merely contains a number
# ("bali-5-nights", "slug-100") is still a slug, and treating it as an id would
# split one content type into two groups and drop whichever lost.
_NUMERIC = re.compile(r"^\d+$")


def _is_identifier(seg: str) -> bool:
    digits = sum(c.isdigit() for c in seg)
    letters = sum(c.isalpha() for c in seg)
    return digits >= 3 and digits >= letters


def _group_by_shape(urls: list[str]) -> dict[str, list[str]]:
    """Group URLs by path shape, learning which segments vary from the set.

    A segment is treated as variable when many different values appear in that
    position across URLs of the same length and prefix. That is what separates
    ``/tour/d/<num>/<slug>`` (two fixed words, two varying parts) from
    ``/blog/<slug>`` without knowing anything about the site in advance.
    """
    # First bucket by depth and by the first segment, so unrelated sections are
    # never compared against each other.
    buckets: dict[tuple, list[list[str]]] = defaultdict(list)
    originals: dict[tuple, list[str]] = defaultdict(list)

    for url in urls:
        parts = [p for p in urlparse(url).path.split("/") if p]
        if not parts:
            continue
        key = (len(parts), parts[0].lower())
        buckets[key].append(parts)
        originals[key].append(url)

    shaped: dict[str, list[str]] = defaultdict(list)

    for key, rows in buckets.items():
        depth = key[0]
        # For each position, how many distinct values does it take?
        distinct = [len({r[i].lower() for r in rows}) for i in range(depth)]
        total = len(rows)

        for parts, url in zip(rows, originals[key]):
            out = []
            for i, seg in enumerate(parts):
                if _NUMERIC.match(seg):
                    out.append("<num>")
                elif _is_identifier(seg):
                    out.append("<id>")
                elif total >= 4 and distinct[i] > max(2, total * 0.5):
                    # Varies across most URLs here: a title, not a section name.
                    out.append("<slug>")
                else:
                    out.append(seg.lower())  # fixed word: "tour", "d", "blog"
            shaped["/" + "/".join(out)].append(url)

    return shaped


def _score_group(group: UrlGroup, total_urls: int = 0) -> float:
    """Judge whether this URL shape is tour detail pages.

    Writes the score and the plain-word reasons onto ``group`` and also returns
    the score. Signals are weighed rather than combined into a single rule,
    because no one signal holds on all fifty sites: some sites number their
    tours, some do not, and some put the tour word in a parent path only.
    """
    score = 0.0
    reasons: list[str] = []
    shape_low = group.shape.lower()

    matched_word = next((w for w in TOUR_WORDS if w in shape_low), None)
    if matched_word:
        score += 0.4
        reasons.append(f"path holds '{matched_word}'")

    # A varying id or slug in the path means one page per thing — a detail page.
    if "<num>" in group.shape or "<id>" in group.shape:
        score += 0.3
        reasons.append("numbered series")
    elif "<slug>" in group.shape:
        score += 0.2
        reasons.append("one page per item")

    # Listing and category pages hold links, not a tour. Strong enough to sink a
    # group on its own, since "tour-category" also matches the tour word above.
    # Segments are split on "-" too, so "tour-category" is caught while
    # "package" is not mistaken for the "page" marker.
    segments = {
        piece
        for seg in shape_low.strip("/").split("/")
        for piece in [seg, *seg.split("-")]
    }
    listing = next((s for s in LISTING_SIGNS if s in segments), None)
    if listing:
        score -= 0.5
        reasons.append(f"listing pattern ('{listing}')")

    if group.shape.count("/") >= 2:
        score += 0.1
        reasons.append("reasonable depth")

    # A content type has many pages; "/about-us" has one. Relative to the site,
    # so it works for a 60-tour site and a 3,000-tour site alike.
    if total_urls and group.count >= max(5, total_urls * 0.05):
        score += 0.2
        reasons.append(f"{group.count} pages share this shape")

    group.score = max(0.0, min(1.0, score))
    group.reasons = reasons
    return group.score


async def _discover_from_api(
    client: httpx.AsyncClient, profile: SiteProfile
) -> UrlDiscovery:
    """Pull tour URLs straight from the site's JSON API."""
    result = UrlDiscovery(domain=profile.domain, source=SOURCE_API)

    if not profile.tech.has_json_api or not profile.tech.json_api_url:
        result.notes.append("no working API found")
        return result

    api_url = profile.tech.json_api_url
    page = 1
    seen = set()

    while page <= MAX_API_PAGES:
        try:
            resp = await client.get(api_url, params={"per_page": 100, "page": page})
        except Exception as exc:
            result.notes.append(f"API request failed: {exc}")
            break

        if resp.status_code != 200:
            result.notes.append(f"API returned {resp.status_code} on page {page}")
            break

        try:
            data = resp.json()
        except Exception:
            result.notes.append(f"API page {page} is not JSON")
            break

        if not isinstance(data, list):
            result.notes.append(f"API page {page} is not a list")
            break

        if not data:
            break  # empty page = end

        for item in data:
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if link and link not in seen:
                result.tour_urls.append(link)
                seen.add(link)

        page += 1

        # Check x-wp-total to see if we hit the end
        total_header = resp.headers.get("x-wp-total")
        if total_header and len(seen) >= int(total_header):
            break

    result.total_seen = len(seen)
    result.total_kept = len(result.tour_urls)
    result.is_partial = (page > MAX_API_PAGES)

    if result.is_partial:
        result.notes.append(f"stopped at page {MAX_API_PAGES}")

    return result


def _extract_locs(xml: str) -> list[str]:
    """Pull <loc> values from a sitemap."""
    return [m.strip() for m in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I | re.S)]


async def _discover_from_sitemap(
    client: httpx.AsyncClient, profile: SiteProfile
) -> UrlDiscovery:
    """Read sitemap, group URLs by shape, pick the tour group."""
    result = UrlDiscovery(domain=profile.domain, source=SOURCE_SITEMAP)

    if not profile.sitemap.exists or not profile.sitemap.location:
        result.notes.append("no sitemap found")
        return result

    urls_seen: set[str] = set()

    # Read up to MAX_SITEMAP_URLS
    try:
        resp = await client.get(profile.sitemap.location)
        if resp.status_code == 200 and "<" in resp.text:
            locs = _extract_locs(resp.text)

            # If it's an index, read a few children
            if "<sitemapindex" in resp.text.lower():
                for child_url in locs[:5]:
                    try:
                        child_resp = await client.get(child_url)
                        if child_resp.status_code == 200:
                            child_locs = _extract_locs(child_resp.text)
                            urls_seen.update(child_locs[:MAX_SITEMAP_URLS])
                            if len(urls_seen) >= MAX_SITEMAP_URLS:
                                result.is_partial = True
                                break
                    except Exception:
                        continue
            else:
                urls_seen.update(locs[:MAX_SITEMAP_URLS])
                if len(locs) > MAX_SITEMAP_URLS:
                    result.is_partial = True
    except Exception as exc:
        result.notes.append(f"sitemap fetch failed: {exc}")
        return result

    result.total_seen = len(urls_seen)

    # Group by shape, learning the varying segments from the whole set.
    shape_map = _group_by_shape(sorted(urls_seen))

    # Score each group. The winner is the shape that looks most like one page
    # per tour, judged on signals rather than a hardcoded path.
    groups = [UrlGroup(shape=shape, urls=url_list) for shape, url_list in shape_map.items()]
    for g in groups:
        _score_group(g, total_urls=result.total_seen)

    groups.sort(key=lambda x: (x.score, x.count), reverse=True)
    if groups and groups[0].score >= MIN_GROUP_SCORE:
        winner = groups[0]
        winner.is_selected = True
        result.tour_urls = sorted(winner.urls)
        result.total_kept = len(winner.urls)
    else:
        best = groups[0].score if groups else 0.0
        result.notes.append(f"no group scored above {MIN_GROUP_SCORE} (best {best:.2f})")

    result.groups = groups
    result.rejected_samples = [g.urls[0] for g in groups if not g.is_selected][:5]

    return result


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def discover_tour_urls(
    profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> UrlDiscovery:
    """Find tour URLs from a profiled site.

    Tries API first (cheapest, exact), then sitemap (good), then gives up. A
    future version could follow links when neither works, but two paths cover
    the known sites.
    """
    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        # API path
        if profile.tech.has_json_api:
            result = await _discover_from_api(client, profile)
            if result.total_kept > 0:
                return result

        # Sitemap path
        if profile.sitemap.exists:
            result = await _discover_from_sitemap(client, profile)
            if result.total_kept > 0:
                return result

    # Nothing worked
    return UrlDiscovery(domain=profile.domain, source=SOURCE_NONE)


def discover_tour_urls_sync(profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> UrlDiscovery:
    """Synchronous wrapper."""
    return asyncio.run(discover_tour_urls(profile, timeout=timeout))
