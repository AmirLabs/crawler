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
from urllib.parse import unquote, urljoin, urlparse
from collections import defaultdict
from typing import Optional

import httpx

from crawler.config import DEFAULT_TIMEOUT_SECONDS, USER_AGENT
from crawler.extraction.extractor import _resolve_crawl_delay
from crawler.profiler.models import SiteProfile
from crawler.filtering.models import (
    UrlDiscovery,
    UrlGroup,
    SOURCE_API,
    SOURCE_SITEMAP,
    SOURCE_CRAWL,
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

# How many pages the link walk may open per site. Fixed rather than
# proportional, so the cost of checking a sitemap's honesty does not grow with
# the site. Sixty was enough to find 144 unlisted tours on bisanseir.com.
CRAWL_BUDGET_PAGES = 60
MIN_GROUP_SCORE = 0.5    # a group must score at least this to be selected

# How far below the best-scoring shape another shape may sit and still be kept.
_SHAPE_MARGIN = 0.1

_NUMERIC = re.compile(r"^\d+$")

# When a path segment counts as a varying title rather than a fixed section
# name. A slug repeats wherever a tour has several departures, so the share has
# to sit low: radingasht.com's 407 tour URLs share 26 slugs, and at 10% they
# were read as 26 fixed words instead of one shape.
_MIN_DISTINCT_FOR_SLUG = 4
_SLUG_SHARE = 0.02


def _is_identifier(seg: str) -> bool:
    digits = sum(c.isdigit() for c in seg)
    letters = sum(c.isalpha() for c in seg)
    return digits >= 3 and digits >= letters


def group_by_shape(urls: list[str]) -> dict[str, list[str]]:
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
        # Percent-encoding has to come off before the shape is read. A Persian
        # slug arrives as "%d8%aa%d9%88%d8%b1-..." — a run of hex digits that
        # _is_identifier counts as a number, so the same content type split
        # into an "<id>" group and a dozen literal ones, and only the largest
        # was kept. bisanseir.com lost 60 tour URLs to exactly that.
        parts = [p for p in unquote(urlparse(url).path).split("/") if p]
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
                elif distinct[i] >= max(_MIN_DISTINCT_FOR_SLUG, total * _SLUG_SHARE):
                    # Varies enough to be a title rather than a section name.
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


# One <url> entry of a sitemap, which is where a date sits beside its address.
# <loc> alone cannot be paired with <lastmod>, since a plain findall over the
# whole document gives two lists that only line up when every entry is dated.
_URL_ENTRY = re.compile(r"<url>(.*?)</url>", re.I | re.S)
_LOC_IN = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)
_LASTMOD_IN = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.I | re.S)


def _extract_dated_locs(xml: str) -> dict[str, str]:
    """Pull ``{url: lastmod}`` from a sitemap, for the entries that carry one.

    Undated entries are absent rather than empty, so a caller can tell a page
    the site dated from one it did not. bisanseir.com and tahagasht.com date
    nothing at all, and their lists simply keep the order they arrived in.

    Example:
        >>> _extract_dated_locs(
        ...     '<url><loc>https://x.com/a</loc><lastmod>2026-08-19</lastmod></url>'
        ...     '<url><loc>https://x.com/b</loc></url>'
        ... )
        {'https://x.com/a': '2026-08-19'}
    """
    dated: dict[str, str] = {}
    for entry in _URL_ENTRY.findall(xml):
        loc = _LOC_IN.search(entry)
        lastmod = _LASTMOD_IN.search(entry)
        if loc and lastmod:
            dated[loc.group(1).strip()] = lastmod.group(1).strip()
    return dated


def _newest_first(urls: list[str], dated: dict[str, str]) -> list[str]:
    """Order URLs by the date the site last changed them, newest first.

    Undated URLs keep their own order and go last: nothing is dropped for
    lacking a date, it simply cannot be ranked.

    Example:
        >>> _newest_first(
        ...     ["https://x.com/old", "https://x.com/undated", "https://x.com/new"],
        ...     {"https://x.com/old": "2024-01-01", "https://x.com/new": "2026-08-19"},
        ... )
        ['https://x.com/new', 'https://x.com/old', 'https://x.com/undated']
    """
    known = [u for u in urls if u in dated]
    unknown = [u for u in urls if u not in dated]
    known.sort(key=lambda u: dated[u], reverse=True)
    return known + unknown


async def _discover_from_sitemap(
    client: httpx.AsyncClient, profile: SiteProfile
) -> UrlDiscovery:
    """Read sitemap, group URLs by shape, pick the tour group."""
    result = UrlDiscovery(domain=profile.domain, source=SOURCE_SITEMAP)

    if not profile.sitemap.exists or not profile.sitemap.location:
        result.notes.append("no sitemap found")
        return result

    urls_seen: set[str] = set()
    dated: dict[str, str] = {}

    # Read up to MAX_SITEMAP_URLS
    try:
        resp = await client.get(profile.sitemap.location)
        if resp.status_code == 200 and "<" in resp.text:
            locs = _extract_locs(resp.text)

            # If it's an index, read its children. Every one of them: reading
            # the first few silently loses whatever the rest hold, and
            # nahalgasht.com lists six.
            if "<sitemapindex" in resp.text.lower():
                for child_url in locs:
                    try:
                        child_resp = await client.get(child_url)
                        if child_resp.status_code == 200:
                            child_locs = _extract_locs(child_resp.text)
                            urls_seen.update(child_locs[:MAX_SITEMAP_URLS])
                            dated.update(_extract_dated_locs(child_resp.text))
                            if len(urls_seen) >= MAX_SITEMAP_URLS:
                                result.is_partial = True
                                result.notes.append(
                                    f"stopped after {MAX_SITEMAP_URLS} URLs; "
                                    f"the index has {len(locs)} sitemaps"
                                )
                                break
                    except Exception:
                        continue
            else:
                urls_seen.update(locs[:MAX_SITEMAP_URLS])
                dated.update(_extract_dated_locs(resp.text))
                if len(locs) > MAX_SITEMAP_URLS:
                    result.is_partial = True
    except Exception as exc:
        result.notes.append(f"sitemap fetch failed: {exc}")
        return result

    selected = _from_urls(profile, sorted(urls_seen), SOURCE_SITEMAP)
    selected.last_modified = dated
    selected.is_partial = result.is_partial
    selected.notes = result.notes + selected.notes
    # Newest first rather than alphabetical: the list is the whole archive, and
    # the order decides what a partial run reaches.
    selected.tour_urls = _newest_first(selected.tour_urls, dated)
    return selected


def _same_host(url: str, host: str) -> bool:
    """Whether ``url`` belongs to ``host``, ignoring a leading ``www.``.

    Example:
        >>> _same_host("https://www.bisanseir.com/tour/x", "bisanseir.com")
        True
        >>> _same_host("https://instagram.com/bisanseir", "bisanseir.com")
        False
    """
    return urlparse(url).netloc.replace("www.", "").lower() == host.replace("www.", "").lower()


def _tourish(url: str) -> bool:
    """Whether a URL's path names a tour, once its encoding is undone.

    Example:
        >>> _tourish("https://bisanseir.com/tour/%D8%AA%D9%88%D8%B1-%D8%A8%D8%A7%DA%A9%D9%88")
        True
        >>> _tourish("https://bisanseir.com/about-us")
        False
    """
    decoded = unquote(urlparse(url).path).lower()
    return any(word in decoded for word in TOUR_WORDS)


async def _discover_from_crawl(
    client: httpx.AsyncClient,
    profile: SiteProfile,
    budget: int,
    delay: float,
) -> list[str]:
    """Walk the site from its home page, collecting tour URLs by following links.

    This exists because a sitemap can be wrong. bisanseir.com lists 276 URLs
    while the site itself carries at least 144 more tour pages, so its sitemap
    describes under half of what it sells — and no amount of filtering recovers
    a URL that was never offered.

    The walk is breadth-first and bounded by ``budget`` pages, which keeps the
    cost fixed rather than proportional to the site. It is a supplement, not a
    replacement: what it finds is merged with the sitemap's list and shaped and
    scored alongside it, so a crawled URL has to look like a tour for the same
    reasons a listed one does.

    Returns the URLs found, home page included, in the order they were reached.
    """
    root = profile.domain if profile.domain.startswith("http") else f"https://{profile.domain}"
    host = urlparse(root).netloc

    visited: set[str] = set()
    queue: list[str] = [root]
    found: list[str] = []
    opened = 0

    while queue and opened < budget:
        page_url = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        try:
            resp = await client.get(page_url)
        except Exception:
            continue
        opened += 1
        await asyncio.sleep(delay)

        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
            continue

        for match in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', resp.text, re.I):
            href = match.group(1).strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute = urljoin(page_url, href).split("#")[0]
            if not _same_host(absolute, host):
                continue
            if _tourish(absolute):
                if absolute not in found:
                    found.append(absolute)
            # A page is worth walking whether or not it is a tour: the links to
            # the tours are on the category pages.
            if absolute not in visited and len(queue) < budget * 2:
                queue.append(absolute)

    return found


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def discover_tour_urls(
    profile: SiteProfile,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    crawl_budget: int = CRAWL_BUDGET_PAGES,
) -> UrlDiscovery:
    """Find tour URLs from a profiled site, keeping only shapes that sell one.

    A thin wrapper over :func:`_discover` so the probe runs once for every
    path through it. The body below has four returns — API only, crawl only,
    nothing new, merged — and repeating the probe at each is how one of them
    quietly keeps the old behaviour forever.

    See :mod:`crawler.filtering.shape_probe` for why the shapes have to be
    asked about rather than read off the URL.
    """
    # Imported here, not at module scope: shape_probe reads group_by_shape
    # from this module, and importing it eagerly closes the loop.
    from crawler.filtering.shape_probe import keep_proven_shapes

    found = await _discover(profile, timeout, crawl_budget)
    return await keep_proven_shapes(profile, found)


async def _discover(
    profile: SiteProfile,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    crawl_budget: int = CRAWL_BUDGET_PAGES,
) -> UrlDiscovery:
    """Find tour URLs from a profiled site.

    Three sources, cheapest first: the site's JSON API, its sitemap, and a
    bounded walk of its own links. The walk runs *in addition* to the sitemap
    rather than only when the sitemap fails, because a sitemap can be present,
    readable, and still wrong — bisanseir.com's lists under half the tours it
    sells, and a run that trusted it silently lost the rest.

    Crawled URLs are not trusted on their own. They join the sitemap's list and
    the whole set is shaped and scored together, so a crawled URL is kept for
    the same reasons a listed one is.

    Pass ``crawl_budget=0`` to skip the walk, which is what a caller wants when
    it only needs a quick answer and can accept the sitemap's word.
    """
    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        result: Optional[UrlDiscovery] = None

        # API path
        if profile.tech.has_json_api:
            found = await _discover_from_api(client, profile)
            if found.total_kept > 0:
                result = found

        # Sitemap path
        if result is None and profile.sitemap.exists:
            found = await _discover_from_sitemap(client, profile)
            if found.total_kept > 0:
                result = found

        if crawl_budget <= 0:
            return result or UrlDiscovery(domain=profile.domain, source=SOURCE_NONE)

        delay, _ = _resolve_crawl_delay(profile)
        crawled = await _discover_from_crawl(client, profile, crawl_budget, delay)

    if result is None:
        # Nothing listed anything; the walk is all there is.
        return _from_urls(profile, crawled, SOURCE_CRAWL) if crawled else UrlDiscovery(
            domain=profile.domain, source=SOURCE_NONE
        )

    added = [url for url in crawled if url not in set(result.tour_urls)]
    if not added:
        result.notes.append(f"walked {crawl_budget} pages and found nothing new")
        return result

    # Re-shape and re-score the two lists together rather than appending: a URL
    # the sitemap never mentioned still has to earn its place.
    everything = sorted(set(result.tour_urls) | set(added))
    merged = _from_urls(profile, everything, result.source)
    merged.last_modified = result.last_modified
    merged.total_seen = result.total_seen + len(added)
    merged.notes = result.notes + [
        f"walking the site added {merged.total_kept - result.total_kept} tour URLs "
        f"the {result.source} never listed"
    ]
    merged.tour_urls = _newest_first(merged.tour_urls, merged.last_modified)
    return merged


def _from_urls(profile: SiteProfile, urls: list[str], source: str) -> UrlDiscovery:
    """Shape, score and select a set of URLs, whatever produced them.

    Shared by the sitemap and the walk so both are judged by one rule. Without
    this the walk would need its own idea of what a tour URL looks like, and
    two ideas drift apart.
    """
    result = UrlDiscovery(domain=profile.domain, source=source)
    result.total_seen = len(urls)

    groups = [
        UrlGroup(shape=shape, urls=url_list)
        for shape, url_list in group_by_shape(sorted(urls)).items()
    ]
    for group in groups:
        _score_group(group, total_urls=result.total_seen)
    groups.sort(key=lambda g: (g.score, g.count), reverse=True)

    if groups and groups[0].score >= MIN_GROUP_SCORE:
        # Every shape that scores level with the best one, not only the best.
        # A site can publish tours under two shapes and mean both: tahagasht
        # lists a tour family at /tours/<slug> and each bookable departure at
        # /tours/item/<slug>, and the departures — 325 pages, all of them real
        # offers — were being dropped.
        #
        # Level with, not merely above the threshold. Keeping everything above
        # 0.5 would also have taken tahagasht's 1173 flight pages and 358 blog
        # posts, which sit exactly on the minimum. The winner's own score is the
        # yardstick, so the bar rises with the confidence of the best answer.
        best = groups[0].score
        kept: list[str] = []
        for group in groups:
            if group.score >= max(MIN_GROUP_SCORE, best - _SHAPE_MARGIN):
                group.is_selected = True
                kept.extend(group.urls)
        result.tour_urls = sorted(set(kept))
        result.total_kept = len(result.tour_urls)
        chosen = [g.shape for g in groups if g.is_selected]
        if len(chosen) > 1:
            result.notes.append(f"kept {len(chosen)} shapes: {', '.join(chosen[:4])}")
    else:
        best = groups[0].score if groups else 0.0
        result.notes.append(f"no group scored above {MIN_GROUP_SCORE} (best {best:.2f})")

    result.groups = groups
    result.rejected_samples = [g.urls[0] for g in groups if not g.is_selected][:5]
    return result


def discover_tour_urls_sync(profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> UrlDiscovery:
    """Synchronous wrapper."""
    return asyncio.run(discover_tour_urls(profile, timeout=timeout))
