"""Step 3: fetch a tour page and pull out its raw material.

This module answers one question: for a single URL, what is the raw HTML and
what structured JSON did the site itself embed in it? It does not decide what
any of that means yet — matching page data to schema fields is a later layer.
Keeping fetch separate from mapping means a slow network call is never repeated
just to try a new mapping rule.

Fetch strategy follows the site profile, not a guess per URL: a page on a site
the profiler marked as needing JavaScript is rendered with Playwright, every
other page is a plain HTTP GET.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from crawler.config import (
    CRAWL_DELAY_JITTER,
    DEFAULT_CRAWL_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_CRAWL_DELAY_SECONDS,
    MIN_CRAWL_DELAY_SECONDS,
    RENDER_SETTLE_SECONDS,
    USER_AGENT,
)
from crawler.extraction.rendering import needs_rendering
from crawler.extraction.tour_api import find_api_base
from crawler.profiler.models import SiteProfile
from crawler.utils.http import get_with_retry
from crawler.extraction.models import (
    ExtractedPage,
    FETCH_HTTP,
    FETCH_PLAYWRIGHT,
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
}

# Matches any <script> whose type marks it as data rather than executable code:
# application/json (Next.js' __NEXT_DATA__ and most custom blobs) and
# application/ld+json (schema.org markup, common for product/offer data).
_JSON_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\'](?:application/(?:ld\+)?json)["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)


def _extract_embedded_json(html: str) -> list[dict]:
    """Pull every JSON object a page embeds in a ``<script>`` tag.

    Blobs that fail to parse, or that decode to something other than a dict
    (e.g. a bare list or number), are skipped rather than raised on — one
    malformed tracking-pixel script should not stop extraction of the tour
    data sitting in the next one.

    Example:
        >>> html = '<script type="application/json">{"title": "تور کوالالامپور"}</script>'
        >>> _extract_embedded_json(html)
        [{'title': 'تور کوالالامپور'}]
    """
    blobs: list[dict] = []
    for match in _JSON_SCRIPT_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            blobs.append(data)
    return blobs


def _resolve_crawl_delay(profile: SiteProfile) -> tuple[float, Optional[str]]:
    """Decide how long to wait between requests to this site.

    What the site declared in ``robots.txt`` wins, clamped into the configured
    range. A site that declared nothing gets the default rather than no wait at
    all: "no rule" means we know nothing about what the host tolerates, not
    that it tolerates everything. When clamping changes the declared value, the
    reason is returned so it ends up on the page record instead of being
    silently applied.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> profile.robots.crawl_delay = None
        >>> _resolve_crawl_delay(profile)
        (1.0, 'site declared no crawl-delay; using default 1.0s')
        >>> profile.robots.crawl_delay = 30.0
        >>> _resolve_crawl_delay(profile)
        (10.0, 'site asked for 30.0s; capped at 10.0s')
    """
    declared = profile.robots.crawl_delay

    if declared is None:
        return (
            DEFAULT_CRAWL_DELAY_SECONDS,
            f"site declared no crawl-delay; using default {DEFAULT_CRAWL_DELAY_SECONDS}s",
        )

    if declared > MAX_CRAWL_DELAY_SECONDS:
        return (
            MAX_CRAWL_DELAY_SECONDS,
            f"site asked for {declared}s; capped at {MAX_CRAWL_DELAY_SECONDS}s",
        )

    if declared < MIN_CRAWL_DELAY_SECONDS:
        return (
            MIN_CRAWL_DELAY_SECONDS,
            f"site asked for {declared}s; raised to floor {MIN_CRAWL_DELAY_SECONDS}s",
        )

    return declared, None


async def _wait_between_requests(delay: float) -> None:
    """Sleep for ``delay`` seconds, scaled by a random jitter factor.

    The average rate stays at ``delay``, but the exact interval varies, so the
    request pattern does not look like a metronome to a WAF.

    Example:
        >>> await _wait_between_requests(1.0)  # sleeps somewhere in 0.5s..1.5s
    """
    if delay <= 0:
        return
    low, high = CRAWL_DELAY_JITTER
    await asyncio.sleep(delay * random.uniform(low, high))


def _content_hash(html: str) -> str:
    """Hash a page body so a daily refresh can skip pages that did not change.

    Example:
        >>> _content_hash("<html>hi</html>")
        '3a5d5c3417c8e0b1...'
    """
    return hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string.

    Example:
        >>> _now_iso()
        '2026-08-08T12:00:00+00:00'
    """
    return datetime.now(timezone.utc).isoformat()


async def _fetch_http(client: httpx.AsyncClient, url: str) -> ExtractedPage:
    """Fetch one URL with a plain GET request.

    Example:
        >>> async with httpx.AsyncClient() as c:
        ...     page = await _fetch_http(c, "https://example.com/tour/1")
        >>> page.fetch_method
        'http'
    """
    page = ExtractedPage(url=url, fetch_method=FETCH_HTTP)
    resp, error = await get_with_retry(client, url)
    if resp is None:
        page.error = f"request failed: {error}"
        return page

    page.status_code = resp.status_code
    page.fetched_at = _now_iso()

    if resp.status_code != 200:
        page.error = f"unexpected status {resp.status_code}"
        return page

    html = resp.text
    page.html_raw = html
    page.embedded_json = _extract_embedded_json(html)
    page.content_hash = _content_hash(html)
    return page


async def _fetch_playwright(url: str, timeout: float) -> ExtractedPage:
    """Fetch one URL with a headless browser, for pages that need JS to render.

    Mirrors the Chromium fallback chain in
    :func:`crawler.profiler.profiler._probe_rendering`: Playwright's own
    Chromium first, then an installed Chrome or Edge, so a missing
    ``playwright install`` does not stop extraction on a machine that has a
    system browser.

    Example:
        >>> page = await _fetch_playwright("https://example.com/tour/1", timeout=15.0)
        >>> page.fetch_method
        'playwright'
    """
    page = ExtractedPage(url=url, fetch_method=FETCH_PLAYWRIGHT)
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = None
            for channel in (None, "chrome", "msedge"):
                try:
                    kwargs = {"headless": True}
                    if channel:
                        kwargs["channel"] = channel
                    browser = await pw.chromium.launch(**kwargs)
                    break
                except Exception:
                    continue
            if browser is None:
                raise RuntimeError(
                    "no usable Chromium: run `playwright install chromium` "
                    "or install Google Chrome"
                )

            browser_page = await browser.new_page(user_agent=USER_AGENT)
            # "networkidle" never arrives on a page with analytics or a live
            # socket — it timed out on every alefbatour tour page. Wait for the
            # content instead: the template placeholders disappearing is what
            # "finished" actually means here.
            response = await browser_page.goto(
                url, wait_until="domcontentloaded", timeout=timeout * 1000
            )
            try:
                await browser_page.wait_for_load_state(
                    "networkidle", timeout=RENDER_SETTLE_SECONDS * 1000
                )
            except Exception:
                # Whatever rendered by now is still better than the raw
                # template, so this is worth a note rather than a failure.
                page.notes.append(
                    f"network had not settled after {RENDER_SETTLE_SECONDS}s"
                )
            html = await browser_page.content()
            await browser.close()

        page.status_code = response.status if response else None
        page.fetched_at = _now_iso()
        page.html_raw = html
        page.embedded_json = _extract_embedded_json(html)
        page.content_hash = _content_hash(html)
    except Exception as exc:
        page.error = f"render failed: {exc}"
    return page


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def fetch_page(
    url: str,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExtractedPage:
    """Fetch one tour URL, using a browser when the markup turns out to need one.

    The profiler's ``needs_playwright`` decides the first attempt. It is a
    judgement about the *site*, made from its home page, and that is not always
    the right answer for a tour page: alefbatour.com serves a home page built on
    the server and tour pages whose hotel list is left as ``{{ }}`` for
    JavaScript to fill. So a cheap HTTP fetch that comes back holding template
    placeholders is fetched again with the browser.

    The second fetch costs a browser launch, so it happens only when the markup
    says it is needed, and never twice for one URL.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> page = await fetch_page("https://example.com/tour/1", profile)
        >>> page.fetch_method
        'http'
    """
    if profile.rendering.needs_playwright:
        return await _fetch_playwright(url, timeout=timeout)

    if client is not None:
        page = await _fetch_http(client, url)
    else:
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout
        ) as owned_client:
            page = await _fetch_http(owned_client, url)

    if page.html_raw and needs_rendering(page.html_raw):
        # A site that publishes its own API needs no browser: the JSON carries
        # what the template would have been filled with, for one small request
        # instead of a browser launch and an eight second wait.
        if find_api_base(page.html_raw):
            page.notes.append("page needs rendering but the site publishes an API")
            return page

        rendered = await _fetch_playwright(url, timeout=timeout)
        if rendered.html_raw:
            rendered.notes = page.notes + [
                "served markup still held template placeholders; rendered instead"
            ] + rendered.notes
            return rendered
        page.notes.append(f"needed rendering but it failed: {rendered.error}")

    return page


async def fetch_pages(
    urls: list[str], profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> list[ExtractedPage]:
    """Fetch many tour URLs from the same site, reusing one HTTP client.

    Pages are fetched one at a time rather than concurrently: the crawl delay
    is meaningless if several requests are in flight at once, so sequential
    fetching is what actually holds the rate down.

    The wait goes *before* each request except the first — waiting after the
    last one delays the caller without protecting the host.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> pages = await fetch_pages(["https://example.com/tour/1"], profile)
        >>> len(pages)
        1
    """
    results: list[ExtractedPage] = []
    delay, delay_note = _resolve_crawl_delay(profile)

    if profile.rendering.needs_playwright:
        for index, url in enumerate(urls):
            if index:
                await _wait_between_requests(delay)
            page = await _fetch_playwright(url, timeout=timeout)
            if delay_note:
                page.notes.append(delay_note)
            results.append(page)
        return results

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout
    ) as client:
        for index, url in enumerate(urls):
            if index:
                await _wait_between_requests(delay)
            page = await _fetch_http(client, url)
            if delay_note:
                page.notes.append(delay_note)
            results.append(page)

    return results


def fetch_pages_sync(
    urls: list[str], profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> list[ExtractedPage]:
    """Synchronous wrapper around :func:`fetch_pages`.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> pages = fetch_pages_sync(["https://example.com/tour/1"], profile)
        >>> len(pages)
        1
    """
    return asyncio.run(fetch_pages(urls, profile, timeout=timeout))
