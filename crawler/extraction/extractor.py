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


# What a Next.js App Router page streams its data through, one call per chunk.
# The argument is a JSON string, and the chunks are halves of one document, so
# they are joined before anything is read out of them. Not a rule about any one
# site: this is the framework's own transport.
_NEXT_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")\]\)')

# The smallest object worth pulling out of that stream. Below this it is
# routing and styling, not content.
_MIN_NEXT_OBJECT = 200


def _next_payload_objects(html: str) -> list[dict]:
    """Every JSON object a Next.js page streams into its own markup.

    The framework writes its data as ``self.__next_f.push([1, "..."])``, so it
    reaches the browser inside a script that is code rather than data and the
    plain ``application/json`` reader never sees it. chakavaktravel.com puts
    its whole tour there — dates, prices, hotels and their photographs — which
    is why that site produced nothing at all.

    Chunks are joined first: the framework splits one document across calls,
    and an object can straddle the join.

    Example:
        >>> _next_payload_objects(
        ...     'x<script>self.__next_f.push([1,"a:{\\"tour\\":'
        ...     '{\\"name\\":\\"' + "تور کیش" + '\\",\\"night\\":3}}"])</script>')
        [{'name': 'تور کیش', 'night': 3}]
    """
    joined = ""
    for match in _NEXT_CHUNK_RE.finditer(html):
        try:
            joined += json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    found: list[dict] = []
    for start, char in enumerate(joined):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(joined)):
            if joined[end] == "{":
                depth += 1
            elif joined[end] == "}":
                depth -= 1
                if depth:
                    continue
                if end - start >= _MIN_NEXT_OBJECT:
                    try:
                        data = json.loads(joined[start:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(data, dict):
                        found.append(data)
                break
    return found


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
    return blobs + _next_payload_objects(html)


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


class BrowserSession:
    """One headless browser, held open across many pages.

    Launching Chromium costs a second or two, and the crawl used to pay it per
    URL: a site needing 800 rendered pages launched 800 browsers. One browser
    for a whole site turns that into one launch, and a fresh tab per page keeps
    the pages from sharing cookies or leftover state.

    The launch is lazy. Most sites never need a browser at all, and a session
    that is opened but never used costs nothing.

    Example:
        >>> async with BrowserSession() as session:
        ...     page = await session.fetch("https://example.com/tour/1", 15.0)
        >>> page.fetch_method
        'playwright'
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    async def __aenter__(self) -> "BrowserSession":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Shut the browser down, if one was ever started."""
        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None

    async def _browser_or_launch(self):
        """The open browser, launching one on first use.

        Tries Playwright's own Chromium first, then an installed Chrome or
        Edge, so a machine where ``playwright install`` was never run still
        renders — which is the case on this one.
        """
        if self._browser is not None:
            return self._browser

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        for channel in (None, "chrome", "msedge"):
            try:
                kwargs = {"headless": True}
                if channel:
                    kwargs["channel"] = channel
                self._browser = await self._playwright.chromium.launch(**kwargs)
                return self._browser
            except Exception:
                continue

        raise RuntimeError(
            "no usable Chromium: run `playwright install chromium` "
            "or install Google Chrome"
        )

    async def fetch(self, url: str, timeout: float) -> ExtractedPage:
        """Render one URL in a fresh tab of this session's browser."""
        page = ExtractedPage(url=url, fetch_method=FETCH_PLAYWRIGHT)
        browser_page = None
        try:
            browser = await self._browser_or_launch()
            browser_page = await browser.new_page(user_agent=USER_AGENT)
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

            page.status_code = response.status if response else None
            page.fetched_at = _now_iso()
            page.html_raw = html
            page.embedded_json = _extract_embedded_json(html)
            page.content_hash = _content_hash(html)
        except Exception as exc:
            page.error = f"render failed: {exc}"
        finally:
            if browser_page is not None:
                try:
                    await browser_page.close()
                except Exception:
                    pass
        return page


async def _fetch_playwright(
    url: str, timeout: float, browser: Optional[BrowserSession] = None
) -> ExtractedPage:
    """Render one URL, reusing ``browser`` when the caller has one open.

    Without a session this opens and closes its own, which is right for a
    single page and wasteful for many — :func:`fetch_pages` passes one in.

    Example:
        >>> page = await _fetch_playwright("https://example.com/tour/1", timeout=15.0)
        >>> page.fetch_method
        'playwright'
    """
    if browser is not None:
        return await browser.fetch(url, timeout)
    async with BrowserSession() as session:
        return await session.fetch(url, timeout)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def fetch_page(
    url: str,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    browser: Optional[BrowserSession] = None,
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
        return await _fetch_playwright(url, timeout=timeout, browser=browser)

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
        # instead of a browser launch and a wait.
        if find_api_base(page.html_raw):
            page.notes.append("page needs rendering but the site publishes an API")
            return page

        # Whether rendering is worth its 6-13 seconds is a fact about the site,
        # and it is measured once rather than assumed — see
        # ``RenderingInfo.render_helps``. Until that measurement exists the
        # page is rendered, so the first page of a site is never the one that
        # silently loses data.
        if profile.rendering.render_helps is False:
            page.notes.append("this site was measured: rendering adds nothing")
            return page

        return await render_beside(
            page,
            "served markup still held template placeholders; rendered instead",
            timeout=timeout,
            browser=browser,
        )

    return page


async def render_beside(
    page: ExtractedPage,
    reason: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    browser: Optional[BrowserSession] = None,
) -> ExtractedPage:
    """Fetch ``page``'s URL again through a browser, keeping the plain markup.

    The plain copy is carried on ``html_before_render`` rather than thrown
    away, because whether the browser was worth its 6 to 13 seconds can only
    be answered by counting the rows each version yields — which is
    :func:`crawler.extraction.page_extract._settle_render_question`'s job.

    A failed render leaves ``page`` as it was: a browser that will not start is
    a reason to carry on with plain markup, not to lose the page.

    Example:
        >>> page = ExtractedPage(url="https://example.com/tour/1", html_raw="<body></body>")
        >>> out = await render_beside(page, "nothing came out of the plain markup")
        >>> out.html_before_render
        '<body></body>'
    """
    rendered = await _fetch_playwright(page.url, timeout=timeout, browser=browser)
    if not rendered.html_raw:
        page.notes.append(f"needed rendering but it failed: {rendered.error}")
        return page

    rendered.notes = page.notes + [reason] + rendered.notes
    rendered.html_before_render = page.html_raw
    return rendered


async def fetch_pages(
    urls: list[str], profile: SiteProfile, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> list[ExtractedPage]:
    """Fetch many tour URLs from the same site, reusing one HTTP client.

    Pages are fetched one at a time rather than concurrently: the crawl delay
    is meaningless if several requests are in flight at once, so sequential
    fetching is what actually holds the rate down.

    The wait goes *before* each request except the first — waiting after the
    last one delays the caller without protecting the host.

    Both the HTTP client and the browser are opened once for the whole list.
    The browser is the expensive one: launching Chromium takes a second or two,
    and this used to pay it per URL. It is also lazy, so a site that needs no
    rendering never starts one.

    Every URL goes through :func:`fetch_page`, so the choice between HTTP, the
    site's API and the browser is made in one place rather than twice.

    Example:
        >>> profile = SiteProfile(domain="example.com")
        >>> pages = await fetch_pages(["https://example.com/tour/1"], profile)
        >>> len(pages)
        1
    """
    results: list[ExtractedPage] = []
    delay, delay_note = _resolve_crawl_delay(profile)

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout
    ) as client:
        async with BrowserSession() as browser:
            for index, url in enumerate(urls):
                if index:
                    await _wait_between_requests(delay)
                page = await fetch_page(
                    url, profile, client=client, timeout=timeout, browser=browser
                )
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
