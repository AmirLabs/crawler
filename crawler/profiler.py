"""Site Profiler — reconnaissance for any domain.

Given a domain, this module inspects it and produces a `SiteProfile`:
  - is it reachable?
  - what does robots.txt allow, and does it list sitemaps?
  - is there a sitemap, and how many URLs does it advertise?
  - is the site static (HTML has content) or dynamic (needs JavaScript)?
  - are there anti-bot systems (Cloudflare, CAPTCHA, blocking)?
  - which crawl strategy should we use?

It is fully domain-agnostic: nothing here is hardcoded to one website.
"""

from __future__ import annotations

import re
import asyncio
from urllib.parse import urljoin, urlparse

import httpx
from protego import Protego
from selectolax.parser import HTMLParser

from .models import (
    SiteProfile,
    RobotsInfo,
    SitemapInfo,
    RenderingInfo,
    AntiBotInfo,
    TechInfo,
)

# A clear, honest User-Agent. Being identifiable is good crawler etiquette.
USER_AGENT = "TourCrawler/0.1 (+profiler; contact: you@example.com)"

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Hints in raw HTML that reveal a JavaScript framework (⇒ likely dynamic site).
JS_FRAMEWORK_SIGNATURES = {
    "next.js": ["__NEXT_DATA__", "/_next/"],
    "nuxt.js": ["__NUXT__", "id=\"__nuxt\"", "/_nuxt/"],
    "react": ["data-reactroot", "react-dom", "id=\"root\""],
    "vue": ["data-v-", "__vue__", "id=\"app\""],
    "angular": ["ng-version", "ng-app", "_nghost"],
    "svelte": ["__svelte", "svelte-"],
}

# Signals that we hit an anti-bot wall.
CLOUDFLARE_SIGNS = ["just a moment", "cf-chl", "cf-ray", "checking your browser", "cloudflare"]
CAPTCHA_SIGNS = ["captcha", "recaptcha", "hcaptcha", "g-recaptcha"]


def _normalize_domain(domain: str) -> str:
    """Make sure the domain has a scheme (http/https)."""
    domain = domain.strip()
    if not domain.startswith(("http://", "https://")):
        domain = "https://" + domain
    return domain


def _base_url(url: str) -> str:
    """Return scheme://host for a URL, e.g. https://example.com."""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _count_links(html: str, base: str) -> int:
    """Count same-site <a href> links in an HTML string."""
    tree = HTMLParser(html)
    host = urlparse(base).netloc
    count = 0
    for a in tree.css("a[href]"):
        href = a.attributes.get("href") or ""
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base, href)
        if urlparse(absolute).netloc == host:
            count += 1
    return count


def _detect_js_framework(html: str) -> str | None:
    low = html.lower()
    for name, signatures in JS_FRAMEWORK_SIGNATURES.items():
        for sig in signatures:
            if sig.lower() in low:
                return name
    return None


# --------------------------------------------------------------------------- #
# Individual probes
# --------------------------------------------------------------------------- #

async def _probe_homepage(client: httpx.AsyncClient, domain: str, profile: SiteProfile) -> str | None:
    """Fetch the homepage. Fills reachability, status, tech, anti-bot. Returns raw HTML."""
    try:
        resp = await client.get(domain)
    except Exception as exc:  # network error, DNS fail, timeout, etc.
        profile.reachable = False
        profile.error = f"{type(exc).__name__}: {exc}"
        return None

    profile.reachable = True
    profile.status_code = resp.status_code
    profile.final_url = str(resp.url)

    # Tech fingerprint from headers.
    profile.tech.server = resp.headers.get("server")
    profile.tech.powered_by = resp.headers.get("x-powered-by")

    html = resp.text or ""

    # Anti-bot signals.
    low = html.lower()
    if resp.status_code in (403, 429):
        profile.anti_bot.blocks_bots = True
        profile.anti_bot.notes.append(f"status {resp.status_code} on homepage")
    if "cf-ray" in {k.lower() for k in resp.headers} or any(s in low for s in CLOUDFLARE_SIGNS):
        profile.anti_bot.cloudflare = True
    if any(s in low for s in CAPTCHA_SIGNS):
        profile.anti_bot.captcha_detected = True

    # CMS hint from the generator meta tag.
    m = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', html, re.I)
    if m:
        profile.tech.cms = m.group(1)
    elif "wp-content" in low:
        profile.tech.cms = "WordPress"

    return html


async def _probe_robots(client: httpx.AsyncClient, base: str, profile: SiteProfile) -> None:
    """Fetch and parse robots.txt."""
    robots_url = urljoin(base, "/robots.txt")
    try:
        resp = await client.get(robots_url)
    except Exception:
        profile.robots.exists = False
        return

    if resp.status_code != 200 or not resp.text.strip():
        profile.robots.exists = False
        return

    profile.robots.exists = True
    profile.robots.fetched_url = robots_url

    rp = Protego.parse(resp.text)
    profile.robots.allows_crawling = rp.can_fetch(base + "/", USER_AGENT)
    try:
        delay = rp.crawl_delay(USER_AGENT)
        profile.robots.crawl_delay = float(delay) if delay is not None else None
    except Exception:
        profile.robots.crawl_delay = None
    profile.robots.sitemaps = list(rp.sitemaps)


async def _probe_sitemap(client: httpx.AsyncClient, base: str, profile: SiteProfile) -> None:
    """Find and parse a sitemap. Prefer robots-declared sitemaps, else /sitemap.xml."""
    candidates = list(profile.robots.sitemaps) or [urljoin(base, "/sitemap.xml")]

    for sm_url in candidates:
        try:
            resp = await client.get(sm_url)
        except Exception:
            continue
        if resp.status_code != 200 or "<" not in resp.text:
            continue

        text = resp.text
        profile.sitemap.exists = True
        profile.sitemap.location = sm_url

        # <sitemapindex> = nested (points to more sitemaps); <urlset> = flat page list.
        is_index = "<sitemapindex" in text.lower()
        profile.sitemap.type = "index" if is_index else "urlset"

        locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", text, re.I | re.S)
        profile.sitemap.url_count = len(locs)
        profile.sitemap.sample_urls = [l.strip() for l in locs[:5]]
        return  # first working sitemap wins


async def _probe_rendering(base: str, raw_html: str, profile: SiteProfile, render: bool) -> None:
    """Decide static vs dynamic by comparing raw-HTML links to rendered links."""
    raw_links = _count_links(raw_html, base)
    profile.rendering.links_in_raw_html = raw_links
    profile.rendering.js_framework_hint = _detect_js_framework(raw_html)

    if not render:
        # Heuristic without a browser: framework + very few raw links ⇒ probably dynamic.
        if profile.rendering.js_framework_hint and raw_links < 5:
            profile.rendering.type = "dynamic"
            profile.rendering.needs_playwright = True
        elif raw_links > 0:
            profile.rendering.type = "static"
        return

    # Render with a real browser and re-count links.
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=USER_AGENT)
            await page.goto(base, wait_until="networkidle", timeout=20000)
            rendered_html = await page.content()
            await browser.close()

        rendered_links = _count_links(rendered_html, base)
        profile.rendering.links_after_render = rendered_links

        # If rendering reveals many more links, JavaScript is building the page.
        if rendered_links > max(raw_links * 2, raw_links + 10):
            profile.rendering.type = "dynamic"
            profile.rendering.needs_playwright = True
        else:
            profile.rendering.type = "static"
    except Exception as exc:
        profile.rendering.notes = f"render failed: {exc}"
        # Fall back to the no-browser heuristic.
        if profile.rendering.js_framework_hint and raw_links < 5:
            profile.rendering.type = "dynamic"
            profile.rendering.needs_playwright = True
        elif raw_links > 0:
            profile.rendering.type = "static"


def _decide_strategy(profile: SiteProfile) -> str:
    """Pick the best crawl strategy from everything we learned."""
    if not profile.reachable:
        return "unreachable"
    if profile.anti_bot.blocks_bots or profile.anti_bot.captcha_detected:
        return "blocked"
    if not profile.robots.allows_crawling:
        return "blocked"
    if profile.sitemap.exists and profile.sitemap.url_count > 0:
        return "sitemap"
    if profile.rendering.needs_playwright:
        return "dynamic-crawl"
    return "static-crawl"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def profile_site(domain: str, render: bool = True, timeout: float = 15.0) -> SiteProfile:
    """Inspect a single domain and return a SiteProfile."""
    domain = _normalize_domain(domain)
    base = _base_url(domain)
    profile = SiteProfile(domain=domain)

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        raw_html = await _probe_homepage(client, domain, profile)
        if not profile.reachable:
            profile.recommended_strategy = _decide_strategy(profile)
            return profile

        # robots first (it may reveal sitemaps), then sitemap.
        await _probe_robots(client, base, profile)
        await _probe_sitemap(client, base, profile)

    await _probe_rendering(base, raw_html or "", profile, render=render)
    profile.recommended_strategy = _decide_strategy(profile)
    return profile


def profile_site_sync(domain: str, render: bool = True, timeout: float = 15.0) -> SiteProfile:
    """Synchronous wrapper for convenience."""
    return asyncio.run(profile_site(domain, render=render, timeout=timeout))
