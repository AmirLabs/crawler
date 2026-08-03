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

from ..config import DEFAULT_TIMEOUT_SECONDS, USER_AGENT
from .models import (
    SiteProfile,
    RobotsInfo,
    SitemapInfo,
    RenderingInfo,
    AntiBotInfo,
    TechInfo,
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    # Persian first (target sites are Persian), English as fallback.
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.6",
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

# Markers that the site sits behind Cloudflare. Being behind Cloudflare is
# normal and does NOT mean we are blocked, so this stays informational.
CLOUDFLARE_SIGNS = ["cf-ray", "cloudflare"]

# Markup that loads a real CAPTCHA widget — not merely the word "captcha",
# which shows up harmlessly in CSS rules and embedded JSON on normal pages.
CAPTCHA_WIDGET_SIGNS = [
    "recaptcha/api.js",
    "hcaptcha.com/1/api.js",
    "challenges.cloudflare.com/turnstile",
    "arcaptcha.co",
    'class="g-recaptcha"',
    "class='g-recaptcha'",
]

# Wording that only appears on an interstitial "prove you are human" page.
CHALLENGE_PAGE_SIGNS = [
    "just a moment",
    "cf-chl",
    "checking your browser",
    "verify you are human",
    "enable javascript and cookies to continue",
    "attention required",
    "access denied",
    "ddos protection by",
]

# A challenge page is small and nearly linkless; a real page is neither.
WALL_MAX_HTML_CHARS = 50_000
WALL_MAX_LINKS = 10

# When a sitemap is an index, how many nested sitemaps to actually open.
# Enough to estimate the real page count without downloading the whole site map.
MAX_CHILD_SITEMAPS_TO_READ = 5

# Words in a WordPress post-type slug that suggest it holds tour listings.
# Checked against the types the site reports, so custom names are found too.
TOUR_POST_TYPE_WORDS = ("tour", "package", "product", "travel", "trip")


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

    html = resp.text or ""

    # Anti-bot signals.
    low = html.lower()

    if resp.status_code in (403, 429):
        profile.anti_bot.blocks_bots = True
        profile.anti_bot.notes.append(f"status {resp.status_code} on homepage")

    if "cf-ray" in {k.lower() for k in resp.headers} or any(s in low for s in CLOUDFLARE_SIGNS):
        profile.anti_bot.cloudflare = True

    # Only treat a CAPTCHA as blocking when the page actually looks like a wall:
    # small, almost no links, and either challenge wording or a widget script.
    # Otherwise it is a normal page that merely mentions/preloads a CAPTCHA.
    final_base = _base_url(str(resp.url))
    looks_like_wall = (
        len(html) < WALL_MAX_HTML_CHARS and _count_links(html, final_base) <= WALL_MAX_LINKS
    )
    challenge_wording = [s for s in CHALLENGE_PAGE_SIGNS if s in low]
    widgets = [s for s in CAPTCHA_WIDGET_SIGNS if s in low]

    if looks_like_wall and (challenge_wording or widgets):
        profile.anti_bot.captcha_detected = True
        profile.anti_bot.blocks_bots = True
        reason = challenge_wording[0] if challenge_wording else widgets[0]
        profile.anti_bot.notes.append(f"challenge page detected ({reason})")
    elif widgets:
        # Widget present on a full page: usually a login/booking form. Note it,
        # but do not call the site blocked.
        profile.anti_bot.notes.append(f"captcha widget on page, content still readable ({widgets[0]})")

    # CMS hint from the generator meta tag.
    m = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', html, re.I)
    if m:
        profile.tech.cms = m.group(1)
    elif "wp-content" in low or "wp-includes" in low:
        # WordPress footprints in script/style paths. More reliable than
        # wp-content alone, which a single linked image would also trigger.
        profile.tech.cms = "WordPress"

    return html


def _find_wp_api_root(html: str, base: str) -> str:
    """Locate the WordPress REST root.

    WordPress advertises it via ``<link rel="https://api.w.org/" href="...">``.
    Trusting that tag handles installs living in a subdirectory, which a plain
    ``base + /wp-json/`` guess would miss. The href is often site-relative, so
    it is resolved against ``base`` before being returned.
    """
    m = re.search(
        r'<link[^>]+rel=["\']https://api\.w\.org/["\'][^>]+href=["\']([^"\']+)', html, re.I
    )
    if not m:
        # Same tag with the attributes in the opposite order.
        m = re.search(
            r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']https://api\.w\.org/["\']', html, re.I
        )
    if m:
        return urljoin(base, m.group(1).strip())
    return urljoin(base, "/wp-json/")


async def _probe_json_api(
    client: httpx.AsyncClient, base: str, html: str, profile: SiteProfile
) -> None:
    """Check whether a JSON API exists that we could read tour data from.

    A JSON endpoint is far cheaper than HTML plus an LLM call, and its
    ``modified`` timestamps let a daily refresh fetch only what changed. Only
    WordPress REST is checked for now, as it is the most common CMS on Persian
    tour sites.

    An endpoint answering with a JSON list counts as working, but one that also
    carries records is preferred: an empty list means "valid API, nothing of
    this type", which is no use for extraction.
    """
    if not profile.tech.cms or "wordpress" not in profile.tech.cms.lower():
        return

    api_root = _find_wp_api_root(html, base).rstrip("/") + "/"

    # Ask the site which post types it actually publishes instead of guessing.
    # Tour sites commonly register a custom type ("tour", "package", "tours")
    # whose name we would never hit by guessing alone.
    candidates: list[str] = []
    try:
        resp = await client.get(urljoin(api_root, "wp/v2/types"))
        if resp.status_code == 200:
            types = resp.json()
            if isinstance(types, dict):
                for slug, info in types.items():
                    if slug.startswith("wp_") or slug in ("attachment", "nav_menu_item"):
                        continue  # WordPress internals, never tour content.
                    rest_base = (info or {}).get("rest_base") or slug
                    if any(w in slug.lower() for w in TOUR_POST_TYPE_WORDS):
                        candidates.insert(0, f"wp/v2/{rest_base}")  # tour-ish first
                    elif slug in ("post", "page"):
                        candidates.append(f"wp/v2/{rest_base}")
    except Exception:
        pass

    if not candidates:
        candidates = ["wp/v2/tour", "wp/v2/package", "wp/v2/product", "wp/v2/posts"]

    empty_but_working: str | None = None

    for path in candidates:
        url = urljoin(api_root, path)
        try:
            # per_page=1 deliberately: some sites 500 on the default page size
            # of 100, and one record is all that is needed to confirm the door.
            resp = await client.get(url, params={"per_page": 1})
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        if data:
            profile.tech.has_json_api = True
            profile.tech.json_api_url = url
            return  # Endpoint with real records: best case, stop here.
        if empty_but_working is None:
            empty_but_working = url

    if empty_but_working:
        # Valid API, but this post type is empty. Worth recording so a later
        # step can try other post types instead of assuming HTML is the only way.
        profile.tech.has_json_api = True
        profile.tech.json_api_url = empty_but_working


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


def _extract_locs(xml: str) -> list[str]:
    """Pull every <loc> value out of a sitemap document."""
    return [m.strip() for m in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I | re.S)]


async def _probe_sitemap(client: httpx.AsyncClient, base: str, profile: SiteProfile) -> None:
    """Find and parse a sitemap. Prefer robots-declared sitemaps, else /sitemap.xml.

    A sitemap can be a flat ``<urlset>`` (its <loc> entries are pages) or a
    ``<sitemapindex>`` (its <loc> entries are *other sitemaps*). For an index we
    open a few children and count their pages, otherwise ``url_count`` would
    report the number of sitemaps and make a large site look tiny.
    """
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

        is_index = "<sitemapindex" in text.lower()
        profile.sitemap.type = "index" if is_index else "urlset"
        locs = _extract_locs(text)

        if not is_index:
            profile.sitemap.url_count = len(locs)
            profile.sitemap.sample_urls = locs[:5]
            return

        # Index: the locs are child sitemaps, not pages.
        profile.sitemap.child_sitemap_count = len(locs)
        children = locs[:MAX_CHILD_SITEMAPS_TO_READ]
        profile.sitemap.url_count_is_partial = len(locs) > len(children)

        page_count = 0
        samples: list[str] = []
        for child_url in children:
            try:
                child_resp = await client.get(child_url)
            except Exception:
                profile.sitemap.url_count_is_partial = True
                continue
            if child_resp.status_code != 200 or "<" not in child_resp.text:
                profile.sitemap.url_count_is_partial = True
                continue
            # A child can itself be an index; count only real page URLs.
            if "<sitemapindex" in child_resp.text.lower():
                nested = _extract_locs(child_resp.text)
                profile.sitemap.child_sitemap_count += len(nested)
                profile.sitemap.url_count_is_partial = True
                continue
            child_locs = _extract_locs(child_resp.text)
            page_count += len(child_locs)
            if len(samples) < 5:
                samples.extend(child_locs[: 5 - len(samples)])

        profile.sitemap.url_count = page_count
        profile.sitemap.sample_urls = samples
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
            browser = None
            # Prefer Playwright's own Chromium; fall back to an installed Chrome
            # or Edge when the browser download was never completed.
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

async def profile_site(
    domain: str, render: bool = True, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> SiteProfile:
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

        # robots first (it may reveal sitemaps), then sitemap, then the JSON API
        # (which needs the CMS hint that _probe_homepage filled in).
        await _probe_robots(client, base, profile)
        await _probe_sitemap(client, base, profile)
        await _probe_json_api(client, base, raw_html or "", profile)

    await _probe_rendering(base, raw_html or "", profile, render=render)
    profile.recommended_strategy = _decide_strategy(profile)
    return profile


def profile_site_sync(
    domain: str, render: bool = True, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> SiteProfile:
    """Synchronous wrapper for convenience."""
    return asyncio.run(profile_site(domain, render=render, timeout=timeout))
