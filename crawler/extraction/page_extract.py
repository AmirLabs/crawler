"""Step 3's entry point: one URL in, one :class:`ExtractedPage` out.

Everything before this module reads one kind of thing — where a page repeats,
what a price says, which date is printed. This is where those parts are put
together, and it is the first place the whole stage runs end to end.

The page is read in two passes, because a tour page states two kinds of fact:

* **Once per page** — the tour's name, its picture, who is selling it. These
  come from the document head and the agency's own markup, which is the one
  part of these sites that follows a shared convention: all seven publish
  Open Graph tags, so ``og:title`` and ``og:image`` work everywhere without a
  per-site rule.
* **Once per bookable option** — handed to
  :func:`crawler.extraction.row_mapping.map_rows`, which reads the repeating
  block :mod:`crawler.extraction.html_rows` picked out.

``origin`` and ``destination`` are deliberately left unresolved for now. They
are readable only from prose on most of the target sites, and a guess would be
worse than an empty field that the fill rate can point at.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from crawler.config import DEFAULT_TIMEOUT_SECONDS
from crawler.extraction.extractor import fetch_page
from crawler.extraction.html_rows import MIN_ROW_SCORE, find_row_containers
from crawler.extraction.json_paths import flatten_json
from crawler.extraction.models import (
    METHOD_EXACT_KEY,
    METHOD_HEURISTIC_HTML,
    ExtractedField,
    ExtractedPage,
    record_attempt,
)
from crawler.extraction.row_mapping import map_rows
from crawler.extraction.target_fields import PAGE_FIELDS, unresolved
from crawler.profiler.models import SiteProfile
from crawler.schema.output_models import normalize_transport

# JSON-LD keys that hold the publisher's logo. Yoast, which several of the
# target sites use, writes the first of these.
_LOGO_PATH_ENDINGS = (".logo.url", ".logo.contentUrl", ".publisher.logo.url")

# Only this much of the page text is scanned for a transport word. The word
# belongs near the tour's own description; further down sit the footer's links
# to every other kind of tour the agency sells, and those would answer for it.
_TRANSPORT_SCAN_CHARS = 3000

# Which page-level fields this module actually looks for, and where it looks.
# ``origin``, ``destination`` and ``is_installment_available`` are missing on
# purpose: no rule reads them yet, and saying so is more useful than implying
# they were searched for and absent.
_PAGE_ATTEMPTS: dict[str, str] = {
    "tour_name": f"{METHOD_HEURISTIC_HTML}:h1/og:title",
    "image_url": f"{METHOD_HEURISTIC_HTML}:og:image",
    "agency_name": f"{METHOD_HEURISTIC_HTML}:og:site_name",
    "website_url": f"{METHOD_EXACT_KEY}:page_host",
    "logo_url": f"{METHOD_EXACT_KEY}:json_ld_logo",
    "transportation_type": f"{METHOD_HEURISTIC_HTML}:page_text",
}


def _meta_content(tree: HTMLParser, *names: str) -> Optional[str]:
    """Read the first Open Graph or ``name=`` meta tag that is present.

    Example:
        >>> tree = HTMLParser('<meta property="og:title" content="تور کوالالامپور">')
        >>> _meta_content(tree, "og:title")
        'تور کوالالامپور'
    """
    for name in names:
        for attribute in ("property", "name"):
            node = tree.css_first(f'meta[{attribute}="{name}"]')
            if node is not None:
                content = (node.attributes.get("content") or "").strip()
                if content:
                    return content
    return None


def _read_tour_name(tree: HTMLParser) -> Optional[str]:
    """Read the tour's name from the page.

    The ``<h1>`` is preferred over ``og:title`` because the Open Graph title
    usually carries the agency's name too — "تور کوالالامپور | نهال گشت" —
    while the heading is the tour alone.

    Example:
        >>> _read_tour_name(HTMLParser('<h1> تور کوالالامپور ۷ شب </h1>'))
        'تور کوالالامپور ۷ شب'
    """
    heading = tree.css_first("h1")
    if heading is not None:
        text = " ".join((heading.text(deep=True) or "").split())
        if text:
            return text

    for value in (_meta_content(tree, "og:title", "twitter:title"), None):
        if value:
            return value

    title = tree.css_first("title")
    if title is not None:
        text = " ".join((title.text(deep=True) or "").split())
        if text:
            return text
    return None


def _read_logo_url(embedded_json: list[dict], base: str) -> Optional[str]:
    """Find the agency's logo in the page's JSON-LD, if it published one.

    Example:
        >>> _read_logo_url([{"publisher": {"logo": {"url": "/img/logo.png"}}}], "https://x.com")
        'https://x.com/img/logo.png'
    """
    for blob in embedded_json:
        for path, values in flatten_json(blob).items():
            if any(path.endswith(ending) for ending in _LOGO_PATH_ENDINGS):
                if values and isinstance(values[0], str):
                    return urljoin(base, values[0])
    return None


def read_page_fields(
    tree: HTMLParser, url: str, embedded_json: Optional[list[dict]] = None
) -> dict[str, ExtractedField]:
    """Read the facts that hold for the whole page.

    Example:
        >>> tree = HTMLParser(
        ...     '<html><head>'
        ...     '<meta property="og:image" content="/img/tour.jpg">'
        ...     '<meta property="og:site_name" content="نهال گشت">'
        ...     '</head><body><h1>تور کوالالامپور</h1><p>پرواز مستقیم</p></body></html>'
        ... )
        >>> fields = read_page_fields(tree, "https://nahalgasht.com/tour/1")
        >>> fields["tour_name"].value, fields["agency_name"].value
        ('تور کوالالامپور', 'نهال گشت')
        >>> fields["image_url"].value
        'https://nahalgasht.com/img/tour.jpg'
        >>> fields["transportation_type"].value
        'air'
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else url
    fields: dict[str, ExtractedField] = {}

    def record(name: str, value: object, method: str, source: str) -> None:
        if value in (None, ""):
            return
        fields[name] = ExtractedField(value=value, method=method, source_key=source)

    record("tour_name", _read_tour_name(tree), METHOD_HEURISTIC_HTML, "h1/og:title")

    image = _meta_content(tree, "og:image", "twitter:image")
    record("image_url", urljoin(base, image) if image else None, METHOD_HEURISTIC_HTML, "og:image")

    record(
        "agency_name",
        _meta_content(tree, "og:site_name", "application-name"),
        METHOD_HEURISTIC_HTML,
        "og:site_name",
    )
    # The agency's site is the page's own host, which is a fact rather than a
    # reading, so it is recorded as an exact match.
    record("website_url", base if parsed.netloc else None, METHOD_EXACT_KEY, "page host")

    record(
        "logo_url",
        _read_logo_url(embedded_json or [], base),
        METHOD_EXACT_KEY,
        "json-ld logo",
    )

    body = tree.body
    text = " ".join((body.text(deep=True) if body else "").split())[:_TRANSPORT_SCAN_CHARS]
    record(
        "transportation_type",
        normalize_transport(text),
        METHOD_HEURISTIC_HTML,
        "page text",
    )

    return fields


def extract_page_from_html(
    url: str, html: str, embedded_json: Optional[list[dict]] = None
) -> ExtractedPage:
    """Build an :class:`ExtractedPage` from markup already in hand.

    Kept separate from :func:`extract_page` so a rule can be tried against a
    saved page without going back to the network, which is how every mapping
    rule in this stage was checked.

    Example:
        >>> page = extract_page_from_html(
        ...     "https://x.com/tour/1",
        ...     '<html><body><h1>تور دبی</h1>'
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div></body></html>',
        ... )
        >>> page.row_count, page.page_fields["tour_name"].value
        (2, 'تور دبی')
    """
    tree = HTMLParser(html)
    # Scripts and styles hold code, not content, and their text would otherwise
    # be read as part of a row.
    for node in tree.css("script, style, noscript"):
        node.decompose()

    page = ExtractedPage(url=url, html_raw=html, embedded_json=embedded_json or [])
    page.page_fields = read_page_fields(tree, url, page.embedded_json)
    page.unresolved_page_fields = unresolved(page.page_fields, PAGE_FIELDS)
    for name, scope in _PAGE_ATTEMPTS.items():
        if name not in page.page_fields:
            record_attempt(page.page_attempts, name, scope)

    containers = find_row_containers(tree, page_url=url)
    if not containers:
        page.notes.append("no repeating block with prices found")
        return page

    best = containers[0]
    if best.score < MIN_ROW_SCORE:
        page.notes.append(
            f"best block scored {best.score:.2f}, below {MIN_ROW_SCORE} "
            f"({'; '.join(best.reasons)})"
        )
        return page

    page.rows = map_rows(best)
    page.notes.append(f"rows from {best.path} (score {best.score:.2f})")
    return page


async def extract_page(
    url: str,
    profile: SiteProfile,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExtractedPage:
    """Fetch one tour URL and read everything step 3 knows how to read.

    A fetch that failed is returned as it came back, error and all, rather than
    being turned into an empty page: the reason a URL yielded nothing is worth
    as much to the daily refresh as the data would have been.

    Example:
        >>> profile = SiteProfile(domain="nahalgasht.com")
        >>> page = await extract_page("https://nahalgasht.com/tour/1", profile)
        >>> page.status_code
        200
    """
    fetched = await fetch_page(url, profile, client=client, timeout=timeout)
    if not fetched.html_raw:
        return fetched

    page = extract_page_from_html(url, fetched.html_raw, fetched.embedded_json)
    # Carry over what the fetch layer learned, so provenance survives.
    page.fetch_method = fetched.fetch_method
    page.status_code = fetched.status_code
    page.fetched_at = fetched.fetched_at
    page.content_hash = fetched.content_hash
    page.notes = fetched.notes + page.notes
    return page
