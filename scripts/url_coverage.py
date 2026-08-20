"""How many tour URLs are we missing, per site, and why.

Step 2 hands step 3 a list of URLs. Anything not on that list is invisible to
the whole pipeline, so a URL lost here costs more than a field lost later. This
script measures that loss two ways, because there are two different leaks:

* **Dropped by us.** The URL is in the site's sitemap, but
  :func:`~crawler.filtering.url_filter.discover_tour_urls` keeps only the one
  path shape that scored best, and a site that publishes tours under two shapes
  loses the second one entirely.
* **Missing from the sitemap.** The tour exists on the site and is reachable by
  clicking, but the sitemap never lists it. No amount of filtering can recover
  that one — it has to be crawled.

Neither number means anything without knowing what a lost URL actually was, so
nothing here is judged by its shape alone: a sample of the lost URLs is opened
and checked against what a tour page really looks like. A category page and an
"about us" page are losses of nothing and are reported separately.

Writes one file per site under ``logs/url_coverage/`` plus a combined summary::

    python3 scripts/url_coverage.py
    python3 scripts/url_coverage.py --domain bisanseir --sample 20
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from crawler.config import PROJECT_ROOT, USER_AGENT
from crawler.extraction.extractor import fetch_page
from crawler.extraction.page_extract import extract_page_from_html
from crawler.extraction.readers.prices import find_prices
from crawler.filtering.url_filter import TOUR_WORDS, discover_tour_urls
from crawler.profiler.models import SiteProfile
from crawler.profiler.profiler import profile_site
from crawler.utils.domains import load_domains

LOGS = PROJECT_ROOT / "logs" / "url_coverage"

# How many of the lost URLs to open and judge. Enough to tell "we are dropping
# real tours" from "we are dropping category pages" without a second crawl.
DEFAULT_SAMPLE = 15

# How far to walk the site looking for tours the sitemap never listed.
CRAWL_PAGES = 40
CRAWL_DELAY = 1.0

# How much pricing a page must show before it counts as selling something.
_MIN_PRICES_FOR_TOUR = 2

# Two tests were tried and dropped, both worth recording so they are not tried
# again. A heading fails because bisanseir.com's tour pages have no <h1>.
# Counting the tour links in a page fails because the bands overlap: two real
# tour pages carried 34 and 36, two category pages 30 and 49.
#
# What does separate them is the extractor itself. A tour page has a repeating
# block of priced options and yields rows; a category page shows one "from"
# price per tour and yields none. Measured on the same four pages: tours gave 4
# and 104 rows, categories gave 0 and 0.

VERDICT_TOUR = "tour"        # rows came out — a page we could use and are losing
VERDICT_MAYBE = "maybe"      # prices but no rows — a category, or a tour we cannot read
VERDICT_LISTING = "listing"  # says so in its own path — losing it is fine
VERDICT_OTHER = "other"      # no prices at all — losing it is fine
VERDICT_DEAD = "dead"        # did not answer


@dataclass
class Judged:
    """One opened URL and what it turned out to be."""

    url: str
    verdict: str
    rows: int = 0
    prices: int = 0
    heading: str = ""
    note: str = ""


@dataclass
class SiteCoverage:
    """Everything this script learned about one site's URL coverage."""

    domain: str
    name: str
    source: str = "none"
    seen: int = 0                     # URLs the sitemap/API offered
    kept: int = 0                     # URLs step 2 passed on
    dropped_shapes: list[tuple[str, int]] = field(default_factory=list)
    judged_dropped: list[Judged] = field(default_factory=list)
    # shape -> (how many URLs have it, what an opened one of them turned out
    # to be). Judging a shape rather than the whole pool is what makes the
    # answer actionable: alefbatour drops 610 URLs, but 329 of them are blog
    # posts and 18 are tours, and only the second number is a loss.
    shape_verdicts: dict[str, tuple[int, str]] = field(default_factory=dict)
    crawled: int = 0                  # pages walked looking for unlisted tours
    outside_sitemap: list[str] = field(default_factory=list)
    judged_outside: list[Judged] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        return sum(count for _, count in self.dropped_shapes)

    @staticmethod
    def _rates(judged: list[Judged]) -> tuple[float, float]:
        """Share of a sample that were tours, counting ``maybe`` out then in.

        Two numbers rather than one because ``maybe`` is genuinely undecided: a
        page with prices and no rows is either a category page or a tour we
        cannot read. The truth is between them, and pretending otherwise would
        make the estimate look firmer than the evidence.
        """
        usable = [j for j in judged if j.verdict != VERDICT_DEAD]
        if not usable:
            return 0.0, 0.0
        sure = sum(1 for j in usable if j.verdict == VERDICT_TOUR)
        unsure = sum(1 for j in usable if j.verdict == VERDICT_MAYBE)
        return sure / len(usable), (sure + unsure) / len(usable)

    @property
    def lost_in_sitemap(self) -> tuple[int, int]:
        """Tour URLs dropped from the sitemap, as (certain, plus undecided).

        Counted shape by shape. Scaling the whole dropped pool by one sample
        rate mixed a site's blog posts in with its tours and gave a range so
        wide it said nothing — alefbatour's came out 44 either way while the
        real answer was a handful of clearly named shapes.
        """
        certain = sum(
            count for count, verdict in self.shape_verdicts.values()
            if verdict == VERDICT_TOUR
        )
        undecided = sum(
            count for count, verdict in self.shape_verdicts.values()
            if verdict == VERDICT_MAYBE
        )
        return certain, certain + undecided

    @property
    def lost_outside(self) -> tuple[int, int]:
        low, high = self._rates(self.judged_outside)
        count = len(self.outside_sitemap)
        return round(count * low), round(count * high)


def _short(domain: str) -> str:
    """Site name without scheme or www.

    Example:
        >>> _short("https://www.alefbatour.com")
        'alefbatour'
    """
    return (
        domain.replace("https://", "").replace("http://", "").replace("www.", "").split(".")[0]
    )


def _looks_tourish(url: str) -> bool:
    """Whether a URL is worth opening at all when hunting for lost tours.

    Only a first sieve: it decides what to spend a request on, never what the
    page is. That answer comes from :func:`_judge`.

    Example:
        >>> _looks_tourish("https://x.com/tour/d/152904/tour-armanestan")
        True
        >>> _looks_tourish("https://x.com/blog/post/10-lost-cities")
        False
    """
    decoded = unquote(urlparse(url).path.lower())
    return any(word in decoded for word in TOUR_WORDS)


def _judge(url: str, html: str, embedded_json: Optional[list[dict]]) -> Judged:
    """Decide what an opened page actually is.

    Judged on what the page yields, not on what it is called: a page the
    extractor gets rows out of is a tour page we should have had. A page with
    prices but no rows is reported apart, as ``maybe`` — it is either a category
    page, which costs nothing to lose, or a tour whose markup we cannot read
    yet, which is a loss of a different kind. Saying so keeps the answer honest
    where the evidence stops.

    Example:
        >>> j = _judge("https://x.com/tour/1",
        ...     '<body><div class="beds"><div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div></div></body>', None)
        >>> j.verdict, j.rows
        ('tour', 2)
        >>> _judge("https://x.com/blog/post/10-cities", "<body>سفر</body>", None).verdict
        'listing'
        >>> _judge("https://x.com/about", "<body>درباره ما</body>", None).verdict
        'other'
    """
    tree = HTMLParser(html)

    heading = ""
    node = tree.css_first("h1")
    if node is not None:
        heading = " ".join((node.text(deep=True) or "").split())[:80]

    # The navigation names every kind of tour a site sells, so links are counted
    # only after the page's furniture is gone. Counting them first made every
    # bisanseir tour page look like a category.
    for node in tree.css("script, style, noscript, nav, header, footer"):
        node.decompose()

    tour_links = {
        urljoin(url, (a.attributes.get("href") or "").strip())
        for a in tree.css("a[href]")
        if (a.attributes.get("href") or "").strip()
    }
    tour_links = {link for link in tour_links if _looks_tourish(link)}

    body = " ".join(((tree.body.text(deep=True) if tree.body else "") or "").split())
    prices = len(find_prices(body))

    page = extract_page_from_html(url, html, embedded_json)
    rows = len(page.rows)

    decoded = unquote(urlparse(url).path.lower())
    listing_marks = ("category", "دسته", "archive", "tag", "/blog", "/page/")

    if any(mark in decoded for mark in listing_marks):
        verdict = VERDICT_LISTING
    elif rows > 0:
        verdict = VERDICT_TOUR
    elif prices >= _MIN_PRICES_FOR_TOUR:
        verdict = VERDICT_MAYBE
    else:
        verdict = VERDICT_OTHER

    return Judged(
        url=url,
        verdict=verdict,
        rows=rows,
        prices=prices,
        heading=heading,
        note=f"{len(tour_links)} tour links",
    )


async def _open_and_judge(
    urls: list[str], profile: SiteProfile, limit: int
) -> list[Judged]:
    """Open a spread of ``urls`` and say what each one is.

    Picks evenly across the list rather than taking the head, since a sitemap is
    usually ordered and the first entries are all one kind of page.
    """
    if not urls:
        return []
    step = max(1, len(urls) // limit)
    picks = [urls[i] for i in range(0, len(urls), step)][:limit]

    judged: list[Judged] = []
    for url in picks:
        fetched = await fetch_page(url, profile, timeout=40)
        if not fetched.html_raw:
            judged.append(Judged(url=url, verdict=VERDICT_DEAD, note=str(fetched.error)))
        else:
            judged.append(_judge(url, fetched.html_raw, fetched.embedded_json))
        await asyncio.sleep(CRAWL_DELAY)
    return judged


async def _crawl_for_unlisted(
    profile: SiteProfile, known: set[str], budget: int
) -> tuple[list[str], int]:
    """Walk the site from its home page, collecting tour URLs the sitemap omits.

    This is the only way to answer "is the sitemap complete?" — the sitemap
    cannot report what it left out. The walk is deliberately shallow and small:
    it is measuring a gap, not building the list.

    Returns the URLs found that are not in ``known``, and how many pages were
    opened to find them.
    """
    root = profile.domain if profile.domain.startswith("http") else f"https://{profile.domain}"
    host = urlparse(root).netloc.replace("www.", "")

    seen_pages: set[str] = set()
    queue: list[str] = [root]
    found: set[str] = set()
    opened = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30
    ) as client:
        while queue and opened < budget:
            page_url = queue.pop(0)
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)

            try:
                resp = await client.get(page_url)
            except Exception:
                continue
            opened += 1
            await asyncio.sleep(CRAWL_DELAY)
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                continue

            tree = HTMLParser(resp.text)
            for anchor in tree.css("a[href]"):
                href = (anchor.attributes.get("href") or "").strip()
                if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                absolute = urljoin(page_url, href).split("#")[0].rstrip("/") + "/"
                if urlparse(absolute).netloc.replace("www.", "") != host:
                    continue
                if not _looks_tourish(absolute):
                    # Still worth walking a listing page, just not recording it.
                    if len(queue) < budget and absolute not in seen_pages:
                        queue.append(absolute)
                    continue
                if absolute not in known and absolute.rstrip("/") not in known:
                    found.add(absolute)
                if absolute not in seen_pages and len(queue) < budget:
                    queue.append(absolute)

    return sorted(found), opened


async def measure_site(domain: str, sample: int, crawl: int) -> SiteCoverage:
    """Measure both kinds of URL loss for one site."""
    coverage = SiteCoverage(domain=domain, name=_short(domain))

    profile = await profile_site(domain, render=False)
    if not profile.reachable:
        coverage.notes.append(f"site did not answer: {profile.error}")
        return coverage

    discovery = await discover_tour_urls(profile)
    coverage.source = discovery.source
    coverage.seen = discovery.total_seen
    coverage.kept = discovery.total_kept
    coverage.notes.extend(discovery.notes)

    kept = set(discovery.tour_urls)

    # --- leak one: in the sitemap, dropped by our shape filter -------------- #
    dropped: list[str] = []
    shapes: dict[str, int] = defaultdict(int)
    shape_urls: dict[str, list[str]] = {}
    for group in discovery.groups:
        if group.is_selected:
            continue
        tourish = [u for u in group.urls if _looks_tourish(u)]
        if tourish:
            shapes[group.shape] = len(tourish)
            shape_urls[group.shape] = tourish
            dropped.extend(tourish)
    coverage.dropped_shapes = sorted(shapes.items(), key=lambda kv: -kv[1])

    # One URL per dropped shape, so every shape gets an answer rather than the
    # largest ones getting all the sampling. Shapes are taken biggest first,
    # since those are where the loss would be.
    for shape, count in coverage.dropped_shapes[:sample]:
        first = next((u for u in shape_urls[shape]), None)
        if first is None:
            continue
        judged = (await _open_and_judge([first], profile, 1))[0]
        judged.note = shape
        coverage.judged_dropped.append(judged)
        coverage.shape_verdicts[shape] = (count, judged.verdict)

    # --- leak two: not in the sitemap at all -------------------------------- #
    everything_listed = kept | set(dropped)
    for group in discovery.groups:
        everything_listed.update(group.urls)
    outside, opened = await _crawl_for_unlisted(profile, everything_listed, crawl)
    coverage.crawled = opened
    coverage.outside_sitemap = outside
    coverage.judged_outside = await _open_and_judge(outside, profile, sample)

    return coverage


def write_site_report(coverage: SiteCoverage) -> Path:
    """Write one site's findings in a form that can be read without the code."""
    LOGS.mkdir(parents=True, exist_ok=True)
    lines = [
        f"site    {coverage.domain}",
        f"source  {coverage.source}",
        "",
        "WHAT STEP 2 SAW AND KEPT",
        "-" * 70,
        f"  URLs the sitemap/API offered      {coverage.seen}",
        f"  URLs passed to extraction         {coverage.kept}",
        "",
        "LEAK 1 — in the sitemap, dropped by our shape filter",
        "-" * 70,
        f"  tour-shaped URLs dropped          {coverage.dropped_total}",
    ]
    lines.append("  each shape below was opened once and judged on what came out:")
    lines.append("")
    lines.append(f"    {'count':>6}  {'verdict':9}  shape")
    for shape, count in coverage.dropped_shapes[:14]:
        verdict = coverage.shape_verdicts.get(shape, (0, "-"))[1]
        lines.append(f"    {count:6}  {verdict:9}  {shape[:80]}")

    if coverage.judged_dropped:
        low, high = coverage.lost_in_sitemap
        lines.append("")
        lines.append(f"  => {low} tour URLs certainly lost here, up to {high} counting the undecided")
        lines.append("")
        for judged in coverage.judged_dropped[:10]:
            lines.append(f"    [{judged.verdict:7}] rows={judged.rows:4} prices={judged.prices:4}  {judged.heading[:46]}")
            lines.append(f"                {judged.url[:110]}")

    lines += [
        "",
        "LEAK 2 — tour pages the sitemap never listed",
        "-" * 70,
        f"  pages walked from the home page   {coverage.crawled}",
        f"  tour-shaped URLs found unlisted   {len(coverage.outside_sitemap)}",
    ]
    if coverage.judged_outside:
        counts = defaultdict(int)
        for judged in coverage.judged_outside:
            counts[judged.verdict] += 1
        lines.append(f"  a sample of {len(coverage.judged_outside)} of them, opened:")
        for verdict, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {count:4}  {verdict}")
        low, high = coverage.lost_outside
        lines.append(f"  => {low} certainly lost here, up to {high} counting the undecided")
        lines.append("")
        for judged in coverage.judged_outside[:8]:
            lines.append(f"    [{judged.verdict:7}] rows={judged.rows:4} prices={judged.prices:4}  {judged.heading[:46]}")
            lines.append(f"                {judged.url[:110]}")

    if coverage.notes:
        lines += ["", "NOTES", "-" * 70]
        lines += [f"  - {note[:150]}" for note in coverage.notes[:10]]

    path = LOGS / f"{coverage.name}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_summary(results: list[SiteCoverage]) -> None:
    """One table putting every site's loss beside every other's."""
    LOGS.mkdir(parents=True, exist_ok=True)
    lines = [
        "Tour URLs we never reach, per site.",
        "",
        "'dropped'  is in the sitemap but thrown away by our shape filter.",
        "'unlisted' exists on the site but is not in the sitemap at all.",
        "Both are estimates: the count of candidates scaled by the share of an",
        "opened sample that turned out to be a real tour page rather than a",
        "category or a blog post.",
        "",
        f"{'site':13}{'seen':>7}{'kept':>7}{'dropped':>9}{'lost':>11}{'unlisted':>10}{'lost':>11}",
        "-" * 70,
    ]
    for coverage in results:
        dl, dh = coverage.lost_in_sitemap
        ol, oh = coverage.lost_outside
        lines.append(
            f"{coverage.name:13}{coverage.seen:>7}{coverage.kept:>7}"
            f"{coverage.dropped_total:>9}{f'{dl}-{dh}':>12}"
            f"{len(coverage.outside_sitemap):>10}{f'{ol}-{oh}':>12}"
        )

    total_kept = sum(c.kept for c in results)
    low = sum(c.lost_in_sitemap[0] + c.lost_outside[0] for c in results)
    high = sum(c.lost_in_sitemap[1] + c.lost_outside[1] for c in results)
    lines += [
        "-" * 70,
        "",
        f"We reach {total_kept} tour pages and miss somewhere between {low} and {high}.",
        "",
        "The range is the honest width of the evidence: its low end counts only",
        "pages the extractor got rows out of, its high end also counts pages that",
        "quote prices but yield nothing — each of those is either a category page",
        "or a tour whose markup we cannot read yet.",
    ]
    (LOGS / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


async def main(domain: Optional[str], sample: int, crawl: int) -> None:
    domains = load_domains()
    if domain:
        domains = [d for d in domains if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return

    results = []
    for target in domains:
        print(f"measuring {target} ...", flush=True)
        coverage = await measure_site(target, sample, crawl)
        results.append(coverage)
        path = write_site_report(coverage)
        dl, dh = coverage.lost_in_sitemap
        ol, oh = coverage.lost_outside
        print(
            f"  seen={coverage.seen} kept={coverage.kept} "
            f"dropped={coverage.dropped_total} ({dl}-{dh} tours), "
            f"unlisted={len(coverage.outside_sitemap)} ({ol}-{oh} tours) -> {path}",
            flush=True,
        )

    write_summary(results)
    print(f"\nwrote {LOGS / 'summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="URLs to open per leak")
    parser.add_argument("--crawl", type=int, default=CRAWL_PAGES, help="pages to walk per site")
    args = parser.parse_args()
    asyncio.run(main(args.domain, args.sample, args.crawl))
