"""Show, for the four sites that used to yield nothing, how a URL is chosen and what it gives.

Two questions are open at once, and one file each answers one of them:

    filtering.txt   which URL shapes this site has, which were opened to find
                    out, what each sample gave, and which shapes the crawl will
                    follow from now on
    tours.txt       the pages behind the approved shapes, and every value the
                    client would receive from them

Only live offers reach ``tours.txt``. A departure that has already gone says
nothing about whether the rules work, and it crowds out the rows that do.

The four sites are the ones that returned zero rows before rendering and shape
probing were added: tahagasht, lachinseir, chakavaktravel and radingasht.
alefbatour and avisatravel are deliberately absent — they publish a JSON API
that answers exactly, so nothing here would be testing our rules.

Run::

    .venv/bin/python scripts/run_shape_check.py
    .venv/bin/python scripts/run_shape_check.py --domain tahagasht
    .venv/bin/python scripts/run_shape_check.py --pages 5

Expect four to six minutes per site before the first page is read: every shape
is sampled, and a sample that yields nothing is opened again in a browser.
Progress is printed as it goes, so a long run can be told from a hung one.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.config import PROJECT_ROOT
from crawler.extraction.models import ExtractedPage
from crawler.extraction.page_extract import extract_page
from crawler.filtering.shape_probe import SAMPLES_PER_SHAPE, keep_proven_shapes
from crawler.filtering.url_filter import _discover, group_by_shape
from crawler.profiler.models import SiteProfile
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour

LOGS = PROJECT_ROOT / "logs" / "shape_check"

SITES = ("tahagasht", "lachinseir", "chakavaktravel", "radingasht")

DEFAULT_PAGES = 10

# How many candidates may be opened per page wanted. tahagasht.com answered 404
# on nine of its first ten, so a small budget reports an empty site that is not.
ATTEMPTS_PER_PAGE = 12

ROWS_SHOWN = 20
SAMPLE_URLS_SHOWN = 3

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def site_name(domain: str) -> str:
    """The folder a domain's files go in.

    Example:
        >>> site_name("https://www.tahagasht.com")
        'tahagasht'
    """
    host = domain.split("//")[-1].split("/")[0]
    return host.replace("www.", "").split(".")[0]


def live_rows(page: ExtractedPage) -> list[dict]:
    """Every row of a page that has not already departed, as the client sees it.

    Example:
        >>> live_rows(ExtractedPage(url="https://x.com/t/1"))
        []
    """
    if not page.rows:
        return []
    kept = ExtractedPage(
        url=page.url,
        page_fields=page.page_fields,
        embedded_json=page.embedded_json,
        rows=page.rows,
    )
    return build_tour(kept).to_rows()


async def check_site(domain: str, pages: int) -> dict:
    """Probe one site's shapes, then read what the approved ones give."""
    name = site_name(domain)
    print(f"\n=== {name}", flush=True)
    started = time.monotonic()

    result: dict = {"name": name, "domain": domain, "stamp": stamp()}

    print("  profiling ...", flush=True)
    profile: SiteProfile = await profile_site(domain, render=False)
    result["profile"] = profile
    if not profile.reachable:
        result["error"] = profile.error
        print(f"  UNREACHABLE: {profile.error}", flush=True)
        return result

    print("  finding tour URLs ...", flush=True)
    found = await _discover(profile)
    result["source"] = found.source
    result["seen"] = found.total_seen
    before = list(found.tour_urls)
    result["before"] = before
    result["groups"] = group_by_shape(sorted(before))
    print(f"  {found.total_seen} URLs seen, {len(before)} kept,"
          f" {len(result['groups'])} shapes", flush=True)

    print(f"  opening up to {SAMPLES_PER_SHAPE} URLs of each shape ...", flush=True)
    found = await keep_proven_shapes(profile, found)
    result["after"] = list(found.tour_urls)
    result["shapes"] = profile.url_shapes
    print(f"  approved {profile.url_shapes.tour_shapes or '(none)'};"
          f" {len(before)} -> {len(found.tour_urls)} URLs", flush=True)

    result["pages"], result["skipped"] = await read_pages(
        profile, found.tour_urls, pages
    )
    print(f"  done in {time.monotonic() - started:.0f}s", flush=True)
    return result


def _spread(urls: list[str], wanted: int) -> list[str]:
    """The approved list reordered so the first tries are spread across it.

    Taking the first N would take one destination: these lists are alphabetical
    and a site's first forty URLs are often forty departures of one tour.

    Example:
        >>> _spread(["a", "b", "c", "d"], 2)
        ['a', 'c', 'b', 'd']
    """
    step = max(1, len(urls) // max(wanted, 1))
    first = urls[::step]
    rest = [url for url in urls if url not in set(first)]
    return first + rest


async def read_pages(
    profile: SiteProfile, urls: list[str], wanted: int
) -> tuple[list[dict], list[dict]]:
    """Keep opening approved URLs until ``wanted`` of them still sell something.

    Returns the pages that sold and, beside them, every URL that did not and
    the reason. tahagasht.com's sitemap lists departures it has since taken
    down, so nine of ten first tries answered 404 — a fact worth reading rather
    than a run worth cutting short.
    """
    out: list[dict] = []
    skipped: list[dict] = []
    if not urls:
        return out, skipped

    for url in _spread(urls, wanted)[:wanted * ATTEMPTS_PER_PAGE]:
        if len(out) >= wanted:
            break
        started = time.monotonic()
        try:
            page = await extract_page(url, profile, timeout=40)
        except Exception as error:
            skipped.append({"url": url, "why": f"{type(error).__name__}: {error}"})
            print(f"    skipped ({type(error).__name__})  {url[:62]}", flush=True)
            continue

        rows = live_rows(page)
        took = time.monotonic() - started
        if rows:
            out.append({"url": url, "rows": rows, "notes": page.notes,
                        "departed": len(page.expired_rows)})
            print(f"    [{len(out)}/{wanted}] {len(rows)} live,"
                  f" {len(page.expired_rows)} departed, {took:.0f}s  {url[:58]}",
                  flush=True)
        else:
            if page.status_code and page.status_code >= 400:
                why = f"the site answered {page.status_code}"
            elif page.expired_rows:
                why = f"{len(page.expired_rows)} offers, all departed"
            else:
                why = "no offer could be read"
            skipped.append({"url": url, "why": why})
            print(f"    skipped ({why})  {url[:56]}", flush=True)
        await asyncio.sleep(1.0)

    if len(out) < wanted:
        print(f"    only {len(out)} of {wanted} sold after"
              f" {len(out) + len(skipped)} tries", flush=True)
    return out, skipped


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

HEAD = "=" * 86


def write_filtering(result: dict, folder: Path) -> None:
    lines: list[str] = [
        f"{result['domain']} — how a URL is chosen",
        f"audited {result['stamp']}",
        "",
        "A shape is kept only when a page of that shape gave a row. A shape",
        "nothing could be read from is kept too: losing a site to a bad",
        "afternoon is worse than opening a few pages that sell nothing.",
        HEAD,
    ]
    if result.get("error"):
        lines.append(f"no URL was audited: {result['error']}")
        folder.joinpath("filtering.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    shapes = result["shapes"]
    lines += [
        f"[where the URLs came from]",
        f"   source            {result['source']}",
        f"   URLs seen         {result['seen']}",
        f"   kept by scoring   {len(result['before'])}",
        f"   kept after probe  {len(result['after'])}",
        "",
        "[the shapes this site has]",
    ]
    for shape, urls in sorted(result["groups"].items(), key=lambda kv: -len(kv[1])):
        if shape in shapes.tour_shapes:
            verdict = "APPROVED — the crawl follows this"
        elif shape in shapes.guide_shapes:
            verdict = "REJECTED — read, and it sold nothing"
        else:
            verdict = "not judged"
        lines.append(f"   {shape}   ({len(urls)} URLs)   {verdict}")
        for url in sorted(urls)[:SAMPLE_URLS_SHOWN]:
            lines.append(f"       {url}")

    lines += ["", "[what each opened page gave]"]
    for note in shapes.notes:
        lines.append(f"   {note}")
    if not shapes.notes:
        lines.append("   (nothing was opened)")

    lines += [
        "",
        "[the answer kept on the site's profile]",
        f"   tour shapes   {shapes.tour_shapes or '(none)'}",
        f"   guide shapes  {shapes.guide_shapes or '(none)'}",
        f"   tested        {shapes.tested}",
        "",
        "[URLs the crawl will now use]",
    ]
    for url in result["after"][:20]:
        lines.append(f"   {url}")
    if len(result["after"]) > 20:
        lines.append(f"   ... and {len(result['after']) - 20} more")

    folder.joinpath("filtering.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tours(result: dict, folder: Path) -> None:
    pages = result.get("pages") or []
    skipped = result.get("skipped") or []
    lines: list[str] = [
        f"{result['domain']} — what the approved pages give",
        f"{len(pages)} pages sold, {len(skipped)} gave nothing,"
        f" audited {result['stamp']}",
        "",
        "Only offers that have not departed are listed. Empty fields are left",
        "out; a page showing more than "
        f"{ROWS_SHOWN} rows is cut with its full count kept.",
        HEAD,
    ]
    if result.get("error"):
        lines.append(f"no page was audited: {result['error']}")
    elif not pages:
        lines.append("no approved URL produced a live offer")

    if skipped:
        lines += [
            "",
            f"[{len(skipped)} URLs that passed the filter and gave nothing]",
            "Open these to say whether the page is really gone or we misread it.",
        ]
        for entry in skipped:
            lines.append(f"   {entry['why']}")
            lines.append(f"       {entry['url']}")
        lines.append(HEAD)

    for number, page in enumerate(pages, 1):
        lines += [
            "",
            f"page {number} of {len(pages)}   {len(page['rows'])} live rows,"
            f" {page['departed']} departed",
            f"  PAGE {page['url']}",
        ]
        for note in page["notes"][:6]:
            lines.append(f"  note  {note}")
        for index, row in enumerate(page["rows"][:ROWS_SHOWN], 1):
            lines.append(f"  --- row {index} of {len(page['rows'])} ---")
            for field, value in row.items():
                if value in (None, "", [], {}):
                    continue
                lines.append(f"      {field:<36} {value}")
        if len(page["rows"]) > ROWS_SHOWN:
            lines.append(f"  ... {len(page['rows']) - ROWS_SHOWN} more rows not shown")
        lines.append(HEAD)

    folder.joinpath("tours.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_site(result: dict) -> Path:
    folder = LOGS / result["name"]
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file():
            old.unlink()
    write_filtering(result, folder)
    write_tours(result, folder)
    return folder


async def main(domain: Optional[str], pages: int) -> None:
    from crawler.utils.domains import load_domains

    known = load_domains()
    wanted = [d for d in known if any(s in d for s in SITES)]
    if domain:
        wanted = [d for d in wanted if domain in d]
        if not wanted:
            print(f"no domain matching {domain!r} among {', '.join(SITES)}")
            return

    for target in wanted:
        result = await check_site(target, pages)
        folder = write_site(result)
        print(f"  wrote {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help=f"live pages to read per site (default {DEFAULT_PAGES})")
    args = parser.parse_args()
    asyncio.run(main(args.domain, args.pages))
