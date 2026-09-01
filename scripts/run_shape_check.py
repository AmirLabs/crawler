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

# How many candidates may be opened to find ``pages`` that still sell something.
# A shape can hold a page whose departures have all gone, and giving up at the
# first of those would report far fewer pages than the site has.
ATTEMPTS_PER_PAGE = 3

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

    result["pages"] = await read_pages(profile, found.tour_urls, pages)
    print(f"  done in {time.monotonic() - started:.0f}s", flush=True)
    return result


async def read_pages(
    profile: SiteProfile, urls: list[str], wanted: int
) -> list[dict]:
    """Read pages from the approved list until ``wanted`` of them still sell."""
    out: list[dict] = []
    if not urls:
        return out

    step = max(1, len(urls) // max(wanted * ATTEMPTS_PER_PAGE, 1))
    candidates = urls[::step][: wanted * ATTEMPTS_PER_PAGE]

    for url in candidates:
        if len(out) >= wanted:
            break
        started = time.monotonic()
        try:
            page = await extract_page(url, profile, timeout=40)
        except Exception as error:
            print(f"    FAILED {type(error).__name__} {url[:70]}", flush=True)
            continue
        rows = live_rows(page)
        print(f"    [{len(out) + 1}/{wanted}] {len(rows)} live,"
              f" {len(page.expired_rows)} departed, {time.monotonic() - started:.0f}s"
              f"  {url[:64]}", flush=True)
        if not rows:
            continue
        out.append({"url": url, "rows": rows, "notes": page.notes,
                    "departed": len(page.expired_rows)})
        await asyncio.sleep(1.0)
    return out


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
    lines: list[str] = [
        f"{result['domain']} — what the approved pages give",
        f"{len(pages)} pages, audited {result['stamp']}",
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
