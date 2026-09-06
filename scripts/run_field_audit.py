"""Crawl a few tour pages per site and write down every field we got and every one we missed.

The question this answers is the one that decides what to fix next: **which
exact field, on which exact page, came out empty** — with the page URL beside
it, so Amir can open that page and say whether the site stated the value or
never had it. Only he can make that call; this script's job is to make the
check fast.

Four files per site, and no more, under ``logs/<site>/`` — or under
``logs/<release>/<site>/`` when ``--out`` names a release, so one run's numbers
can be kept beside the next one's and compared:

    report.txt    every page audited and every value it produced
    missing.txt   by field: the pages where it came out empty, and why
    score.txt     how much of each field we managed to read, and the sum
    skipped.txt   the URLs that sold nothing, and what they answered instead

Only a page that actually sold something is scored. A URL that answers 404, or
whose every departure has already gone, says nothing about our rules: counting
it put lachinseir.com at 0% on all sixty fields when the five URLs picked were
all dead, which read as broken extraction and was not. Those URLs go to
``skipped.txt`` for Amir to judge, and the script keeps opening more until it
has the five real pages it was asked for.

The field list is not written here. It is read from
:meth:`crawler.schema.output_models.Tour.to_rows`, so a column added to the
client's output appears in the report the same day. An earlier version listed
its fields by hand and showed thirteen of forty-five; the child and infant
prices were empty on every site and it never showed, because the report never
asked about them.

Run::

    .venv/bin/python scripts/run_field_audit.py
    .venv/bin/python scripts/run_field_audit.py --domain bisanseir
    .venv/bin/python scripts/run_field_audit.py --domain bisanseir --pages 3
    .venv/bin/python scripts/run_field_audit.py --repeat-seen
    .venv/bin/python scripts/run_field_audit.py --out v1.1.1

Progress is printed as it goes — one line per page, with the row count and the
time it took — so a long run can be told from a hung one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The project is not installed as a package, so a plain `python scripts/...`
# cannot see `crawler`. Adding the repo root here means the command in the
# docstring works as written, with no PYTHONPATH to remember.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.config import PROJECT_ROOT
from crawler.extraction.models import ExtractedPage
from crawler.extraction.page_extract import extract_page
from crawler.filtering.tour_groups import group_by_tour
from crawler.filtering.url_filter import discover_tour_urls
from crawler.llm.tour_names import fill_page_places
from crawler.profiler.models import SiteProfile
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.schema.output_models import Hotel, Tour, TourDate, TourOffer
from crawler.utils.domains import load_domains

LOGS = PROJECT_ROOT / "logs"

# Five is Amir's number, and it is meant to stay small: these pages get opened
# by hand, and a list nobody finishes checking is worth less than a short one
# somebody does.
DEFAULT_PAGES = 5

# How many rows of a page are written out in full. One bisanseir page over its
# 22 departures produced 863 rows and a 1.4 MB file, and a file nobody scrolls
# to the end of is not a check. Every field of every row shown is complete;
# what is cut is repetition, and the count of what was cut is printed.
ROWS_SHOWN = 25

# How many URLs may be opened for each page wanted, before giving up. Nine of
# tahagasht.com's first ten tries answer 404 because its sitemap still lists
# departures it has taken down, so a budget near the wanted count would report
# a live site as an empty one.
ATTEMPTS_PER_PAGE = 12

# httpx logs a line per request at INFO, and this script makes hundreds. Left
# on, its own progress lines are lost in the middle of them — which defeats the
# point of printing progress at all.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def output_fields() -> tuple[str, ...]:
    """Every column the client receives, asked of the model rather than listed.

    Building one empty row is what makes the report impossible to leave
    incomplete: a column added to ``to_rows`` shows up here with no edit.

    Example:
        >>> fields = output_fields()
        >>> "departure_date_jalali" in fields, "logo_url" in fields
        (True, True)
    """
    tour = Tour(tour_url="u", tour_name="t")
    tour.offers = [TourOffer(hotel=Hotel(name="h"), departure=TourDate(), ret=TourDate())]
    return tuple(tour.to_rows()[0])


ALL_FIELDS: tuple[str, ...] = output_fields()

# The same fields grouped the way the eye reads them. Grouping matters: "the
# whole date group is empty" and "one of eight is empty" are different
# problems, and a flat list of sixty hides which one this is.
#
# Nothing is lost by being left out. Whatever these groups do not name is
# collected into "anything else" by :func:`field_groups`, so the report stays
# complete even when this list falls behind the model.
# What every price column is spelled with, once per traveller type.
_PRICE_SUFFIXES = ("", "_currency", "_foreign", "_foreign_currency",
                   "_basis", "_includes_flight")

_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("what tour", ("tour_name", "origin", "destination", "destinations",
                   "transportation_type", "is_installment_available")),
    ("who sells it", ("agency_name", "website_url", "tour_url", "image_url", "logo_url")),
    ("when", ("departure_date", "return_date", "departure_date_jalali",
              "return_date_jalali", "departure_time", "return_time",
              "number_of_nights")),
    ("where you sleep", ("hotels", "hotel_star_rating", "hotel_city", "board_type")),
    ("how you travel", ("transport_name", "transport_duration")),
    ("price: two people sharing", tuple(f"adult_price{s}" for s in _PRICE_SUFFIXES)),
    ("price: single room", tuple(f"single_price{s}" for s in _PRICE_SUFFIXES)),
    ("price: three sharing", tuple(f"triple_price{s}" for s in _PRICE_SUFFIXES)),
    ("price: child with a bed", tuple(f"child_price{s}" for s in _PRICE_SUFFIXES)),
    ("price: child without a bed", tuple(f"child_no_bed_price{s}" for s in _PRICE_SUFFIXES)),
    ("price: infant", tuple(f"infant_price{s}" for s in _PRICE_SUFFIXES)),
)


def field_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The groups, plus every field the groups forgot.

    Example:
        >>> named = {f for _, fields in field_groups() for f in fields}
        >>> named == set(ALL_FIELDS)
        True
    """
    named = {f for _, fields in _GROUPS if fields for f in fields}
    rest = tuple(f for f in ALL_FIELDS if f not in named)
    groups = tuple((title, tuple(f for f in fields if f in ALL_FIELDS))
                   for title, fields in _GROUPS if fields)
    return groups + ((("anything else", rest),) if rest else ())


# What a missing field means, in the two cases the pipeline can tell apart.
_NEVER_LOOKED = "nothing looked for it"
_TRIED_AND_FAILED = "looked, found nothing"
_NO_ROWS = "the page produced no rows at all"

# Output names that are read under a different name inside extraction, so
# "why is it missing" can find the attempts that were recorded against it.
_EXTRACTION_NAME: dict[str, str] = {
    "number_of_nights": "nights",
    "hotels": "hotel_name",
    "departure_date_jalali": "departure_date",
    "return_date_jalali": "return_date",
    "departure_time": "departure_date",
    "return_time": "return_date",
    **{
        f"{prefix}_price{suffix}": "price"
        for prefix in ("adult", "single", "triple", "child", "child_no_bed", "infant")
        for suffix in ("", "_currency", "_foreign", "_foreign_currency",
                       "_basis", "_includes_flight")
    },
}


def _short(name: str) -> str:
    """Site name without scheme or www, for folder names and table rows.

    Example:
        >>> _short("https://www.alefbatour.com")
        'alefbatour'
    """
    return name.replace("https://", "").replace("http://", "").replace("www.", "").split(".")[0]


def seen_urls(site: str) -> set[str]:
    """Every page URL this site has already been audited on.

    Read back out of the site's own report rather than kept in a state file, so
    deleting the folder is all it takes to start over and the record can never
    drift from the report it came from.

    Example:
        >>> isinstance(seen_urls("no-such-site"), set)
        True
    """
    path = LOGS / site / "report.txt"
    if not path.exists():
        return set()
    return {
        line.split("PAGE", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("PAGE")
    }


def _why_missing(page: ExtractedPage, field: str) -> str:
    """Whether a missing field was looked for and failed, or never looked for.

    The distinction is the whole point of ``attempts``: a field nothing looked
    for is a gap in the rules, while a field several methods tried and failed on
    may be a page that simply does not state it.

    Example:
        >>> page = ExtractedPage(url="u")
        >>> page.page_attempts["origin"] = ["heuristic_html:title"]
        >>> _why_missing(page, "origin")
        'looked, found nothing (heuristic_html:title)'
        >>> _why_missing(page, "destination")
        'nothing looked for it'
    """
    name = _EXTRACTION_NAME.get(field, field)
    tried = list(page.page_attempts.get(name, []))
    for row in page.rows:
        for method in row.attempts.get(name, []):
            if method not in tried:
                tried.append(method)
    if tried:
        return f"{_TRIED_AND_FAILED} ({', '.join(tried[:3])})"
    return _NEVER_LOOKED


def audit_page(page: ExtractedPage) -> dict:
    """One page's report: its URL, its rows, and why each empty field is empty.

    A field counts as filled for the page when *any* row carries it. A page
    selling one date across nine hotels states the date once, and calling that
    a miss on eight rows would drown the real gaps — so the rows are counted
    separately, in ``rows_with``.

    Example:
        >>> report = audit_page(ExtractedPage(url="https://x/tour/1"))
        >>> report["url"], report["rows"], report["filled"]
        ('https://x/tour/1', 0, [])
    """
    rows = build_tour(page).to_rows()

    filled: list[str] = []
    missing: list[tuple[str, str]] = []
    rows_with: dict[str, int] = {}
    for field in ALL_FIELDS:
        carried = [r for r in rows if r.get(field) not in (None, "", [])]
        rows_with[field] = len(carried)
        if carried:
            filled.append(field)
        elif not rows:
            missing.append((field, _NO_ROWS))
        else:
            missing.append((field, _why_missing(page, field)))

    return {
        "url": page.url,
        "rows": len(rows),
        "expired_rows": len(page.expired_rows),
        "filled": filled,
        "rows_with": rows_with,
        "missing": [{"field": f, "why": w} for f, w in missing],
        "notes": list(page.notes),
        "data": rows,
    }


def _nothing_sold(page: ExtractedPage) -> str:
    """Why a URL that answered carried no offer, in the terms Amir judges it in.

    He is the one who decides whether a page was never a tour or is a tour that
    has gone, so the reason has to say which of the two the page looked like
    rather than folding both into "empty".

    Example:
        >>> _nothing_sold(ExtractedPage(url="u", status_code=404))
        'the site answered 404'
        >>> _nothing_sold(ExtractedPage(url="u"))
        'no offer could be read'
    """
    if page.status_code and page.status_code >= 400:
        return f"the site answered {page.status_code}"
    if page.expired_rows:
        return f"{len(page.expired_rows)} offers, all already departed"
    return "no offer could be read"


async def read_pages(
    profile: SiteProfile, urls: list[str], wanted: int
) -> tuple[list[ExtractedPage], list[dict]]:
    """Keep opening tour URLs until ``wanted`` of them sold something.

    Returns the pages that sold and, beside them, every URL that did not and
    what it answered instead. Nothing in the second list reaches the score.

    extract_page, not extract_page_from_html: the second reads the markup and
    nothing else, so a site whose offers come from its own API or from a page
    that offers several dates reports zero rows and looks broken.
    """
    kept: list[ExtractedPage] = []
    skipped: list[dict] = []
    for url in urls[: wanted * ATTEMPTS_PER_PAGE]:
        if len(kept) >= wanted:
            break
        started = time.monotonic()
        print(f"  [{len(kept) + 1}/{wanted}] {url[:88]}", flush=True)
        try:
            page = await extract_page(url, profile, timeout=40)
        except Exception as error:  # a dead page must not end the run
            skipped.append({"url": url, "why": f"{type(error).__name__}: {error}"})
            print(f"        skipped ({type(error).__name__}) after"
                  f" {time.monotonic() - started:.0f}s", flush=True)
            continue

        took = time.monotonic() - started
        sold = build_tour(page).to_rows()
        if sold:
            kept.append(page)
            print(f"        {len(sold)} rows, {len(page.expired_rows)} expired,"
                  f" {took:.0f}s", flush=True)
        else:
            why = _nothing_sold(page)
            skipped.append({"url": url, "why": why})
            print(f"        skipped ({why}) {took:.0f}s", flush=True)
        await asyncio.sleep(1.0)

    if len(kept) < wanted:
        print(f"  only {len(kept)} of {wanted} URLs sold, after"
              f" {len(kept) + len(skipped)} tries", flush=True)
    return kept, skipped


async def audit_site(domain: str, pages: int, repeat_seen: bool) -> dict:
    """Crawl one site until ``pages`` of its tour pages sell, and report each."""
    site = _short(domain)
    result: dict = {
        "domain": domain, "name": site, "status": "ok",
        "pages": [], "notes": [], "skipped": [], "skipped_seen": 0,
    }

    print(f"  profiling {domain} ...", flush=True)
    profile = await profile_site(domain, render=False)
    if not profile.reachable:
        result["status"] = "UNREACHABLE"
        result["notes"].append(f"site did not answer: {profile.error}")
        print(f"  UNREACHABLE: {profile.error}", flush=True)
        return result

    print("  finding tour URLs ...", flush=True)
    discovery = await discover_tour_urls(profile)
    if not discovery.tour_urls:
        result["status"] = "NO URLS"
        result["notes"].append(
            f"site answered but no tour URLs were found "
            f"(source={discovery.source}, seen={discovery.total_seen})"
        )
        print("  no tour URLs found", flush=True)
        return result

    groups = group_by_tour(discovery.tour_urls)
    already = set() if repeat_seen else seen_urls(site)
    print(f"  {len(discovery.tour_urls)} URLs -> {len(groups)} tours", flush=True)

    # Spread the picks across the whole list rather than taking the first few,
    # which on these sites are all one destination — and skip what a previous
    # run already reported on, so each run tests the rules on new pages.
    candidates = [g.primary for g in groups]
    step = max(1, len(candidates) // max(pages * 4, 1))
    ordered = list(dict.fromkeys(candidates[::step] + candidates))
    fresh = [u for u in ordered if u not in already]
    result["skipped_seen"] = len(ordered) - len(fresh)

    # Unseen pages first. If a site has run out of them, fall back to ones
    # already audited rather than reporting nothing: an empty report reads like
    # a broken site, and a site with too few pages is a fact worth stating.
    tries = fresh + [u for u in ordered if u in already]
    if len(fresh) < pages:
        result["notes"].append(
            f"only {len(fresh)} unseen tour URLs left; pages beyond that were"
            f" audited before"
        )

    if not tries:
        result["status"] = "NO PAGES"
        result["notes"].append(f"no tour URLs to audit ({len(candidates)} candidates)")
        return result

    extracted, result["skipped"] = await read_pages(profile, tries, pages)
    if not extracted:
        result["status"] = "NO LIVE PAGES"
        result["notes"].append(
            f"{len(result['skipped'])} URLs were opened and none sold anything"
            f" — see skipped.txt"
        )

    # One LLM call for all the titles rather than one per page: the instructions
    # cost far more than a title does. Most pages never get here — the title
    # rule already answered them during extraction — so the two counts are
    # reported apart. Reporting only the model's share said "0 of 5" on a site
    # whose score.txt showed 100%, which read as a fault and was not one.
    if extracted:
        print("  reading destinations from the titles ...", flush=True)
        by_rule = sum(1 for page in extracted if "destinations" in page.page_fields)
        by_model = fill_page_places(extracted)
        result["notes"].append(
            f"destinations: {by_rule} of {len(extracted)} pages from the title rule,"
            f" {by_model} more from the model"
        )

    result["pages"] = [audit_page(page) for page in extracted]
    return result


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def write_report(result: dict, folder: Path) -> None:
    """Every page audited and every value it gave, under its URL."""
    pages = result["pages"]
    lines = [
        f"{result['domain']} — what we extracted",
        f"{len(pages)} pages, audited {_stamp()}",
        "",
        "Every row we would send the client, with the page it came from. Empty",
        "fields are left out here and listed in missing.txt instead, and a page",
        f"showing more than {ROWS_SHOWN} rows is cut with its full count kept.",
        "=" * 86,
    ]
    for note in result["notes"]:
        lines.append(f"note  {note}")
    if not pages:
        lines += ["", f"no page was audited: {result['status']}"]
    for number, page in enumerate(pages, start=1):
        lines += ["", "=" * 86,
                  f"page {number} of {len(pages)}   {page['rows']} rows,"
                  f" {page['expired_rows']} already departed",
                  f"  PAGE {page['url']}"]
        for note in page["notes"]:
            lines.append(f"  note  {note}")
        if not page["data"]:
            lines.append("  (no rows)")
        for index, row in enumerate(page["data"][:ROWS_SHOWN], start=1):
            lines.append(f"  --- row {index} of {len(page['data'])} ---")
            for title, fields in field_groups():
                carried = [(f, row[f]) for f in fields if row.get(f) not in (None, "", [])]
                if not carried:
                    continue
                lines.append(f"      [{title}]")
                for name, value in carried:
                    lines.append(f"        {name:36} {value}")
        if len(page["data"]) > ROWS_SHOWN:
            lines.append(f"  ... and {len(page['data']) - ROWS_SHOWN} more rows on this"
                         f" page, the same shape as the ones above")
    (folder / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_missing(result: dict, folder: Path) -> None:
    """By field: the pages where it came out empty, and what was tried."""
    pages = result["pages"]
    by_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for page in pages:
        for item in page["missing"]:
            by_field[item["field"]].append((page["url"], item["why"]))

    lines = [
        f"{result['domain']} — what we could not read",
        f"{len(pages)} pages, audited {_stamp()}",
        "",
        "Open a page and check the field against it. An empty field is not",
        "always a fault: a site that never states the hotel city has nothing",
        "for us to read. The two reasons are kept apart —",
        f"  '{_NEVER_LOOKED}'   no rule even tried: a gap in our rules",
        f"  '{_TRIED_AND_FAILED}'  rules ran and came back empty: maybe the page",
        "=" * 86,
    ]
    # No pages is not "every field read". A site that did not answer has to say
    # so here, or its report reads like a clean sweep.
    if not pages:
        lines += ["", f"no page was audited: {result['status']}"]
        for note in result["notes"]:
            lines.append(f"  {note}")
        (folder / "missing.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    for title, fields in field_groups():
        gaps = [(f, by_field[f]) for f in fields if by_field.get(f)]
        lines += ["", f"[{title}]"]
        if not gaps:
            lines.append("   every field read on every page")
            continue
        for name, where in gaps:
            lines.append(f"   {name}   empty on {len(where)} of {len(pages)} pages")
            for url, why in where:
                lines.append(f"       {why}")
                lines.append(f"       {url}")
    (folder / "missing.txt").write_text("\n".join(lines), encoding="utf-8")


def write_score(result: dict, folder: Path) -> None:
    """How much of each field we managed to read, and how that is counted."""
    pages = result["pages"]
    total_pages = len(pages)
    total_rows = sum(p["rows"] for p in pages)

    lines = [
        f"{result['domain']} — how much we read",
        f"{total_pages} pages, {total_rows} rows, audited {_stamp()}",
        "",
        "Two numbers per field, because they answer different questions:",
        "",
        "  pages  the share of pages where at least one row carried the field.",
        "         A page states its departure once for nine hotels, so this is",
        "         the honest measure of 'did we read it from this page at all'.",
        "  rows   the share of all rows that carry it. Lower is not always",
        "         worse: a single-room price exists only where the site sells",
        "         one, and a row without it is not a miss.",
        "",
        "A low score is a question, not a verdict. Check missing.txt for the",
        "pages behind it before deciding the rules are at fault.",
        "=" * 86,
    ]
    if not total_pages:
        lines.append(f"\nnothing to score: {result['status']}")
        for note in result["notes"]:
            lines.append(f"  {note}")
        (folder / "score.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    overall: list[float] = []
    for title, fields in field_groups():
        lines += ["", f"[{title}]", f"   {'field':38}{'pages':>7}{'rows':>7}   "]
        for name in fields:
            on_pages = sum(1 for p in pages if name in p["filled"])
            in_rows = sum(p["rows_with"].get(name, 0) for p in pages)
            page_pct = on_pages * 100 / total_pages
            row_pct = in_rows * 100 / total_rows if total_rows else 0
            overall.append(page_pct)
            bar = "#" * int(page_pct / 5)
            lines.append(f"   {name:38}{page_pct:6.0f}%{row_pct:6.0f}%   {bar}")

    lines += ["", "=" * 86,
              f"average across all {len(ALL_FIELDS)} fields: "
              f"{sum(overall) / len(overall):.0f}% of pages"]
    (folder / "score.txt").write_text("\n".join(lines), encoding="utf-8")


def write_skipped(result: dict, folder: Path) -> None:
    """The URLs that passed the filter and sold nothing, for Amir to judge.

    Kept out of the score on purpose: a page that has been taken down measures
    the site, not our rules. Which of the two it is — never a tour, or a tour
    whose dates have passed — is his call, and this is the list to make it on.
    """
    skipped = result.get("skipped") or []
    lines = [
        f"{result['domain']} — URLs that gave nothing",
        f"{len(result['pages'])} pages scored, {len(skipped)} skipped,"
        f" audited {_stamp()}",
        "",
        "None of these counted towards the score. Open one and say which it is:",
        "a URL that was never a tour, or a tour whose departures have passed.",
        "=" * 86,
        "",
    ]
    if not skipped:
        lines.append("every URL opened sold something")
    for entry in skipped:
        lines += [f"   {entry['why']}", f"   {entry['url']}", ""]
    (folder / "skipped.txt").write_text("\n".join(lines), encoding="utf-8")


def write_site(result: dict) -> Path:
    """The three files for one site, replacing whatever was there before."""
    folder = LOGS / result["name"]
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file():
            old.unlink()
    write_report(result, folder)
    write_missing(result, folder)
    write_score(result, folder)
    write_skipped(result, folder)
    return folder


async def main(domain: Optional[str], pages: int, repeat_seen: bool) -> None:
    domains = load_domains()
    if domain:
        domains = [d for d in domains if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return

    started = time.monotonic()
    print(f"auditing {len(domains)} site(s), up to {pages} pages each\n", flush=True)
    for number, target in enumerate(domains, start=1):
        print(f"[{number}/{len(domains)}] {target}", flush=True)
        result = await audit_site(target, pages, repeat_seen)
        folder = write_site(result)
        rows = sum(p["rows"] for p in result["pages"])
        gaps = sum(len(p["missing"]) for p in result["pages"])
        print(f"  -> {len(result['pages'])} pages, {rows} rows, {gaps} empty fields,"
              f" {len(result['skipped'])} URLs skipped"
              f"  written to {folder}\n", flush=True)

    print(f"done in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="pages per site")
    parser.add_argument(
        "--repeat-seen",
        action="store_true",
        help="allow pages a previous run already audited",
    )
    parser.add_argument("--out", help="write under logs/<OUT>/ instead of logs/")
    args = parser.parse_args()
    if args.out:
        LOGS = LOGS / args.out
    asyncio.run(main(args.domain, args.pages, args.repeat_seen))
