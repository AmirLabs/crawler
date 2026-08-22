"""Crawl a few unseen pages per site and write down every field we got and every one we missed.

``run_audit.py`` answers "how much did we extract". This answers a different
question, the one that decides whether the LLM step is needed at all: **which
exact field, on which exact page, came out empty** — with the page URL beside
it, so Amir can open that page and say whether the site stated the value or
never had it. Only he can make that call; this script's job is to make the
check fast.

Three things it does that the older audit does not:

* **It picks pages we have not looked at before.** Every URL already recorded
  in ``logs/*/rows.jsonl`` or ``logs/*/pages.jsonl`` is skipped, so a second run
  reports on new pages instead of confirming the same ones. Judging the rules on
  the pages they were written against is how a rule looks better than it is.
* **It says why a field is empty**, not just that it is. Extraction records
  every method it tried and that came back empty (``record_attempt``), so
  "nothing looked for this" and "looked and found nothing" stay apart. The first
  is a gap in the rules; the second may simply be a page that does not say.
* **It indexes the misses by field**, so the question becomes "here are the 5
  pages where the departure date is empty — are they empty on the site too?"

Output, per site, under ``logs/<site>/``:

    fields.txt   page by page: the URL, what was filled, what was missing
    missing.txt  by field: every page URL where that field came out empty
    pages.jsonl  the same thing as JSON, and the record of what has been seen

and one combined ``logs/missing_all.txt`` putting the sites side by side.

Run::

    .venv/bin/python scripts/run_field_audit.py
    .venv/bin/python scripts/run_field_audit.py --domain tahagasht --pages 5
    .venv/bin/python scripts/run_field_audit.py --repeat-seen     # re-use old pages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
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
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.utils.domains import load_domains

LOGS = PROJECT_ROOT / "logs"

# Five is Amir's number, and it is meant to stay small: these pages get opened
# by hand, and a list nobody finishes checking is worth less than a short one
# somebody does.
DEFAULT_PAGES = 5

# The client's own output fields, grouped the way the eye reads them. Grouping
# matters here: "the whole date group is empty" and "one of eight fields is
# empty" are different problems, and a flat list hides which one this is.
FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("what tour", ("tour_name", "origin", "destination", "transportation_type")),
    ("when", ("departure_date", "return_date", "number_of_nights")),
    ("where you sleep", ("hotels", "hotel_star_rating", "board_type")),
    ("what it costs", ("adult_price", "adult_price_currency", "single_price")),
)
REVIEW_FIELDS: tuple[str, ...] = tuple(f for _, fields in FIELD_GROUPS for f in fields)

# An output column and the extraction field behind it, where the two are named
# differently. Needed to look up why a value is missing: the reason is recorded
# under the extraction name, and the report is written in the client's names.
_EXTRACTION_NAME: dict[str, str] = {
    "number_of_nights": "nights",
    "hotels": "hotel_name",
    "adult_price": "price",
    "adult_price_currency": "price",
    "single_price": "price",
}

# What a missing field means, in the two cases the pipeline can tell apart.
_TRIED_AND_FAILED = "looked, found nothing"
_NEVER_LOOKED = "nothing looked for it"
# A page that produced no rows at all fails every field for one single reason,
# and saying "nothing looked for it" thirteen times hides that. The page is the
# problem, not the fields.
_NO_ROWS = "no rows were read from this page — see the note below"


def _short(name: str) -> str:
    """Site name without scheme or www, for filenames and table rows.

    Example:
        >>> _short("https://www.alefbatour.com")
        'alefbatour'
    """
    return name.replace("https://", "").replace("http://", "").replace("www.", "").split(".")[0]


def seen_urls(site: str) -> set[str]:
    """Every page URL this site has already been audited on.

    Read from the logs rather than kept in a state file, so deleting the log is
    all it takes to start over, and so the record cannot drift from the report
    it came from.

    Example:
        >>> isinstance(seen_urls("no-such-site"), set)
        True
    """
    found: set[str] = set()
    folder = LOGS / site
    for name in ("rows.jsonl", "pages.jsonl"):
        path = folder / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            url = record.get("tour_url") or record.get("url")
            if url:
                found.add(url)
    return found


def _why_missing(page: ExtractedPage, field: str) -> str:
    """Whether a missing field was looked for and failed, or never looked for.

    The distinction is the whole point of ``attempts``: a field nothing looked
    for is a gap in the rules, while a field several methods tried and failed on
    may be a page that simply does not state it. Only the second kind is worth
    sending to the LLM.

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
    """One page's report: the URL, the fields filled, and the ones missing.

    A field counts as filled when *any* row on the page carries it. A page
    selling one date across nine hotels fills the date once, and calling that a
    miss on eight rows would drown the real gaps.

    Example:
        >>> report = audit_page(ExtractedPage(url="https://x/tour/1"))
        >>> report["url"], report["rows"], report["filled"]
        ('https://x/tour/1', 0, [])
    """
    tour = build_tour(page)
    rows = tour.to_rows()

    filled: list[str] = []
    missing: list[tuple[str, str]] = []
    for field in REVIEW_FIELDS:
        if any(row.get(field) not in (None, "", []) for row in rows):
            filled.append(field)
        elif not rows:
            missing.append((field, _NO_ROWS))
        else:
            missing.append((field, _why_missing(page, field)))

    return {
        "url": page.url,
        "tour_url": page.url,
        "rows": len(rows),
        "expired_rows": len(page.expired_rows),
        "filled": filled,
        "missing": [{"field": f, "why": w} for f, w in missing],
        "notes": list(page.notes),
    }


async def audit_site(domain: str, pages: int, repeat_seen: bool) -> dict:
    """Crawl up to ``pages`` unseen tour pages of one site and report each one."""
    site = _short(domain)
    result: dict = {
        "domain": domain,
        "name": site,
        "status": "ok",
        "pages": [],
        "notes": [],
        "skipped_seen": 0,
    }

    profile = await profile_site(domain, render=False)
    if not profile.reachable:
        result["status"] = "UNREACHABLE"
        result["notes"].append(f"site did not answer: {profile.error}")
        return result

    discovery = await discover_tour_urls(profile)
    if not discovery.tour_urls:
        result["status"] = "NO URLS"
        result["notes"].append(
            f"site answered but no tour URLs were found "
            f"(source={discovery.source}, seen={discovery.total_seen})"
        )
        return result

    groups = group_by_tour(discovery.tour_urls)
    already = set() if repeat_seen else seen_urls(site)

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
    # a broken site, and a site with too few pages is a fact worth stating, not
    # a reason to skip the site.
    picks = fresh[:pages]
    if len(picks) < pages:
        topped_up = [u for u in ordered if u not in picks][: pages - len(picks)]
        picks += topped_up
        if topped_up:
            result["notes"].append(
                f"only {len(fresh)} unseen tour URLs left; "
                f"{len(topped_up)} of these pages were audited before"
            )

    if not picks:
        result["status"] = "NO PAGES"
        result["notes"].append(f"no tour URLs to audit ({len(candidates)} candidates)")
        return result

    # extract_page, not extract_page_from_html: the second one reads the markup
    # and nothing else, so a site whose offers come from its own API or from a
    # rendered page reports zero rows and looks broken. Measured on one
    # avisatravel page: 0 rows through the markup, 55 through this.
    extracted: list[ExtractedPage] = []
    for url in picks:
        try:
            page = await extract_page(url, profile, timeout=40)
        except Exception as error:  # a dead page must not end the run
            result["notes"].append(f"{url} -> {error}")
            continue
        extracted.append(page)
        await asyncio.sleep(1.0)

    # One LLM call for all the titles rather than one per page: the instructions
    # cost far more than a title does.
    if extracted:
        filled = fill_page_places(extracted)
        result["notes"].append(f"destinations read for {filled} of {len(extracted)} pages")

    result["pages"] = [audit_page(page) for page in extracted]
    return result


def write_site_report(result: dict) -> Path:
    """Write one site's page-by-page report, its missing-field index, and its JSON."""
    folder = LOGS / result["name"]
    folder.mkdir(parents=True, exist_ok=True)
    pages = result["pages"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"{result['domain']}",
        f"{len(pages)} pages, audited {stamp}",
        f"status {result['status']}   skipped as already audited: {result['skipped_seen']}",
        "",
        "Open each PAGE and check the MISSING list against it. A field this",
        "found is not necessarily right — it is only present.",
        "=" * 86,
    ]
    for number, page in enumerate(pages, start=1):
        lines.append("")
        lines.append(f"--- page {number} of {len(pages)} ---")
        lines.append(f"  PAGE     {page['url']}")
        lines.append(f"  rows     {page['rows']}"
                     + (f"   ({page['expired_rows']} already departed)" if page["expired_rows"] else ""))
        for title, fields in FIELD_GROUPS:
            got = [f for f in fields if f in page["filled"]]
            lines.append(f"  {title:16} {len(got)}/{len(fields)}   {', '.join(got) or '-'}")
        if page["missing"]:
            lines.append("  MISSING")
            for item in page["missing"]:
                lines.append(f"      {item['field']:28} {item['why']}")
        for note in page["notes"][:4]:
            lines.append(f"  note     {note[:140]}")
    (folder / "fields.txt").write_text("\n".join(lines), encoding="utf-8")

    by_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for page in pages:
        for item in page["missing"]:
            by_field[item["field"]].append((page["url"], item["why"]))

    lines = [
        f"{result['domain']} — what we did not get, and where to check it",
        f"{len(pages)} pages, audited {stamp}",
        "",
        "'looked, found nothing' means the rules ran and the page gave nothing —",
        "either the page does not say it, or our rule is wrong. Amir decides which.",
        "'nothing looked for it' means no rule even tried, which is ours to fix.",
        "=" * 86,
    ]
    for field in REVIEW_FIELDS:
        hits = by_field.get(field, [])
        lines.append("")
        lines.append(f"{field}   missing on {len(hits)} of {len(pages)} pages")
        for url, why in hits:
            lines.append(f"    {url}")
            lines.append(f"        {why}")
    (folder / "missing.txt").write_text("\n".join(lines), encoding="utf-8")

    with (folder / "pages.jsonl").open("a", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps({**page, "audited_at": stamp}, ensure_ascii=False) + "\n")

    return folder


def write_combined(results: list[dict]) -> None:
    """One table of how often each field was missing, per site."""
    LOGS.mkdir(exist_ok=True)
    names = [r["name"][:11] for r in results]
    totals = [len(r["pages"]) for r in results]
    misses = [
        Counter(item["field"] for page in r["pages"] for item in page["missing"])
        for r in results
    ]

    lines = [
        "How often each field was MISSING, per site (% of pages). Lower is better.",
        "The per-page URLs are in logs/<site>/missing.txt.",
        "",
        f"{'field':28}" + "".join(f"{n:>12}" for n in names),
        "-" * (28 + 12 * len(names)),
    ]
    for field in REVIEW_FIELDS:
        cells = [
            f"{(count[field] / total * 100):>11.0f}%" if total else f"{'-':>12}"
            for count, total in zip(misses, totals)
        ]
        lines.append(f"{field:28}" + "".join(cells))
    lines.append("")
    lines.append(f"{'status':28}" + "".join(f"{r['status']:>12}" for r in results))
    lines.append(f"{'pages':28}" + "".join(f"{t:>12}" for t in totals))
    lines.append(f"{'rows':28}" + "".join(
        f"{sum(p['rows'] for p in r['pages']):>12}" for r in results
    ))
    (LOGS / "missing_all.txt").write_text("\n".join(lines), encoding="utf-8")


async def main(domain: Optional[str], pages: int, repeat_seen: bool) -> None:
    domains = load_domains()
    if domain:
        domains = [d for d in domains if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return

    results = []
    for target in domains:
        print(f"auditing {target} ...", flush=True)
        result = await audit_site(target, pages, repeat_seen)
        results.append(result)
        folder = write_site_report(result)
        gaps = sum(len(p["missing"]) for p in result["pages"])
        print(
            f"  {len(result['pages'])} pages, "
            f"{sum(p['rows'] for p in result['pages'])} rows, "
            f"{gaps} missing fields -> {folder}",
            flush=True,
        )

    if len(results) > 1:
        write_combined(results)
        print("\nwrote logs/missing_all.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="pages per site")
    parser.add_argument(
        "--repeat-seen",
        action="store_true",
        help="allow pages a previous run already audited",
    )
    args = parser.parse_args()
    asyncio.run(main(args.domain, args.pages, args.repeat_seen))
