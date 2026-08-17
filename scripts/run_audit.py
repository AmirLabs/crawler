"""Crawl real tour pages and write what came out, for review by eye.

Fill rates say how much was extracted; they cannot say whether it was the
right thing. So each site gets its own folder under ``logs/`` holding three
files:

* ``summary.txt`` — how often each field was filled.
* ``samples.txt`` — whole rows in readable form, each with the URL that
  produced it, so a value can be checked against the page it came from.
* ``rows.jsonl`` — every row as JSON, for looking closer.

A combined ``logs/summary_all.txt`` puts the sites side by side, which is
where a rule that works on one site and fails on another shows up.

Run it against one site while shaping the rules, then against all of them::

    python3 scripts/run_audit.py --domain bisanseir --pages 20
    python3 scripts/run_audit.py --pages 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from crawler.config import PROJECT_ROOT
from crawler.extraction.extractor import fetch_page
from crawler.extraction.page_extract import extract_page_from_html
from crawler.filtering.tour_groups import group_by_tour
from crawler.filtering.url_filter import discover_tour_urls
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.utils.domains import load_domains

LOGS = PROJECT_ROOT / "logs"

# The columns worth checking by eye, in the order they read best.
REVIEW_FIELDS = (
    "tour_name",
    "origin",
    "destination",
    "transportation_type",
    "departure_date",
    "departure_date_jalali",
    "departure_time",
    "return_date",
    "number_of_nights",
    "hotels",
    "hotel_star_rating",
    "board_type",
    "adult_price",
    "adult_price_currency",
    "adult_price_foreign",
    "adult_price_foreign_currency",
    "single_price",
    "agency_name",
    "image_url",
)

SAMPLES_PER_SITE = 5


def _short(name: str) -> str:
    """Site name without scheme or www, for filenames and table rows.

    Example:
        >>> _short("https://www.alefbatour.com")
        'alefbatour'
    """
    return name.replace("https://", "").replace("http://", "").replace("www.", "").split(".")[0]


async def audit_site(domain: str, pages: int) -> dict:
    """Crawl one site and return everything worth reporting about it."""
    name = _short(domain)
    result: dict = {
        "domain": domain,
        "name": name,
        "rows": [],
        "notes": [],
        "pages": 0,
        # "the site did not answer" and "we extracted nothing" both end in zero
        # rows and mean opposite things: one is their problem and passes on its
        # own, the other is ours and needs a code change.
        "status": "ok",
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
    result["urls_found"] = discovery.total_kept
    result["tours_after_grouping"] = len(groups)

    # Spread the picks over the whole list rather than taking the first few,
    # which on these sites are all the same destination.
    step = max(1, len(groups) // pages)
    picks = [groups[i].primary for i in range(0, len(groups), step)][:pages]

    for url in picks:
        fetched = await fetch_page(url, profile, timeout=40)
        if not fetched.html_raw:
            result["notes"].append(f"{url} -> {fetched.error}")
            continue

        page = extract_page_from_html(url, fetched.html_raw, fetched.embedded_json)
        tour = build_tour(page)
        result["pages"] += 1
        for row in tour.to_rows():
            result["rows"].append(row)
        if not tour.offers:
            result["notes"].append(f"no offers: {url}  ({'; '.join(page.notes)})")
        await asyncio.sleep(1.0)

    return result


def _fill_counts(rows: list[dict]) -> Counter:
    """How many rows carry a value for each reviewed field."""
    counts: Counter = Counter()
    for row in rows:
        for field in REVIEW_FIELDS:
            if row.get(field) not in (None, "", []):
                counts[field] += 1
    return counts


def write_site_report(result: dict) -> Path:
    """Write one site's three files into its own folder under ``logs/``."""
    folder = LOGS / result["name"]
    folder.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    counts = _fill_counts(rows)

    lines = [
        f"site   {result['domain']}",
        f"status {result.get('status', 'ok')}",
        f"pages  {result['pages']}",
        f"rows   {len(rows)}",
        f"urls   {result.get('urls_found', 0)} found -> {result.get('tours_after_grouping', 0)} tours after grouping",
        "",
        "filled per field (%)",
        "-" * 60,
    ]
    for field in REVIEW_FIELDS:
        share = (counts[field] / len(rows) * 100) if rows else 0
        lines.append(f"  {field:32} {share:5.0f}%  {'#' * int(share / 5)}")
    if result["notes"]:
        lines.append("")
        lines.append(f"notes ({len(result['notes'])})")
        lines.append("-" * 60)
        for note in result["notes"][:20]:
            lines.append(f"  - {note[:160]}")
    (folder / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    lines = [f"{result['domain']}   {len(rows)} rows from {result['pages']} pages", "=" * 90]
    seen_urls: set[str] = set()
    shown = 0
    for row in rows:
        if shown >= SAMPLES_PER_SITE:
            break
        if row.get("tour_url") in seen_urls:
            continue
        seen_urls.add(row.get("tour_url"))
        shown += 1
        lines.append(f"\n--- sample {shown} ---")
        lines.append(f"  {'PAGE':32} {row.get('tour_url')}")
        for field in REVIEW_FIELDS:
            value = row.get(field)
            if value not in (None, "", []):
                lines.append(f"  {field:32} {value}")
    (folder / "samples.txt").write_text("\n".join(lines), encoding="utf-8")

    with (folder / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return folder


def write_combined_summary(results: list[dict]) -> None:
    """Put every site's fill rates in one table, for comparing them."""
    LOGS.mkdir(exist_ok=True)
    names = [r["name"][:11] for r in results]
    counts = [_fill_counts(r["rows"]) for r in results]
    totals = [len(r["rows"]) for r in results]

    lines = [
        "How often each field was filled, per site (%).",
        "A field that works on one site and not another is where the rules are",
        "still site-specific — that is what this table is for.",
        "",
        f"{'field':32}" + "".join(f"{n:>12}" for n in names),
        "-" * (32 + 12 * len(names)),
    ]
    for field in REVIEW_FIELDS:
        cells = []
        for count, total in zip(counts, totals):
            cells.append(f"{(count[field] / total * 100):>11.0f}%" if total else f"{'-':>12}")
        lines.append(f"{field:32}" + "".join(cells))

    lines.append("")
    lines.append(f"{'status':32}" + "".join(f"{r.get('status', 'ok'):>12}" for r in results))
    lines.append(f"{'pages':32}" + "".join(f"{r['pages']:>12}" for r in results))
    lines.append(f"{'rows':32}" + "".join(f"{t:>12}" for t in totals))
    lines.append(f"{'rows per page':32}" + "".join(
        f"{(t / r['pages']):>12.1f}" if r["pages"] else f"{'-':>12}"
        for t, r in zip(totals, results)
    ))
    (LOGS / "summary_all.txt").write_text("\n".join(lines), encoding="utf-8")


async def main(domain: Optional[str], pages: int) -> None:
    domains = load_domains()
    if domain:
        domains = [d for d in domains if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return

    results = []
    for target in domains:
        print(f"auditing {target} ...", flush=True)
        result = await audit_site(target, pages)
        results.append(result)
        folder = write_site_report(result)
        print(f"  {result['pages']} pages, {len(result['rows'])} rows -> {folder}", flush=True)

    if len(results) > 1:
        write_combined_summary(results)
        print("\nwrote logs/summary_all.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--pages", type=int, default=20, help="pages per site")
    args = parser.parse_args()
    asyncio.run(main(args.domain, args.pages))
