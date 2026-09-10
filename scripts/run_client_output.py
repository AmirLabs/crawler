"""Crawl every site and write both of the JSON files the client asked for.

They asked for two shapes of the same catalogue and this writes them side by
side, from one crawl, so the two can be read against each other::

    logs/client/first_structure.json    one object per date, hotel and room
    logs/client/second_structure.json   one object per date, with a "from" price
    logs/client/summary.txt             what was crawled, and what came out

Every site goes into the same two files, because that is what the client is
sent. Which agency a row came from is inside the row in the second structure —
``agency_name`` and the ``agency_info`` block — and nowhere yet in the first,
which carries no agency key of its own.

The grains differ on purpose. The first structure is the catalogue — every room
of every hotel on every date, which is what a person books. The second is a
listing — one line per departure, carrying the route, the carrier and the
cheapest price, which is what a search page shows before anyone clicks.

**A tour keeps one ``tour_code``, in both files and across nights.** One
:class:`IdStore` serves the whole run and is read from and written back to
``data/ids.json``, so a listing in the second file can be looked up in the
first, and a tour we sent last night is the same tour tonight. The client
checks a record against ``api_id``, which is built on that code: a code that
moved overnight would point their database at a different tour.

That file cannot be rebuilt if it is lost — nothing can work out which tour
held ``31540001`` yesterday — so it is written through a temporary name and a
rename, with dated copies kept beside it. Wherever this ends up running, it
needs a backup that is not on the same disk.

``logs/`` is emptied first. What was there came from earlier runs of a
different script and reading a new run against it would compare two different
questions.

Only one of these runs at a time, and everything it prints is kept under
``data/runs`` for a fortnight — ``logs/`` is emptied at the start of every
crawl, so a night that went wrong would otherwise leave no record. It exits
non-zero when a site failed, which is what tells whatever schedules it.

Run::

    .venv/bin/python scripts/run_client_output.py --pages 0   # every page
    .venv/bin/python scripts/run_client_output.py --domain nahalgasht --pages 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.api.store import CHANGES_FILE, PACKAGES_FILE, TOURS_FILE, publish
from crawler.automation.memory import UNCHANGED_NOTE, RunMemory
from crawler.automation.scheduler import AlreadyRunning, only_one_run, run_log
from crawler.config import PROJECT_ROOT
from crawler.extraction.page_extract import extract_page
from crawler.filtering.tour_groups import group_by_tour
from crawler.filtering.url_filter import discover_tour_urls
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.schema.identifiers import IdStore
from crawler.schema.safarbank import tour_to_objects
from crawler.schema.safarbank_tours import listing_document, tour_to_listings
from crawler.utils.domains import load_domains

LOGS = PROJECT_ROOT / "logs"
OUT = LOGS / "client"

DEFAULT_PAGES = 1

# How many URLs may be opened for each page wanted. Nine of tahagasht's first
# ten tries answer 404 because its sitemap still lists departures it has taken
# down, so a budget near the wanted count would report a live site as an empty
# one. The same figure the field audit settled on.
ATTEMPTS_PER_PAGE = 12

# Between requests to one site, on top of whatever its robots.txt asked for.
POLITE_PAUSE = 1.0


def site_name(domain: str) -> str:
    """The short name a site's files are written under.

    Example:
        >>> site_name("https://www.alefbatour.com"), site_name("https://x.co/")
        ('alefbatour', 'x')
    """
    host = domain.split("//")[-1].strip("/")
    return host.replace("www.", "").split(".")[0]


def clear_logs() -> None:
    """Empty ``logs/`` so a run is read on its own and not against an old one."""
    if LOGS.exists():
        shutil.rmtree(LOGS)
    OUT.mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


async def crawl_site(
    domain: str, pages: int, ids: IdStore, memory: RunMemory
) -> dict:
    """Open tour pages until ``pages`` of them sold something, and convert each.

    Only a page that actually sells is kept. A URL that answers 404, or whose
    every departure has already gone, says nothing about our rules — it says
    the site has moved on — so it is recorded and the crawl keeps going.
    """
    name = site_name(domain)
    result: dict = {
        "domain": domain, "name": name, "first": [], "second": [],
        "kept": [], "skipped": [], "notes": [], "unchanged": 0,
    }
    print(f"\n=== {name}", flush=True)

    profile = await profile_site(domain, render=False)
    if not profile.reachable:
        result["notes"].append(f"unreachable: {profile.error}")
        print(f"  UNREACHABLE: {profile.error}", flush=True)
        return result

    print("  finding tour URLs ...", flush=True)
    found = await discover_tour_urls(profile)
    groups = group_by_tour(found.tour_urls)
    candidates = [group.primary for group in groups]
    print(f"  {len(found.tour_urls)} URLs -> {len(groups)} tours", flush=True)
    if not candidates:
        result["notes"].append("no tour URLs found")
        return result

    budget = len(candidates) if pages == 0 else pages * ATTEMPTS_PER_PAGE
    for url in candidates[:budget]:
        if pages and len(result["kept"]) >= pages:
            break
        started = time.monotonic()
        print(f"  [{len(result['kept']) + 1}/{pages}] {url[:84]}", flush=True)
        try:
            page = await extract_page(url, profile, timeout=40,
                                      unchanged=memory.unchanged)
        except Exception as error:  # a dead page must not end the run
            result["skipped"].append({"url": url, "why": f"{type(error).__name__}"})
            print(f"        skipped ({type(error).__name__})", flush=True)
            continue

        if UNCHANGED_NOTE in page.notes:
            result["unchanged"] += 1
            print(f"        unchanged, not read ({time.monotonic() - started:.0f}s)",
                  flush=True)
            await asyncio.sleep(POLITE_PAUSE)
            continue

        memory.remember_page(url, page.content_hash)
        tour = build_tour(page)
        first = tour_to_objects(tour, ids, name)
        second = tour_to_listings(tour, ids)
        took = time.monotonic() - started

        if not first and not second:
            why = (f"answered {page.status_code}" if page.status_code
                   and page.status_code >= 400 else "no offer could be read")
            result["skipped"].append({"url": url, "why": why})
            print(f"        skipped ({why}) {took:.0f}s", flush=True)
            await asyncio.sleep(POLITE_PAUSE)
            continue

        result["first"].extend(first)
        result["second"].extend(second)
        result["kept"].append({
            "url": url,
            "tour_code": (first or second)[0].get("tour_code"),
            "first": len(first),
            "second": len(second),
        })
        print(f"        {len(first)} first, {len(second)} second, {took:.0f}s",
              flush=True)
        await asyncio.sleep(POLITE_PAUSE)

    if pages and len(result["kept"]) < pages:
        result["notes"].append(
            f"only {len(result['kept'])} of {pages} URLs sold anything, after"
            f" {len(result['kept']) + len(result['skipped'])} tries"
        )
    return result


def write_everything(results: list[dict], memory: RunMemory) -> None:
    """Both files, with every site inside them, plus a note on what was crawled.

    One file per structure and not one per site, because that is what the
    client is sent: their catalogue is the ten agencies together, and a reader
    who has to open ten files to see it cannot tell whether a tour is missing
    or in the file they have not opened yet.

    The sites are told apart inside the data, not by the filename —
    ``agency_name`` in both structures, and ``website_url`` in the first.
    """
    first = [obj for result in results for obj in result["first"]]
    second = [obj for result in results for obj in result["second"]]

    (OUT / "first_structure.json").write_text(
        json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "second_structure.json").write_text(
        json.dumps(listing_document(second), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # What the client does not already hold. The two files stay complete, so a
    # night they missed can still be caught up from them.
    changed = memory.to_send(first) + memory.to_send(second)

    # Where the API reads them. logs/ is emptied at the start of every run, so
    # a call at midnight would otherwise be served half a catalogue.
    publish(PACKAGES_FILE, first)
    publish(TOURS_FILE, second)
    memory.remember_sent(changed)
    publish(CHANGES_FILE, memory.tonight)

    codes_first = {obj.get("tour_code") for obj in first}
    codes_second = {obj.get("tour_code") for obj in second}
    lines = [
        "what the two files hold",
        f"crawled {_stamp()}",
        "",
        f"   first_structure.json    {len(first)} objects,"
        f" {len(codes_first)} tour codes",
        f"   second_structure.json   {len(second)} objects,"
        f" {len(codes_second)} tour codes",
        f"   codes in both           {len(codes_first & codes_second)}",
        f"   new or changed tonight  {len(memory.tonight)} records",
        "",
        f"{'site':18}{'tours':>7}{'skipped':>9}{'unread':>8}{'first':>8}{'second':>8}",
        "=" * 78,
    ]
    for result in results:
        lines.append(
            f"   {result['name']:15}{len(result['kept']):>7}"
            f"{len(result['skipped']):>9}{result.get('unchanged', 0):>8}"
            f"{len(result['first']):>8}{len(result['second']):>8}"
        )
    for result in results:
        if not (result["notes"] or result["skipped"]):
            continue
        lines += ["", f"{result['domain']}"]
        for note in result["notes"]:
            lines.append(f"   note  {note}")
        for entry in result["skipped"]:
            lines.append(f"   skipped  {entry['why']:<26} {entry['url']}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main(domain: Optional[str], pages: int) -> int:
    domains = load_domains()
    if domain:
        domains = [d for d in domains if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return 1

    clear_logs()
    wanted = f"{pages} pages each" if pages else "every page"
    print(f"cleared logs/, crawling {len(domains)} site(s), {wanted}")

    # One store for the whole run, read from disk and written back to it.
    ids = IdStore()
    # What last night saw, so tonight reads and sends only what moved.
    memory = RunMemory()

    started = time.monotonic()
    failed = 0
    results: list[dict] = []
    for index, site in enumerate(domains, start=1):
        print(f"\n[{index}/{len(domains)}] {site}", flush=True)
        try:
            result = await crawl_site(site, pages, ids, memory)
        except Exception as error:  # one site must not take the others with it
            failed += 1
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            continue
        results.append(result)
        # Written after every site, so a run stopped halfway still leaves both
        # files holding everything crawled up to that point.
        write_everything(results, memory)
        # Saved after every site for the same reason the files are: a run that
        # stops halfway must not mint these numbers again tomorrow.
        ids.save()
        memory.save()
        print(f"  -> {len(result['first'])} first, {len(result['second'])} second"
              f"  ({sum(len(r['first']) for r in results)} in the file)", flush=True)

    print(f"\ndone in {time.monotonic() - started:.0f}s, files in {OUT}")
    if failed:
        print(f"{failed} of {len(domains)} sites failed", flush=True)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help="tour pages per site; 0 for every page found")
    args = parser.parse_args()
    try:
        with only_one_run(), run_log():
            sys.exit(1 if asyncio.run(main(args.domain, args.pages)) else 0)
    except AlreadyRunning as reason:
        print(f"not starting: {reason}")
        sys.exit(1)
