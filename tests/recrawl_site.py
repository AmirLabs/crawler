# Read one site's known URLs again and replace what the database holds for it.
# No discovery: the URLs come from the pages table, so a rule change can be
# tried against exactly what the last full crawl found, and nothing else.
#
# It writes. The site's tours and packages are replaced, a page that now reads
# as a menu has its records deleted and the tours it links to are queued, and
# the whole thing runs as one crawl so changed=true still means something.
#
#     .venv/bin/python tests/recrawl_site.py --site bisanseir
#     .venv/bin/python tests/recrawl_site.py --site bisanseir --workers 4
#     .venv/bin/python tests/recrawl_site.py --site nahalgasht --limit 100 --spread
#     .venv/bin/python tests/recrawl_site.py --site vanrosh --limit 20
#     .venv/bin/python tests/recrawl_site.py --site bisanseir --db /tmp/copy.db

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.extraction.page_extract import extract_page
from crawler.filtering.menu_pages import says_nothing, tour_links
from crawler.filtering.tour_groups import group_by_tour
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.schema.safarbank import tour_to_objects
from crawler.schema.safarbank_tours import tour_to_listings
from crawler.storage.database import DEFAULT_DATABASE_PATH, Database
from crawler.storage.tables import Page

TIMEOUT = 45.0
POLITE_PAUSE = 1.0
MENU_BUDGET = 300


def known_urls(db: Database, site: str) -> list[str]:
    return list(db.session.scalars(
        select(Page.url).where(Page.url.contains(site)).order_by(Page.url)
    ))


def _thinned(urls: list[str], wanted: int) -> list[str]:
    """``wanted`` URLs taken evenly across the list, in its own order.

    The pages table is sorted by address, so the first hundred of nahalgasht's
    1,767 are all one corner of the site. Stepping through it instead samples
    the whole of it, which is what a check of a rule needs.

    Example:
        >>> _thinned(["a", "b", "c", "d", "e", "f"], 3)
        ['a', 'c', 'e']
        >>> _thinned(["a", "b"], 5)
        ['a', 'b']
    """
    if wanted >= len(urls):
        return urls
    step = len(urls) / wanted
    return [urls[int(index * step)] for index in range(wanted)]


def site_name(url: str) -> str:
    return urlsplit(url).netloc.replace("www.", "").split(".")[0]


async def recrawl(site: str, path: Path, limit: Optional[int], workers: int,
                  spread: bool) -> int:
    db = Database(path)
    urls = known_urls(db, site)
    if not urls:
        print(f"no page in {path} matches {site!r}")
        db.close()
        return 1
    if limit:
        urls = _thinned(urls, limit) if spread else urls[:limit]

    print(f"{len(urls)} known URLs for {site!r} in {path}, {workers} at a time")
    profile = await profile_site(urlsplit(urls[0]).netloc, render=False)
    if not profile.reachable:
        print(f"unreachable: {profile.error}")
        db.close()
        return 1

    name = site_name(urls[0])
    db.start_run()
    writing = asyncio.Lock()
    queue: asyncio.Queue[str] = asyncio.Queue()
    for url in urls:
        queue.put_nowait(url)
    seen = set(urls)
    tally = {"read": 0, "kept": 0, "menus": 0, "failed": 0,
             "packages": 0, "tours": 0, "queued": 0}
    started = time.monotonic()

    async def work() -> None:
        while True:
            url = await queue.get()
            try:
                await one(url)
            finally:
                queue.task_done()

    async def one(url: str) -> None:
        mark = time.monotonic()
        try:
            page = await extract_page(url, profile, timeout=TIMEOUT)
        except Exception as error:
            tally["failed"] += 1
            print(f"      failed ({type(error).__name__}) {url[:70]}", flush=True)
            return
        if page.status_code and page.status_code >= 400:
            tally["failed"] += 1
            print(f"      answered {page.status_code} {url[:70]}", flush=True)
            return

        took = time.monotonic() - mark
        async with writing:
            tally["read"] += 1
            db.remember_page(url, page.content_hash)
            tour = build_tour(page)
            first = tour_to_objects(tour, db, name)
            second = tour_to_listings(tour, db)

            if says_nothing(first):
                db.store_tour(db.tour_code(url), [], [])
                added = 0
                if tally["queued"] < MENU_BUDGET:
                    for group in group_by_tour(tour_links(page.html_raw or "", url)):
                        if group.primary not in seen:
                            seen.add(group.primary)
                            queue.put_nowait(group.primary)
                            added += 1
                    tally["queued"] += added
                tally["menus"] += 1
                print(f"[{tally['read']}] menu, {added} queued, cleared "
                      f"({took:.0f}s) {url[:60]}", flush=True)
            else:
                db.store_tour(db.tour_code(url), first, second)
                tally["kept"] += 1
                tally["packages"] += len(first)
                tally["tours"] += len(second)
                dates = {record.get("start_date") for record in first}
                print(f"[{tally['read']}] {len(first)} packages, {len(second)} tours, "
                      f"{len(dates)} dates ({took:.0f}s) {url[:60]}", flush=True)
            db.save()
        await asyncio.sleep(POLITE_PAUSE)

    pool = [asyncio.create_task(work()) for _ in range(workers)]
    await queue.join()
    for task in pool:
        task.cancel()
    await asyncio.gather(*pool, return_exceptions=True)

    db.finish_run()
    db.save()
    db.close()
    print()
    for label, key in (("urls read", "read"), ("kept as tours", "kept"),
                       ("menus cleared", "menus"), ("queued from menus", "queued"),
                       ("failed", "failed"), ("packages stored", "packages"),
                       ("tours stored", "tours")):
        print(f"{label:22}{tally[key]}")
    print(f"{'took':22}{time.monotonic() - started:.0f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True,
                        help="substring of the URLs to read again, e.g. bisanseir")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--limit", type=int, help="stop after this many known URLs")
    parser.add_argument("--workers", type=int, default=1,
                        help="how many pages to read at once")
    parser.add_argument("--spread", action="store_true",
                        help="with --limit, sample across the site rather than its first URLs")
    args = parser.parse_args()
    return asyncio.run(recrawl(args.site, args.db, args.limit,
                               max(1, args.workers), args.spread))


if __name__ == "__main__":
    raise SystemExit(main())
