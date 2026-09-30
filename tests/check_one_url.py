# One tour URL in, the client's two structures out, through the whole pipeline.
# Nothing is mocked and nothing is stored: the numbers come from a database in
# memory, so a run here cannot spend a tour_code or touch data/crawler.db.
#
# It answers the question a crawl cannot answer quickly — did this page give us
# its other departure dates, and what did each one carry?
#
#     .venv/bin/python tests/check_one_url.py https://bisanseir.com/tour/تور-کیش-کاسپین
#     .venv/bin/python tests/check_one_url.py <url> --full
#     .venv/bin/python tests/check_one_url.py <url> --name kusadasi
#     .venv/bin/python tests/check_one_url.py <url> --dump
#
# Both structures are always written, one file each, under logs/check/<name>,
# so a change can be looked at rather than described:
#
#     logs/check/<name>/first_structure.json
#     logs/check/<name>/second_structure.json

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.extraction.page_extract import extract_page
from crawler.filtering.menu_pages import says_nothing, tour_links
from crawler.filtering.tour_groups import group_by_tour
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.schema.safarbank import tour_to_objects
from crawler.schema.safarbank_tours import tour_to_listings
from crawler.storage.database import Database

TIMEOUT = 45.0

CHECKS = Path(__file__).resolve().parent.parent / "logs" / "check"


def site_name(url: str) -> str:
    host = urlsplit(url).netloc
    return host.replace("www.", "").split(".")[0]


def folder_for(url: str, name: Optional[str]) -> Path:
    """Where this check writes, named after the tour unless a name was given.

    The query counts. lachinseir keeps the tour in it — tour.bc?lid=1&id=2361574
    — so a name taken from the path alone would write every one of its tours
    over the last.

    Example:
        >>> folder_for("https://x.com/tour/50082-izmir/", None).name
        '50082-izmir'
        >>> folder_for("https://x.com/tour.bc?lid=1&id=2361574", None).name
        'tour.bc-lid-1-id-2361574'
        >>> folder_for("https://x.com/tour/1", "mine").name
        'mine'
    """
    if not name:
        parsed = urlsplit(url)
        parts = [part for part in unquote(parsed.path).split("/") if part]
        name = (parts[-1] if parts else parsed.netloc)
        if parsed.query:
            name += "-" + unquote(parsed.query)
        name = name[:60]
    return CHECKS / re.sub(r"[^\w.\-]+", "-", name).strip("-")


def money(package: dict) -> str:
    rial = max([float(r.get("final_price") or 0) for r in package.get("rooms") or []] or [0])
    foreign = max([float(r.get("currency_price") or 0) for r in package.get("rooms") or []] or [0])
    if rial > 0:
        return f"{int(rial):,}"
    return f"{foreign:,.0f} ارز" if foreign > 0 else "—"


async def check(url: str, name: Optional[str], full: bool, dump: bool) -> int:
    profile = await profile_site(urlsplit(url).netloc, render=False)
    if not profile.reachable:
        print(f"unreachable: {profile.error}")
        return 1

    page = await extract_page(url, profile, timeout=TIMEOUT)
    print(f"status {page.status_code}   fetched by {page.fetch_method}"
          f"   rows {len(page.rows)}   embedded blobs {len(page.embedded_json)}"
          f"   calendar {len(page.calendar)}")
    for note in page.notes:
        print(f"   note  {note}")

    where = Counter()
    for row in page.rows:
        field = row.fields.get("departure_date")
        value = field.value if field else None
        raw = value.get("raw") if isinstance(value, dict) else value
        where[(raw, field.source_key if field else None)] += 1
    print()
    print("what each row says it leaves on, and where that came from:")
    for (raw, source), count in where.most_common(8):
        print(f"   {count:>4}  {str(raw):18} {source}")

    out = folder_for(url, name)
    out.mkdir(parents=True, exist_ok=True)
    if dump:
        (out / "page.html").write_text(page.html_raw or "", encoding="utf-8")
        (out / "embedded.json").write_text(
            json.dumps(page.embedded_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "rows.json").write_text(
            json.dumps([{key: field.value for key, field in row.fields.items()}
                        for row in page.rows], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n   dumped page.html, embedded.json and rows.json")

    db = Database(path=None)
    db.start_run()
    tour = build_tour(page)
    first = tour_to_objects(tour, db, site_name(url))
    second = tour_to_listings(tour, db)

    print()
    print(f"first structure   {len(first)} packages")
    print(f"second structure  {len(second)} tours")

    dates = Counter(record.get("start_date") or "(none)" for record in first)
    print()
    print(f"departure dates in the first structure: {len(dates)}")
    for date, count in sorted(dates.items()):
        print(f"   {date:12} {count:>4} packages")

    print()
    print(f"{'start_date':14}{'hotel':34}{'star':>5}{'price':>16}")
    print("=" * 70)
    for package in first if full else first[:25]:
        hotels = package.get("hotels") or []
        name = (hotels[0].get("name") if hotels else "") or "—"
        star = (hotels[0].get("star_code") if hotels else "") or "—"
        print(f"{package.get('start_date') or '—':14}{name[:33]:34}{star:>5}{money(package):>16}")
    if not full and len(first) > 25:
        print(f"   ... {len(first) - 25} more, pass --full to see them")

    if says_nothing(first):
        links = group_by_tour(tour_links(page.html_raw or "", url))
        print()
        print(f"LABELLED: nothing was read from this page. "
              f"It links to {len(links)} tours:")
        for group in links[:10]:
            print(f"   {group.primary}")

    for stem, records in (("first_structure", first), ("second_structure", second)):
        (out / f"{stem}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"written to {out}")
    print(f"   first_structure.json    {len(first)} packages")
    print(f"   second_structure.json   {len(second)} tours")
    db.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="one tour page")
    parser.add_argument("--name", help="folder name under logs/check, else the tour slug")
    parser.add_argument("--full", action="store_true", help="print every package")
    parser.add_argument("--dump", action="store_true",
                        help="also write the page, its embedded JSON and its rows")
    args = parser.parse_args()
    return asyncio.run(check(args.url, args.name, args.full, args.dump))


if __name__ == "__main__":
    raise SystemExit(main())
