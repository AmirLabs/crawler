"""Crawl until both kinds of tour are found, and write the client's JSON.

This is the deliverable in the shape they asked for: the same 62 keys as
``data/safarbank.json``, in their order and their types. It exists to be sent
to them and compared against their own file.

A tour is one of two kinds, and the client's structure treats them differently,
so one file carries both rather than leaving either to be imagined:

    one hotel     one destination, one hotel, one price — the common case
    many hotels   a route that sleeps in several cities under one price, each
                  hotel with its own city, grade, board and night count

The script keeps opening tours until it has as many of each kind as asked for,
so there is no hunting for a suitable URL by hand. A site that has no
multi-hotel tour says so instead of running forever.

One file per site, because a site that fails should not take the others with
it. A companion ``.txt`` names which tour_code is which kind, so the JSON
itself stays exactly the 62 keys and nothing else.

Run::

    .venv/bin/python scripts/run_json_sample.py
    .venv/bin/python scripts/run_json_sample.py --domain alefbatour --each 2

Files land in ``logs/json/<site>.json``. Codes come from ``data/crawler.db``,
the same ones the nightly run uses, so a tour keeps its ``tour_code``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.config import PROJECT_ROOT
from crawler.extraction.page_extract import extract_page
from crawler.filtering.url_filter import discover_tour_urls
from crawler.profiler.profiler import profile_site
from crawler.schema.converter import build_tour
from crawler.storage.database import Database
from crawler.schema.safarbank import tour_to_objects
from crawler.utils.domains import load_domains

OUT = PROJECT_ROOT / "logs" / "json"

DEFAULT_EACH = 1
DEFAULT_SITES = ("alefbatour", "avisatravel")

# How many tours may be opened per kind wanted before giving up on finding one.
# alefbatour sells almost no multi-city tours, so a small budget would report a
# site that has none when it has a few.
ATTEMPTS_PER_KIND = 25

ONE_HOTEL = "one hotel"
MANY_HOTELS = "many hotels"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def site_name(domain: str) -> str:
    """The name a site's file is written under.

    Example:
        >>> site_name("https://www.alefbatour.com")
        'alefbatour'
    """
    host = domain.split("//")[-1].split("/")[0]
    return host.replace("www.", "").split(".")[0]


def _spread(urls: list[str], wanted: int) -> list[str]:
    """The list reordered so the first tries are spread across it.

    Example:
        >>> _spread(["a", "b", "c", "d"], 2)
        ['a', 'c', 'b', 'd']
    """
    step = max(1, len(urls) // max(wanted, 1))
    first = urls[::step]
    return first + [url for url in urls if url not in set(first)]


def kind_of(objects: list[dict]) -> str:
    """Which of the two kinds a tour's objects are.

    Example:
        >>> kind_of([{"hotels": [{"name": "a"}, {"name": "b"}]}])
        'many hotels'
        >>> kind_of([{"hotels": [{"name": "a"}]}])
        'one hotel'
    """
    if any(len(o.get("hotels") or []) > 1 for o in objects):
        return MANY_HOTELS
    return ONE_HOTEL


async def objects_for(
    domain: str, each: int, ids: Database
) -> tuple[list[dict], list[dict]]:
    """Keep opening tours until both kinds are found, or the budget runs out.

    Returns the objects and, beside them, a note per tour saying which kind it
    was — so the JSON can stay exactly the keys the client sends.
    """
    name = site_name(domain)
    print(f"\n=== {name}", flush=True)

    profile = await profile_site(domain, render=False)
    if not profile.reachable:
        print(f"  UNREACHABLE: {profile.error}", flush=True)
        return [], []

    print("  finding tour URLs ...", flush=True)
    found = await discover_tour_urls(profile)
    print(f"  {len(found.tour_urls)} URLs", flush=True)

    out: list[dict] = []
    found_kinds: list[dict] = []
    counts = {ONE_HOTEL: 0, MANY_HOTELS: 0}
    budget = each * 2 * ATTEMPTS_PER_KIND

    for url in _spread(found.tour_urls, budget)[:budget]:
        if all(counts[k] >= each for k in counts):
            break
        started = time.monotonic()
        try:
            page = await extract_page(url, profile, timeout=40)
        except Exception as error:
            print(f"    skipped ({type(error).__name__})  {url[:56]}", flush=True)
            continue
        if not page.rows:
            continue

        objects = tour_to_objects(build_tour(page), ids, name)
        if not objects:
            continue
        kind = kind_of(objects)
        if counts[kind] >= each:
            continue

        counts[kind] += 1
        out.extend(objects)
        found_kinds.append({
            "kind": kind,
            "tour_code": objects[0]["tour_code"],
            "objects": len(objects),
            "hotels": max(len(o["hotels"]) for o in objects),
            "url": url,
        })
        print(f"    {kind:<12} {len(objects):>4} objects,"
              f" {time.monotonic() - started:.0f}s  {url[:48]}", flush=True)
        await asyncio.sleep(1.0)

    for kind, wanted in counts.items():
        if wanted < each:
            print(f"    only {wanted} of {each} tours with {kind!r} found", flush=True)
    return out, found_kinds


def write_site(name: str, objects: list[dict], kinds: list[dict]) -> Path:
    """The site's JSON, and a plain note saying which tour_code is which kind."""
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"{name} — {len(objects)} objects, {len(kinds)} tours", ""]
    for entry in kinds:
        lines += [
            f"[{entry['kind']}]  tour_code {entry['tour_code']}",
            f"    {entry['objects']} objects, up to {entry['hotels']} hotels each",
            f"    {entry['url']}",
            "",
        ]
    path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    return path


async def main(domain: Optional[str], each: int) -> None:
    domains = [d for d in load_domains() if any(s in d for s in DEFAULT_SITES)]
    if domain:
        domains = [d for d in load_domains() if domain in d]
        if not domains:
            print(f"no domain matching {domain!r} in domains/targets.txt")
            return

    OUT.mkdir(parents=True, exist_ok=True)
    ids = Database()
    for target in domains:
        objects, kinds = await objects_for(target, each, ids)
        path = write_site(site_name(target), objects, kinds)
        print(f"  wrote {len(objects)} objects to {path}", flush=True)
    ids.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="only this site (substring match)")
    parser.add_argument("--each", type=int, default=DEFAULT_EACH,
                        help=f"tours of each kind per site (default {DEFAULT_EACH})")
    args = parser.parse_args()
    asyncio.run(main(args.domain, args.each))
