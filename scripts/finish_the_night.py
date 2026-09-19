"""Crawl the sites a stopped run never reached, and add them to what is served.

The run of 2026-09-15 was interrupted after seven of the ten sites. Re-running
the nightly script would not have finished the job: a page that comes back
unchanged is skipped, contributes no rows, and so falls out of the file that
:mod:`crawler.api.store` publishes — which is built from the rows of that run
alone. The seven sites already crawled would have been replaced by the handful
of tours whose pages had moved overnight.

So this crawls only the sites named, and *merges* what it finds into what is
already published rather than replacing it. The live catalogue never shrinks:
each site is merged and published as it finishes, so an interruption here costs
one site, not the seven that were already paid for.

This is a one-off. The bug it works around is real and still there; the nightly
run needs the fix, not this script.

Run::

    .venv/bin/python scripts/finish_the_night.py
    .venv/bin/python scripts/finish_the_night.py chakavaktravel
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.api.store import CHANGES_FILE, PACKAGES_FILE, TOURS_FILE, publish, read
from crawler.automation.memory import RunMemory, record_key
from crawler.automation.scheduler import AlreadyRunning, only_one_run, run_log
from crawler.schema.identifiers import IdStore
from crawler.utils.domains import load_domains
from scripts.run_client_output import crawl_site

# The three the interrupted run never opened, in the order targets.txt lists.
REMAINING = ("chakavaktravel", "lachinseir", "radingasht")

EVERY_PAGE = 0


def merged(held: list[dict], fresh: list[dict]) -> list[dict]:
    """Both sets of records, with a fresh one winning wherever the ids meet.

    Keyed on :func:`crawler.automation.memory.record_key`, the same id the
    client checks a record against, so a tour cannot arrive twice under two
    spellings of the same identity.

    Example:
        >>> held = [{"api_id": "a", "price": 10}, {"api_id": "b", "price": 20}]
        >>> merged(held, [{"api_id": "b", "price": 25}, {"api_id": "c"}])
        [{'api_id': 'a', 'price': 10}, {'api_id': 'b', 'price': 25}, {'api_id': 'c'}]
        >>> merged(held, []) == held
        True
    """
    by_key = {record_key(record): record for record in held}
    for record in fresh:
        by_key[record_key(record)] = record
    return list(by_key.values())


def wanted_domains(asked: list[str]) -> list[str]:
    """The target URLs matching what was asked for, in the file's own order.

    Example:
        >>> wanted_domains(["nothing-like-this"])
        []
    """
    names = asked or list(REMAINING)
    return [site for site in load_domains()
            if any(name in site for name in names)]


async def main(asked: list[str]) -> int:
    """Crawl each site asked for and publish it beside what is already served."""
    sites = wanted_domains(asked)
    if not sites:
        print(f"no site in domains/targets.txt matches {asked}")
        return 1

    ids = IdStore()
    memory = RunMemory()

    held_first = read(PACKAGES_FILE)
    held_second = read(TOURS_FILE)
    held_changes = read(CHANGES_FILE)
    print(f"holding {len(held_second)} tours and {len(held_first)} packages")
    print(f"crawling {len(sites)} site(s): {', '.join(sites)}", flush=True)

    fresh_first: list[dict] = []
    fresh_second: list[dict] = []
    failed = 0

    for index, site in enumerate(sites, start=1):
        print(f"\n[{index}/{len(sites)}] {site}", flush=True)
        try:
            result = await crawl_site(site, EVERY_PAGE, ids, memory)
        except Exception as error:  # one site must not take the others with it
            failed += 1
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            continue

        fresh_first.extend(result["first"])
        fresh_second.extend(result["second"])

        first = merged(held_first, fresh_first)
        second = merged(held_second, fresh_second)
        memory.remember_sent(memory.to_send(result["first"] + result["second"]))

        publish(PACKAGES_FILE, first)
        publish(TOURS_FILE, second)
        publish(CHANGES_FILE, sorted(set(held_changes) | set(memory.tonight)))
        ids.save()
        memory.save()

        print(f"  -> {len(result['first'])} first, {len(result['second'])} second"
              f"  (now {len(second)} tours, {len(first)} packages)", flush=True)

    print(f"\ndone. {len(merged(held_second, fresh_second))} tours published")
    if failed:
        print(f"{failed} of {len(sites)} sites failed", flush=True)
    return failed


if __name__ == "__main__":
    try:
        with only_one_run(), run_log() as log:
            print(f"finishing the night, logging to {log}", flush=True)
            sys.exit(1 if asyncio.run(main(sys.argv[1:])) else 0)
    except AlreadyRunning as reason:
        print(f"not starting: {reason}")
        sys.exit(1)
