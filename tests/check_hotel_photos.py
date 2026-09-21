# One real tour page per site, in and the hotel photograph out. The point is
# the eye-check: open a few of these addresses and see whether that really is
# the hotel. Each is fetched too, because a broken link reads as data we have.
#
#     .venv/bin/python tests/check_hotel_photos.py --site vanrosh

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.extraction.extractor import DEFAULT_HEADERS, _extract_embedded_json
from crawler.extraction.page_extract import extract_page_from_html
from crawler.extraction.tour_api import api_url_for, enrich_from_api, fetch_tour_json
from crawler.extraction.models import ExtractedPage
from crawler.schema.converter import build_tour
from crawler.schema.identifiers import IdStore
from crawler.schema.safarbank import tour_to_objects

# One live tour page per site, chosen because it sells several hotels. These go
# stale as the agencies take tours down; a 404 here means pick another, not
# that the reading broke.
PAGES: dict[str, str] = {
    "alefbatour": "https://www.alefbatour.com/tour/d/183385/",
    "avisatravel": "https://avisatravel.com/tour/d/97388/",
    "radingasht": "https://www.radingasht.com/tour/d/10890/",
    "nahalgasht": "https://nahalgasht.com/tour/63736-antalya-iranairtour/",
    "vanrosh": "https://vanrosh.com/tours/2606251012223/",
    "bisanseir": "https://bisanseir.com/tour/تور-پاتایا-ماهان-ایر-1405",
    "chakavaktravel": "https://chakavaktravel.com/tour/kuala-lumpur-tour",
}

TIMEOUT = 45.0


async def read_page(
    client: httpx.AsyncClient, url: str
) -> tuple[Optional[ExtractedPage], Optional[str]]:
    """One tour page, read the way the crawl reads it.

    The API is tried first for the same reason the crawl tries it: where a site
    publishes one, it states the hotels rather than drawing them.
    """
    response = await client.get(url)
    if response.status_code >= 400:
        return None, f"page answered {response.status_code}"
    html = response.text

    api = api_url_for(url, html)
    if api:
        data, error = await fetch_tour_json(client, api)
        if data:
            return enrich_from_api(ExtractedPage(url=url, html_raw=html), data), None
        return None, f"api: {error}"

    return extract_page_from_html(url, html, _extract_embedded_json(html)), None


async def check(client: httpx.AsyncClient, address: str) -> str:
    """Whether an address really serves a picture."""
    try:
        response = await client.head(address)
    except Exception as error:
        return type(error).__name__
    kind = response.headers.get("content-type", "").split(";")[0]
    if response.status_code != 200:
        return str(response.status_code)
    return "ok" if kind.startswith("image/") else kind or "not an image"


async def one_site(client: httpx.AsyncClient, name: str, url: str) -> tuple[int, int]:
    print(f"\n=== {name}\n    {url}")
    page, error = await read_page(client, url)
    if page is None:
        print(f"    could not read it: {error}")
        return 0, 0

    rows = tour_to_objects(build_tour(page), IdStore(path=None), name)
    hotels = [hotel for row in rows for hotel in (row.get("hotels") or [])]
    if not hotels:
        print("    no hotels read from this page")
        return 0, 0

    seen: dict[str, str] = {}
    for hotel in hotels:
        seen.setdefault(hotel.get("name") or "(unnamed)", hotel.get("logo") or "")

    addresses = [address for address in seen.values() if address]
    with_photo = len(addresses)
    shared = with_photo - len(set(addresses))
    print(f"    {len(rows)} packages, {len(seen)} hotels, {with_photo} with a photo")

    for hotel, address in seen.items():
        if not address:
            print(f"      {hotel[:30]:32} —")
            continue
        print(f"      {hotel[:30]:32} {await check(client, address):<14} {address}")

    if shared > 0 and with_photo:
        print(f"    WARNING: {shared} hotels share a photo with another hotel")
    return with_photo, len(seen)


async def main(site: Optional[str], url: Optional[str]) -> int:
    pages = {"(given)": url} if url else PAGES
    if site and not url:
        pages = {k: v for k, v in PAGES.items() if site in k}
        if not pages:
            print(f"no site matching {site!r}; known: {', '.join(PAGES)}")
            return 1

    found = total = 0
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers=DEFAULT_HEADERS
    ) as client:
        for name, page_url in pages.items():
            got, all_of_them = await one_site(client, name, page_url)
            found += got
            total += all_of_them

    print(f"\n{found} of {total} hotels came with a photograph")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hotel photographs, per site")
    parser.add_argument("--site", help="only this site (substring match)")
    parser.add_argument("--url", help="one page of your own instead of the list")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.site, args.url)))
