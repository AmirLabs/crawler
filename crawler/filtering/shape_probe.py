"""Asking a site which of its URL shapes actually hold a tour.

Step 2 keeps every URL shape that scores near the best one, and it has to:
tahagasht.com sells its departures under ``/tours/item/<slug>`` while its
countries sit at ``/tours/<slug>``, and dropping the second-place shape lost
325 real offers. The cost is that the guides come along too — 878 of
tahagasht's 1219 kept URLs are countries, and 81 of alefbatour's 596 are
``/tour/bodrum``-style guides.

Nothing in the URL text separates the two. Measured across the sites::

    madrid-barcelona    2 tokens   a real tour     liyaseir.com
    south-korea         2 tokens   a guide         lachinseir.com
    kusadasi            1 token    a real tour     nahalgasht.com
    dubai               1 token    a guide         radingasht.com

The same shape means opposite things on two sites: ``/tours/<slug>`` is the
tour on vanrosh.com and the guide on tahagasht.com. So the site is asked rather
than guessed at — a few URLs of each shape are run through step 3, and a shape
is kept when one of them yields a row.

Reading step 3's rows rather than searching the markup for a price is what
distinguishes a guide that advertises its tours from a tour. Measured on five
pages, the price text was wrong twice and the rows were never wrong::

    tahagasht  /tours/Greece      price text: tour   rows: 0    guide
    tahagasht  /tours/armenia     price text: tour   rows: 0    guide
    tahagasht  /tours/item/<slug> price text: guide  rows: 5    tour
    alefbatour /tour/bodrum       price text: guide  rows: 0    guide
    alefbatour /tour/d/<num>/…    price text: tour   rows: 152  tour

That test was only available once step 3 could read the sites it used to come
back empty on; before that it would have shut alefbatour off entirely.

The two mistakes do not cost the same. Keeping a guide wastes a fetch; dropping
a tour shape loses every tour behind it. So one row is enough to keep a shape,
and a shape nothing could be read from is kept rather than dropped.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from crawler.extraction.page_extract import extract_page
from crawler.filtering.models import UrlDiscovery
from crawler.filtering.url_filter import group_by_shape
from crawler.profiler.models import SiteProfile, UrlShapeInfo

SAMPLES_PER_SHAPE = 3
PROBE_DELAY_SECONDS = 0.8

# A probe reads one page; following its other departures would fetch twenty.
PROBE_FOLLOW_DATES = False


def shape_of(url: str, shapes: dict[str, list[str]]) -> Optional[str]:
    """The shape ``url`` was grouped under, or ``None`` if it was not grouped.

    Example:
        >>> shapes = {"/tour/<slug>": ["https://x.com/tour/bali"]}
        >>> shape_of("https://x.com/tour/bali", shapes)
        '/tour/<slug>'
        >>> shape_of("https://x.com/blog/1", shapes) is None
        True
    """
    for shape, urls in shapes.items():
        if url in urls:
            return shape
    return None


async def prove_shapes(
    profile: SiteProfile,
    urls: list[str],
    samples_per_shape: int = SAMPLES_PER_SHAPE,
) -> UrlShapeInfo:
    """Run a few URLs of each shape through step 3 and report which sell tours.

    Returns ``tested=False`` when no sample could be read at all, which is the
    caller's signal to keep every URL rather than none.

    Example:
        >>> import asyncio
        >>> asyncio.run(prove_shapes(SiteProfile(domain="x.com"), []))
        UrlShapeInfo(tour_shapes=[], guide_shapes=[], tested=False, notes=['no URLs to probe'])
    """
    info = UrlShapeInfo()
    if not urls:
        info.notes.append("no URLs to probe")
        return info

    shapes = group_by_shape(sorted(urls))
    for shape, members in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        sells = False
        read = 0
        for url in members[:samples_per_shape]:
            try:
                page = await extract_page(
                    url, profile, follow_dates=PROBE_FOLLOW_DATES
                )
            except Exception as error:
                info.notes.append(f"{shape}: {type(error).__name__} on a sample")
                continue
            if page.status_code and page.status_code >= 400:
                info.notes.append(f"{shape} <- {url} answered {page.status_code}")
                continue
            read += 1
            info.notes.append(
                f"{shape} <- {url} gave {len(page.rows)} rows"
                f" and {len(page.expired_rows)} departed"
            )
            if page.rows or page.expired_rows:
                sells = True
                break
            await asyncio.sleep(PROBE_DELAY_SECONDS)

        if read == 0:
            info.notes.append(f"{shape}: no sample could be read")
            continue
        info.tested = True
        (info.tour_shapes if sells else info.guide_shapes).append(shape)

    if not info.tested:
        info.notes.append("no shape could be read; keeping every URL")
    elif not info.tour_shapes:
        info.notes.append("no shape yielded a row")
    return info


async def keep_proven_shapes(
    profile: SiteProfile, found: UrlDiscovery
) -> UrlDiscovery:
    """Narrow a discovery to the shapes that were shown to hold tours.

    The result is written to ``profile.url_shapes`` as well as applied, so a
    report can show why URLs went missing and a cached profile can one day skip
    the probe.

    Nothing is dropped unless a shape was read and yielded nothing: losing a
    site to a slow afternoon is worse than fetching some guides.

    Example:
        >>> import asyncio
        >>> from crawler.filtering.models import UrlDiscovery
        >>> found = UrlDiscovery(domain="x.com", source="sitemap")
        >>> asyncio.run(keep_proven_shapes(SiteProfile(domain="x.com"), found)).tour_urls
        []
    """
    info = await prove_shapes(profile, found.tour_urls)
    profile.url_shapes = info

    if not info.tour_shapes:
        found.notes.extend(info.notes)
        return found

    shapes = group_by_shape(sorted(found.tour_urls))
    kept = [
        url for url in found.tour_urls
        if shape_of(url, shapes) in info.tour_shapes
    ]
    dropped = len(found.tour_urls) - len(kept)
    found.tour_urls = kept
    found.total_kept = len(kept)
    if dropped:
        found.notes.append(
            f"dropped {dropped} URLs whose shape yielded no row: "
            + ", ".join(info.guide_shapes[:3])
        )
    return found
