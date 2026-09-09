"""Who each agency is, written down once instead of read off every page.

The client's ``agency_info`` block wants five things — the name, the address,
the landline, the mobile and the logo. They are one value per agency, not one
per tour, and reading them out of markup was measured to be both harder and
worse than writing them down:

* A footer holds three numbers — sales, support, after hours — and nothing in
  the markup says which one the client should ring. A rule would take the first
  and be quietly wrong.
* Only 3 of the 10 sites print an address at all on a tour page. The rest keep
  it on their contact page, which is a second page shape to fetch and parse for
  ten rows of data.
* ``og:site_name`` gives bisanseir's name as "بیـــســـان ســـیر", decorated
  with tatweel, and lachinseir publishes no name at all.

So the ten rows were read by hand from each agency's own contact page on
1405-06-18 and live in ``data/agencies.json``. This is data, not a per-site
rule: no extraction behaviour is keyed on a domain, and ``domains/targets.txt``
is per-site for the same reason.

What the file says wins, because a person checked it. What it leaves empty
falls back to what the page gave, so an agency missing from the file still
comes out as well as it did before. ``checked_on`` says when a human last
looked; it is ours and is never sent.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse

from crawler.config import PROJECT_ROOT

AGENCIES_PATH = PROJECT_ROOT / "data" / "agencies.json"

# The five keys the client's agency_info block carries, in their order.
INFO_KEYS: tuple[str, ...] = ("AgencyName", "Address", "Phone", "Mobile", "Logo")


def host_of(url: str) -> str:
    """The site a URL belongs to, as the file keys it.

    Example:
        >>> host_of("https://www.tahagasht.com/tours/item/x")
        'tahagasht.com'
        >>> host_of("not a url")
        ''
    """
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


@lru_cache(maxsize=1)
def _table() -> dict[str, dict]:
    """The file, read once. A missing file is empty, never an error."""
    try:
        loaded = json.loads(AGENCIES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def known_agency(url: str) -> dict:
    """What the file says about the site this URL is on, or nothing.

    Example:
        >>> known_agency("https://bisanseir.com/tour/x")["AgencyName"]
        'بیسان سیر'
        >>> known_agency("https://example.com/tour/x")
        {}
    """
    return _table().get(host_of(url), {})


def agency_info(
    url: str, name: Optional[str] = None, logo: Optional[str] = None
) -> dict:
    """The client's ``agency_info`` block for one tour's site.

    ``name`` and ``logo`` are what the page itself gave. They fill only the
    keys the file leaves empty, so a checked value is never overwritten by a
    read one.

    Example:
        >>> info = agency_info("https://vanrosh.com/tours/1")
        >>> info["AgencyName"], info["Phone"], info["Mobile"]
        ('روشنی گشت', '02151713', '09120913869')
        >>> agency_info("https://example.com/t", name="مثال سیر")
        {'AgencyName': 'مثال سیر', 'Address': '', 'Phone': '', 'Mobile': '', 'Logo': ''}
    """
    known = known_agency(url)
    read = {"AgencyName": name or "", "Logo": logo or ""}
    return {key: (known.get(key) or read.get(key, "") or "") for key in INFO_KEYS}


def agency_name(url: str, name: Optional[str] = None) -> str:
    """The agency's name, checked where we have it and read where we do not.

    Example:
        >>> agency_name("https://bisanseir.com/tour/x", "بیـــســـان ســـیر")
        'بیسان سیر'
        >>> agency_name("https://example.com/t", "مثال سیر")
        'مثال سیر'
    """
    return known_agency(url).get("AgencyName") or (name or "")
