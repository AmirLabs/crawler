"""Numbers that must mean the same thing tomorrow as they do tonight.

The client's JSON is joined on its ids. ``tour_code`` is how they ask for every
departure of one tour, so a tour that changes its code overnight becomes a
different tour to them, and the code it used to hold now points somewhere else
— a wrong answer that reads like a right one.

Nothing in a crawl remembers anything, so the map from a tour to its number is
kept on disk. It cannot be rebuilt if it is lost: there is no way to work out
which tour held ``31540001`` yesterday. That is why it is written through a
temporary file and renamed, and why a dated copy is kept beside it.

Two values are fixed by the client rather than chosen here: every ``tour_code``
begins ``3154``, and ``provider_id`` is always ``110110110``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from crawler.config import DATA_DIR

TOUR_CODE_PREFIX = "3154"
TOUR_CODE_DIGITS = 4
PROVIDER_ID = 110110110

DEFAULT_STORE_PATH = DATA_DIR / "ids.json"

# How many dated copies to keep beside the live file.
BACKUPS_KEPT = 14


# A query value that is a date, in either calendar. Only these are dropped from
# a tour's identity: everything else in the query is part of which tour it is.
_A_DATE = re.compile(r"^\s*[12]\d{3}[-/]\d{1,2}[-/]\d{1,2}\s*$")


def tour_key(page_url: str) -> str:
    """The identity a tour keeps across days: its page, minus the date asked for.

    A date link asks the same page for another departure — bisanseir.com's
    ``?date=1405-06-11`` is one tour, not nine — so that parameter is dropped
    and those departures share a tour_code.

    The rest of the query stays. Dropping all of it put ten lachinseir.com
    tours under one code: its pages are ``tour.bc?lid=1&id=2289585``, so the
    query is the whole of what tells one tour from another, and the client
    dedupes on the code we send.

    Example:
        >>> tour_key("https://bisanseir.com/tour/tbilisi?date=1405-06-11")
        'https://bisanseir.com/tour/tbilisi'
        >>> tour_key("https://lachinseir.com/tour.bc?lid=1&id=2289585")
        'https://lachinseir.com/tour.bc?id=2289585&lid=1'
    """
    parsed = urlparse(page_url)
    kept = sorted(
        (key, value) for key, value in parse_qsl(parsed.query)
        if not _A_DATE.match(value)
    )
    return urlunparse(parsed._replace(query=urlencode(kept), fragment=""))


def hotel_key(site: str, name: str) -> str:
    """The identity a hotel keeps across days, within one site.

    Kept per site rather than shared: matching "هتل بایوک" on one site to
    "Baiyoke Sky Hotel" on another is guesswork, and Amir's decision was that
    each site numbers its own.

    Example:
        >>> hotel_key("tahagasht", " Baiyoke  Sky Hotel ")
        'tahagasht|baiyoke sky hotel'
    """
    return f"{site}|{' '.join(name.split()).lower()}"


class IdStore:
    """The numbers already handed out, and the next one to hand out.

    Example:
        >>> store = IdStore(path=None)
        >>> store.tour_code("https://x.com/tour/1")
        '31540001'
        >>> store.tour_code("https://x.com/tour/1")
        '31540001'
        >>> store.tour_code("https://x.com/tour/2")
        '31540002'
    """

    def __init__(self, path: Optional[Path] = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self._data: dict = {"tour_code": {}, "hotel_id": {}, "fk_tour_id": {},
                            "row_id": {},
                            "next": {"tour": 1, "hotel": 1, "fk": 1, "row": 1}}
        if path is not None and path.exists():
            self._data.update(json.loads(path.read_text(encoding="utf-8")))
            self._data.setdefault("fk_tour_id", {})
            self._data.setdefault("row_id", {})
            self._data["next"].setdefault("fk", 1)

    def _take(self, counter: str) -> int:
        number = self._data["next"][counter]
        self._data["next"][counter] = number + 1
        return number

    def tour_code(self, page_url: str) -> str:
        """The code for this tour, minted once and reused for ever after.

        Example:
            >>> IdStore(path=None).tour_code("https://x.com/tour/1?date=1405-06-11")
            '31540001'
        """
        key = tour_key(page_url)
        known = self._data["tour_code"].get(key)
        if known is None:
            known = f"{TOUR_CODE_PREFIX}{self._take('tour'):0{TOUR_CODE_DIGITS}d}"
            self._data["tour_code"][key] = known
        return known

    def fk_tour_id(self, tour_code: str, start_date: str) -> str:
        """The id of one tour on one departure date.

        Example:
            >>> store = IdStore(path=None)
            >>> store.fk_tour_id("31540001", "14050604")
            '1'
            >>> store.fk_tour_id("31540001", "14050605")
            '2'
            >>> store.fk_tour_id("31540001", "14050604")
            '1'
        """
        key = f"{tour_code}|{start_date}"
        known = self._data["fk_tour_id"].get(key)
        if known is None:
            known = str(self._take("fk"))
            self._data["fk_tour_id"][key] = known
        return known

    def hotel_id(self, site: str, name: Optional[str]) -> str:
        """The id of one hotel on one site, or ``"0"`` where it has no name.

        Example:
            >>> store = IdStore(path=None)
            >>> store.hotel_id("tahagasht", "Baiyoke Sky Hotel")
            '1'
            >>> store.hotel_id("tahagasht", "baiyoke sky hotel")
            '1'
            >>> store.hotel_id("tahagasht", None)
            '0'
        """
        if not name or not name.strip():
            return "0"
        key = hotel_key(site, name)
        known = self._data["hotel_id"].get(key)
        if known is None:
            known = str(self._take("hotel"))
            self._data["hotel_id"][key] = known
        return known

    def row_id(self, key: str) -> str:
        """The id of one row of output, kept for the same row on every night.

        A number that moved each night changed the record's hash, so an
        unchanged tour was sent under ``changed=true`` again.

        Example:
            >>> store = IdStore(path=None)
            >>> store.row_id("leg|1|1"), store.row_id("leg|1|2"), store.row_id("leg|1|1")
            ('1', '2', '1')
        """
        known = self._data["row_id"].get(key)
        if known is None:
            known = str(self._take("row"))
            self._data["row_id"][key] = known
        return known

    def save(self) -> None:
        """Write the store, keeping the old copy readable throughout.

        A crash halfway through a plain write leaves a truncated file, and a
        truncated id store is the same as no id store. Writing beside it and
        renaming makes the swap atomic.
        """
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            backup = self.path.with_suffix(f".{date.today().isoformat()}.json")
            shutil.copy2(self.path, backup)
            self._prune_backups()

        temporary = self.path.with_suffix(".writing.json")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def _prune_backups(self) -> None:
        copies = sorted(self.path.parent.glob(f"{self.path.stem}.20*.json"))
        for old in copies[:-BACKUPS_KEPT]:
            old.unlink()
