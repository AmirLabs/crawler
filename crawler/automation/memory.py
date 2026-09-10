"""What last night's run saw, so tonight's does only what changed.

The client asks for the tours that are new to them, not the catalogue again,
and nothing in a crawl remembers anything. This is that memory, and it answers
two different questions with two small maps:

``pages``
    the hash of each page's markup when we last read it. A page that comes back
    identical is not extracted again — measured at 11.5 seconds a page against
    the 1 to 3 seconds the fetch itself costs, which is where a night goes.

``rows``
    the hash of each record when we last sent it, under the id the client keys
    on. A record that is byte-for-byte what they already hold is not sent
    again; the same id with a different hash is a price that moved, and that
    is exactly what they asked to receive.

The page is still fetched every night, on every site. There is no way to know
what changed without looking, which is the one thing their developer was right
about when he asked why a cache was needed at all.

Losing this file costs a night of extra work and nothing else: an empty memory
makes tonight look like the first night, which is correct, only slower. That is
why it is not written as carefully as ``data/ids.json``, which cannot be
rebuilt at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from crawler.config import DATA_DIR

DEFAULT_MEMORY_PATH = DATA_DIR / "seen.json"

# Said on a page that came back exactly as it was, so a caller can tell it from
# a page that was read and had nothing to give.
UNCHANGED_NOTE = "unchanged since the last run"

# Fields that carry a different number every run whatever the tour did.
# ``id`` in a package is a running count of output rows, so leaving it in would
# make every package look changed on every night.
_NOT_THE_CONTENT = ("id",)


def _digest(text: str) -> str:
    """A short, stable fingerprint of some text.

    Example:
        >>> _digest("تور آنتالیا")
        '22b2f331e19f'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def record_key(record: dict) -> str:
    """The id a record is remembered under, whichever structure it belongs to.

    A tour carries the client's own ``api_id``. A package has no such field, so
    it is keyed on what actually identifies one: its tour, its departure and
    its number within that departure.

    Example:
        >>> record_key({"api_id": "31540001-1-14050621"})
        '31540001-1-14050621'
        >>> record_key({"tour_code": "31540001", "start_date": "14050621",
        ...             "number_package": "2"})
        '31540001|14050621|2'
    """
    if record.get("api_id"):
        return str(record["api_id"])
    return "|".join(
        str(record.get(field, "")) for field in
        ("tour_code", "start_date", "number_package")
    )


def record_hash(record: dict) -> str:
    """A fingerprint of what a record says, ignoring the fields that always move.

    Example:
        >>> one = {"api_id": "1", "id": "7", "price": 39000000}
        >>> two = {"api_id": "1", "id": "412", "price": 39000000}
        >>> record_hash(one) == record_hash(two)
        True
        >>> record_hash(one) == record_hash({"api_id": "1", "price": 45000000})
        False
    """
    kept = {key: value for key, value in record.items()
            if key not in _NOT_THE_CONTENT}
    return _digest(json.dumps(kept, sort_keys=True, ensure_ascii=False))


class RunMemory:
    """The two maps, read at the start of a run and written back at the end.

    Example:
        >>> memory = RunMemory(path=None)
        >>> memory.unchanged("https://x.com/tour/1", "abc")
        False
        >>> memory.remember_page("https://x.com/tour/1", "abc")
        >>> memory.unchanged("https://x.com/tour/1", "abc")
        True
        >>> memory.unchanged("https://x.com/tour/1", "def")
        False
    """

    def __init__(self, path: Optional[Path] = DEFAULT_MEMORY_PATH) -> None:
        self.path = path
        # Every key sent during this run. The delivery is written after each
        # site, and by then the earlier sites are already remembered, so asking
        # to_send again would report only the last site's changes.
        self.tonight: list[str] = []
        self._data: dict = {"pages": {}, "rows": {}}
        if path is not None and path.exists():
            try:
                self._data.update(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass  # a damaged memory is an empty one, never a stopped run

    def unchanged(self, url: str, content_hash: Optional[str]) -> bool:
        """Whether this page is exactly what it was the last time we read it.

        A page we have never seen is not unchanged, and neither is one whose
        hash the fetch layer could not produce.
        """
        if not content_hash:
            return False
        return self._data["pages"].get(url) == content_hash

    def remember_page(self, url: str, content_hash: Optional[str]) -> None:
        """Note what this page looked like tonight."""
        if content_hash:
            self._data["pages"][url] = content_hash

    def to_send(self, records: list[dict]) -> list[dict]:
        """The records the client does not already hold, unchanged ones dropped.

        Example:
            >>> memory = RunMemory(path=None)
            >>> rows = [{"api_id": "1", "price": 39}, {"api_id": "2", "price": 52}]
            >>> len(memory.to_send(rows))
            2
            >>> memory.remember_sent(rows)
            >>> memory.to_send(rows)
            []
            >>> memory.to_send([{"api_id": "1", "price": 45}])
            [{'api_id': '1', 'price': 45}]
        """
        known = self._data["rows"]
        return [
            record for record in records
            if known.get(record_key(record)) != record_hash(record)
        ]

    def remember_sent(self, records: list[dict]) -> None:
        """Note what went out, once it has actually gone.

        Kept apart from :meth:`to_send` so a delivery that failed is not
        remembered as one that succeeded.

        Example:
            >>> memory = RunMemory(path=None)
            >>> memory.remember_sent([{"api_id": "1"}])
            >>> memory.remember_sent([{"api_id": "2"}])
            >>> memory.tonight
            ['1', '2']
        """
        for record in records:
            key = record_key(record)
            self._data["rows"][key] = record_hash(record)
            if key not in self.tonight:
                self.tonight.append(key)

    def save(self) -> None:
        """Write the memory beside the last one, replacing it atomically."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writing = self.path.with_suffix(".writing.json")
        writing.write_text(
            json.dumps(self._data, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(writing, self.path)
