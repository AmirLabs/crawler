"""How a record is told apart from another, and from itself last night.

A page that comes back identical is not extracted again — measured at 11.5
seconds a page against the 1 to 3 seconds the fetch itself costs, which is
where a night goes. A record whose hash has not moved is not sent again under
``changed=true``. Both are kept in :mod:`crawler.storage.database`; this is
only how the keys and hashes are made.
"""

from __future__ import annotations

import hashlib
import json

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
