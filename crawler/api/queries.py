"""One date form both structures can be compared in.

The two structures write a departure date differently — ``14050619`` in a
package and ``1405-06-19`` in a tour — so a filter that compared them as
written would match nothing, return an empty list, and report success. Every
date is brought to the packed form before it is compared.
"""

from __future__ import annotations

import re
from typing import Optional

from crawler.extraction.readers.prices import normalize_digits

_A_DATE = re.compile(r"^(1[34]\d{2})\D?(\d{1,2})\D?(\d{1,2})$")


def as_packed(text: Optional[str]) -> Optional[str]:
    """One Jalali date in the eight-digit form both files can be compared in.

    Example:
        >>> as_packed("1405-06-19"), as_packed("14050619"), as_packed("1405/6/9")
        ('14050619', '14050619', '14050609')
        >>> as_packed("۱۴۰۵-۰۶-۱۹")
        '14050619'
        >>> as_packed("soon") is None
        True
    """
    if not text:
        return None
    found = _A_DATE.match(normalize_digits(text).strip())
    if not found:
        return None
    year, month, day = (int(part) for part in found.groups())
    return f"{year:04d}{month:02d}{day:02d}"
