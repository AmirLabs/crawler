"""Collapsing the URLs that are really one tour into a single group.

alefbatour.com and avisatravel.com publish one page per departure date. The
pages carry an identical slug and differ only in their id::

    /tour/d/152904/تور-3شب-ارمنستان-اردیبهشت-1405
    /tour/d/152909/تور-3شب-ارمنستان-اردیبهشت-1405
    /tour/d/152910/تور-3شب-ارمنستان-اردیبهشت-1405

Treated as separate tours they produce 738 near-duplicate records where the
site sells 108 tours, and each record holds one date of an offer list the page
already shows in full. Grouping them first is what keeps the output honest.

Grouping is only as safe as the URL makes it, and the target sites differ:

* **Id in its own path segment** (``/tour/d/<id>/<slug>``) — the slug is a pure
  title, so two pages sharing it are the same tour. alefbatour and avisatravel
  work this way, and this is where the large collapse happens.
* **Id embedded in the slug** (``/tour/<id>-<slug>``) — stripping the id can
  leave a generic remainder. nahalgasht.com has ten pages that reduce to plain
  ``mashhad``, and those are ten different tours, not one with ten dates.

So a group records *why* it was formed, and single-token keys are never grouped
at all. The rest of the pipeline can then treat a strong group as one tour and
a weak one as a question rather than a fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import unquote, urlparse

# A leading number this long is an id; anything shorter is part of the title.
# liyaseir.com sells "9-days-southkorea" and "12-days-southkorea" as different
# tours, and stripping those digits merged two real tours into one.
_ID_MIN_DIGITS = 4

# A key shorter than this says too little to group on. nahalgasht.com's ten
# ``/tour/<id>-mashhad/`` pages are ten tours to Mashhad, not one.
_MIN_KEY_TOKENS = 2

# Why a set of URLs was grouped, strongest first.
GROUPED_SEPARATE_ID = "identical slug, id in its own path segment"
GROUPED_EMBEDDED_ID = "identical slug after removing an embedded id"
GROUPED_ALONE = "no other URL shares this slug"


def slug_tokens(url: str) -> list[str]:
    """Words of a URL's last path segment, with a leading id removed.

    Example:
        >>> slug_tokens("https://nahalgasht.com/tour/59262-istanbul-divan-hotel/")
        ['istanbul', 'divan', 'hotel']
        >>> slug_tokens("https://liyaseir.com/product/9-days-southkorea/")
        ['9', 'days', 'southkorea']
    """
    path = unquote(urlparse(url).path)
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return []

    tokens = [t for t in re.split(r"[-_\s]+", segments[-1].lower()) if t]
    if tokens and tokens[0].isdigit() and len(tokens[0]) >= _ID_MIN_DIGITS:
        tokens.pop(0)
    return tokens


def _has_id_segment(url: str) -> bool:
    """Whether a path segment other than the slug is a bare number.

    Example:
        >>> _has_id_segment("https://www.alefbatour.com/tour/d/152904/tour-armanestan")
        True
        >>> _has_id_segment("https://nahalgasht.com/tour/59262-istanbul-divan-hotel/")
        False
    """
    path = unquote(urlparse(url).path)
    segments = [segment for segment in path.split("/") if segment]
    return any(segment.isdigit() for segment in segments[:-1])


def tour_key(url: str) -> Optional[str]:
    """The identity two URLs must share to be the same tour, or ``None``.

    ``None`` means the URL says too little to group on, and the caller should
    keep it on its own rather than risk merging unrelated tours.

    Example:
        >>> tour_key("https://www.alefbatour.com/tour/d/152904/tour-3sb-armanestan")
        'tour-3sb-armanestan'
        >>> tour_key("https://nahalgasht.com/tour/27271-mashhad/") is None
        True
    """
    tokens = slug_tokens(url)
    if len(tokens) < _MIN_KEY_TOKENS:
        return None
    return "-".join(tokens)


@dataclass
class TourGroup:
    """URLs that appear to be one tour, with the evidence for that call.

    ``primary`` is the page to extract from. Any member will do on these sites,
    since each one lists the whole offer set, so the first URL in document
    order is used and the rest are kept for traceability.

    Example:
        >>> group = TourGroup(
        ...     key="tour-3sb-armanestan",
        ...     urls=["/tour/d/152904/tour-3sb-armanestan", "/tour/d/152909/tour-3sb-armanestan"],
        ...     reason=GROUPED_SEPARATE_ID,
        ... )
        >>> group.size, group.primary
        (2, '/tour/d/152904/tour-3sb-armanestan')
    """

    key: Optional[str]
    urls: list[str] = field(default_factory=list)
    reason: str = GROUPED_ALONE

    @property
    def size(self) -> int:
        return len(self.urls)

    @property
    def primary(self) -> str:
        """The URL to fetch for this tour.

        Example:
            >>> TourGroup(key="k", urls=["/a", "/b"]).primary
            '/a'
        """
        return self.urls[0]

    @property
    def is_strong(self) -> bool:
        """Whether the grouping rests on a separate id segment.

        A weak group is still worth surfacing, but it should be checked against
        the pages themselves before its members are treated as one tour.

        Example:
            >>> TourGroup(key="k", urls=["/a"], reason=GROUPED_SEPARATE_ID).is_strong
            True
        """
        return self.reason == GROUPED_SEPARATE_ID


def group_by_tour(urls: list[str]) -> list[TourGroup]:
    """Collapse a discovered URL list into one group per tour.

    URLs keep their original order, and so do the groups, so the output can be
    compared against the input without sorting. A URL whose slug says too
    little is returned as its own group rather than merged.

    Example:
        >>> groups = group_by_tour([
        ...     "https://x.com/tour/d/152904/tour-armanestan",
        ...     "https://x.com/tour/d/152909/tour-armanestan",
        ...     "https://x.com/tour/d/160000/tour-dubai-5-nights",
        ... ])
        >>> [(g.size, g.key) for g in groups]
        [(2, 'tour-armanestan'), (1, 'tour-dubai-5-nights')]
    """
    grouped: dict[str, TourGroup] = {}
    singles: list[TourGroup] = []
    order: list[TourGroup] = []

    for url in urls:
        key = tour_key(url)
        if key is None:
            group = TourGroup(key=None, urls=[url], reason=GROUPED_ALONE)
            singles.append(group)
            order.append(group)
            continue

        existing = grouped.get(key)
        if existing is None:
            reason = (
                GROUPED_SEPARATE_ID if _has_id_segment(url) else GROUPED_EMBEDDED_ID
            )
            group = TourGroup(key=key, urls=[url], reason=reason)
            grouped[key] = group
            order.append(group)
        else:
            existing.urls.append(url)

    # A key met only once was never really grouped; say so rather than implying
    # evidence that was not needed.
    for group in order:
        if group.key is not None and group.size == 1:
            group.reason = GROUPED_ALONE

    return order
