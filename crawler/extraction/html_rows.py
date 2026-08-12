"""Finding the repeating part of a tour page in its markup.

This is the HTML counterpart of :func:`crawler.extraction.json_paths.find_row_paths`.
Where that module reads ``$.hotels[*]`` out of a JSON document, this one finds
the sibling elements a page repeats — the hotel cards, the departure-date rows,
the room-type lines — so their fields can be kept together as
:class:`~crawler.extraction.models.ExtractedRow` objects.

The rule is structural, not name-based, because class names carry no shared
meaning across the target sites: the same repeating unit is ``div.hotelcard``
on nahalgasht.com, a bare ``li`` on avisatravel.com and ``div.hotel__bed`` on
bisanseir.com. What they have in common is the shape — several siblings that
look alike, most of them carrying a price:

* nahalgasht.com: ``div.content`` holding 5x ``div.hotelcard`` (one per hotel)
* avisatravel.com: ``ul`` holding 4x ``li`` (one per departure date)
* bisanseir.com: ``div.hotel__beds`` holding 4x ``div.hotel__bed`` (one per room type)

A price is what separates a real row group from the page's navigation menu or
its list of related tours, so :mod:`crawler.extraction.text_values` does that
part: a group qualifies only when several of its members hold an amount with a
currency word beside it.

Like the JSON module, this reports candidates rather than picking one. Which
group is *the* offer list is a judgement that belongs with the code that
assembles the page, and it needs the counts kept here to make it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from selectolax.parser import HTMLParser, Node

from crawler.extraction.text_values import find_prices

# Elements that never hold page content worth reading as a row.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "-text", "-undef"})

# How many classes go into a signature. Enough to tell two components apart,
# few enough that a utility-class site (Tailwind puts a dozen on one element)
# does not give every sibling a different signature.
_SIGNATURE_CLASSES = 3

# A group must have at least this many members, and at least this many of them
# must hold a price. Two is deliberate: a tour with exactly two hotels is
# ordinary, and requiring three would drop it.
_MIN_MEMBERS = 2
_MIN_WITH_PRICE = 2

# Classes carrying a generated id rather than a meaning: Elementor stamps every
# widget with its own ``elementor-element-af40c31``. Left in the signature they
# make each sibling unique, so a repeating list stops looking repeated — which
# is exactly what hid the offer rows on liyaseir.com and vanrosh.com. Matched
# on the shape of the token (a long hex run or a long digit run) rather than on
# a builder's name, so a site using a different generator is covered too.
_GENERATED_CLASS = re.compile(r"[0-9a-f]{6,}|\d{4,}", re.I)


def _signature(node: Node) -> str:
    """Describe an element by its tag and leading meaningful classes.

    Two siblings with the same signature are the same kind of thing as far as
    the page is concerned, which is what makes them candidate rows. Classes
    that hold a generated id are dropped, since they differ per element by
    design and would make every sibling look unique.

    Example:
        >>> _signature(HTMLParser('<div class="hotel__bed x y z">a</div>').css_first("div"))
        'div.hotel__bed.x.y'
        >>> _signature(HTMLParser('<div class="elementor-element elementor-element-af40c31">a</div>').css_first("div"))
        'div.elementor-element'
    """
    classes = [
        cls
        for cls in (node.attributes.get("class") or "").split()
        if not _GENERATED_CLASS.search(cls)
    ]
    if not classes:
        return node.tag
    return node.tag + "." + ".".join(classes[:_SIGNATURE_CLASSES])


def _element_path(node: Node, max_depth: int = 4) -> str:
    """Build a readable path from an ancestor down to ``node``.

    Kept short on purpose: the point is to identify which part of the page a
    row set came from when reviewing a bad extraction, not to be a selector
    that can be replayed.

    Example:
        >>> tree = HTMLParser('<div class="wrap"><div class="beds"><div class="bed">x</div></div></div>')
        >>> _element_path(tree.css_first("div.bed"))
        'div.wrap > div.beds > div.bed'
    """
    chain: list[str] = [_signature(node)]
    current = node.parent
    while current is not None and current.tag not in ("body", "html", "-undef") and len(chain) < max_depth:
        chain.append(_signature(current))
        current = current.parent
    return " > ".join(reversed(chain))


@dataclass
class RowContainer:
    """A set of sibling elements that look like the page's repeating rows.

    ``members`` are the row elements themselves, in document order, so the
    caller can read fields out of each one without walking the tree again.

    Example:
        >>> tree = HTMLParser(
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div>'
        ... )
        >>> container = find_row_containers(tree)[0]
        >>> container.path, container.count, container.with_price
        ('div.beds > div.bed', 2, 2)
    """

    path: str            # "div.hotel__beds > div.hotel__bed"
    signature: str       # "div.hotel__bed" — what the members have in common
    members: list[Node] = field(default_factory=list)
    with_price: int = 0  # how many members hold an amount plus a currency word

    @property
    def count(self) -> int:
        return len(self.members)

    def texts(self) -> list[str]:
        """The visible text of each member, collapsed to one line.

        Example:
            >>> tree = HTMLParser(
            ...     '<ul><li>رفت 03 شهریور 98,000,000 تومان</li>'
            ...     '<li>رفت 10 شهریور 99,000,000 تومان</li></ul>'
            ... )
            >>> find_row_containers(tree)[0].texts()
            ['رفت 03 شهریور 98,000,000 تومان', 'رفت 10 شهریور 99,000,000 تومان']
        """
        return [" ".join((node.text(deep=True) or "").split()) for node in self.members]


def find_row_containers(tree: HTMLParser | Node) -> list[RowContainer]:
    """Find every group of look-alike siblings that carries prices.

    Groups are ranked with the most price-bearing first, then the largest, so
    the offer list tends to come before an unrelated repeating block. Identical
    groups are reported once: these sites often render the same list twice, one
    copy for mobile and one for desktop, and both would otherwise be returned.

    Returns an empty list when nothing repeats with prices — on a tour page
    that is itself a signal, either that the layout is unusual or that the page
    genuinely quotes no price.

    Example:
        >>> tree = HTMLParser(
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div>'
        ... )
        >>> [(c.path, c.count, c.with_price) for c in find_row_containers(tree)]
        [('div.beds > div.bed', 2, 2)]
    """
    root = tree.body if isinstance(tree, HTMLParser) else tree
    if root is None:
        return []

    found: list[RowContainer] = []

    for parent in root.traverse(include_text=False):
        if parent.tag in _SKIP_TAGS:
            continue

        groups: dict[str, list[Node]] = defaultdict(list)
        child = parent.child
        while child is not None:
            if child.tag not in _SKIP_TAGS:
                groups[_signature(child)].append(child)
            child = child.next

        for signature, members in groups.items():
            if len(members) < _MIN_MEMBERS:
                continue
            with_price = sum(
                1 for node in members if find_prices(node.text(deep=True) or "")
            )
            if with_price < _MIN_WITH_PRICE:
                continue
            found.append(
                RowContainer(
                    path=_element_path(members[0]),
                    signature=signature,
                    members=members,
                    with_price=with_price,
                )
            )

    # Drop duplicates: same members' text means the same list, however it was
    # reached. Keeps the mobile and desktop copies of one list from both
    # counting as offers.
    unique: list[RowContainer] = []
    seen: set[tuple[str, ...]] = set()
    for container in sorted(found, key=lambda c: (-c.with_price, -c.count)):
        fingerprint = tuple(container.texts())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(container)

    return unique
