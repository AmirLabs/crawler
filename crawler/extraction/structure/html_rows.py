"""Finding the repeating part of a tour page in its markup.

This is the HTML counterpart of :func:`crawler.extraction.structure.json_paths.find_row_paths`.
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
its list of related tours, so :mod:`crawler.extraction.readers.prices` does that
part: a group qualifies only when several of its members hold an amount with a
currency word beside it.

Like the JSON module, this reports candidates rather than picking one. Which
group is *the* offer list is a judgement that belongs with the code that
assembles the page, and it needs the counts kept here to make it.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import unquote, urlparse

from selectolax.parser import HTMLParser, Node

from crawler.extraction.readers.labels import names_other_tours
from crawler.extraction.readers.prices import find_prices

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

# Thresholds below were read off the real pages rather than chosen in advance.
# Measured median text length of a true offer row across the six sites ran from
# 22 to 146 characters; the groups that turned out to be whole page sections
# ran from 236 to 1108. The two bands do not overlap, so the cut sits between
# them with room on either side.
_TERSE_MEDIAN = 150   # at or below this, members read like rows
_VERBOSE_MEDIAN = 400  # at or above this, members read like page sections

# Share required to call a block a navigation picker rather than offers. Set
# high on purpose: this rejects a block outright, and the measured gap is total
# — pickers link on every member, offer blocks on none.
_LINK_MAJORITY_STRICT = 0.8

# A group must reach this to be worth treating as the page's offer rows.
MIN_ROW_SCORE = 0.5


# How far to look back and up for the heading that introduces a block.
_HEADING_LEVELS = 4
_HEADING_MAX_CHARS = 60

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


def _with_stragglers(members: list[Node], order: list[Node]) -> list[Node]:
    """Take back the row a list drops at its edge for styling it differently.

    The last item of a list is routinely given fewer classes than the rest —
    it carries no divider, because there is nothing after it to divide from.
    That is ordinary CSS and it is invisible on the page, but it changes the
    signature, so the row falls outside its own group and is never read.
    lachinseir.com sells four rooms and gave three: the fourth, "کودک بدون تخت
    1,400 دلار", differs from its siblings by exactly ``border-l-2
    border-dashed``.

    Taken back only where all four hold: the same tag, a class in common, a
    price of its own, and a place directly beside the run. Adjacency is what
    keeps this tight — a neighbour further along is a different section, and
    absorbing it would put someone else's price in this block.

    Example:
        >>> tree = HTMLParser(
        ...     '<ul><li class="row line">دو تخته 92,000,000 تومان</li>'
        ...     '<li class="row line">یک تخته 99,000,000 تومان</li>'
        ...     '<li class="row">کودک 45,000,000 تومان</li></ul>')
        >>> len(find_row_containers(tree)[0].members)
        3
    """
    if not members:
        return members
    mine = {id(node) for node in members}
    places = [i for i, node in enumerate(order) if id(node) in mine]
    first, last = places[0], places[-1]
    tag = members[0].tag
    shared = {
        cls
        for cls in (members[0].attributes.get("class") or "").split()
        if not _GENERATED_CLASS.search(cls)
    }

    taken = list(members)
    for step in (first - 1, last + 1):
        if not 0 <= step < len(order):
            continue
        node = order[step]
        if id(node) in mine or node.tag != tag:
            continue
        classes = set((node.attributes.get("class") or "").split())
        if not classes & shared:
            continue
        if not find_prices(node.text(deep=True) or ""):
            continue
        taken.append(node)

    taken.sort(key=lambda node: order.index(node))
    return taken


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

    score: float = 0.0
    # Why it scored that way, in plain words. Kept for the same reason as
    # UrlGroup.reasons in crawler.filtering: a wrong pick has to be explainable
    # without a debugger.
    reasons: list[str] = field(default_factory=list)

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


def heading_above(container: RowContainer) -> Optional[str]:
    """The short piece of text a page prints just before a block.

    Found by walking back through earlier siblings and then up a level, which
    is how these pages are built: the heading is a sibling of the list, or of
    the wrapper around it. Only short text counts — a long run before a block is
    the previous section, not a title for this one.

    Example:
        >>> tree = HTMLParser(
        ...     '<section><h3>سایر پیشنهادات</h3>'
        ...     '<ul><li>تور کیش 20,000,000 تومان</li>'
        ...     '<li>تور مشهد 15,000,000 تومان</li></ul></section>'
        ... )
        >>> heading_above(find_row_containers(tree)[0])
        'سایر پیشنهادات'
    """
    if not container.members:
        return None

    node: Optional[Node] = container.members[0].parent
    for _ in range(_HEADING_LEVELS):
        if node is None or node.tag in ("body", "html", "-undef"):
            return None
        previous = node.prev
        while previous is not None:
            text = " ".join((previous.text(deep=True) or "").split())
            if 3 <= len(text) <= _HEADING_MAX_CHARS:
                return text
            previous = previous.prev
        node = node.parent
    return None


def _is_navigation(container: RowContainer) -> bool:
    """Whether this block is a picker that sends the reader somewhere else.

    alefbatour.com and avisatravel.com publish one page per departure and print
    the tour's other departures above the hotels, each one a link to its own
    page. Those pages are crawled in their own right, so reading the picker here
    stores another page's departures as this one's offers and buries the hotel
    list that the page is actually selling.

    The test is that essentially every member carries a link out. A real offer
    row does not: measured over eight rendered pages, every date picker had a
    link on all of its members and every hotel block had none at all.

    Example:
        >>> tree = HTMLParser(
        ...     '<ul><li><a href="/tour/d/1/x">03 شهریور 98,000,000 تومان</a></li>'
        ...     '<li><a href="/tour/d/2/x">10 شهریور 99,000,000 تومان</a></li></ul>'
        ... )
        >>> _is_navigation(find_row_containers(tree)[0])
        True
    """
    if not container.members:
        return False
    linked = 0
    for member in container.members:
        anchor = member.css_first("a[href]")
        href = (anchor.attributes.get("href") or "") if anchor is not None else ""
        if href and not href.startswith("#"):
            linked += 1
    return linked >= container.count * _LINK_MAJORITY_STRICT


def score_row_container(
    container: RowContainer, page_url: Optional[str] = None
) -> float:
    """Judge whether this group is the page's own offer rows.

    Writes the score and its plain-word reasons onto ``container`` and returns
    the score, the same way :func:`crawler.filtering.url_filter._score_group`
    does. Signals are weighed rather than combined into one rule, because no
    single one holds on all six sites: bisanseir's room rows carry no links at
    all, alefbatour's date rows link to a sibling page, and both are genuine.

    ``page_url`` enables the link signals. Without it the group is judged on
    shape alone, which still separates a row from a page section but cannot
    tell this tour's departures from a list of other tours.

    Example:
        >>> tree = HTMLParser(
        ...     '<div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div>'
        ... )
        >>> container = find_row_containers(tree)[0]
        >>> round(container.score, 2), container.reasons
        (0.55, ['every member holds a price', 'members read like rows (median 24 chars)'])
    """
    score = 0.0
    reasons: list[str] = []

    # --- how many members actually carry a price -------------------------- #
    density = container.with_price / container.count if container.count else 0.0
    if density >= 0.8:
        score += 0.25
        reasons.append("every member holds a price")
    elif density >= 0.4:
        score += 0.1
        reasons.append(f"{container.with_price} of {container.count} members hold a price")

    # --- rows are terse; page sections are not ---------------------------- #
    texts = container.texts()
    median_length = statistics.median([len(t) for t in texts]) if texts else 0
    if median_length <= _TERSE_MEDIAN:
        score += 0.3
        reasons.append(f"members read like rows (median {median_length:.0f} chars)")
    elif median_length >= _VERBOSE_MEDIAN:
        score -= 0.4
        reasons.append(f"members read like page sections (median {median_length:.0f} chars)")

    # An empty member means the group is a layout wrapper that happens to
    # repeat, not a list of comparable things.
    if any(not t for t in texts):
        score -= 0.25
        reasons.append("some members are empty")

    # --- position in the page --------------------------------------------- #
    if container.path.count(">") >= 2:
        score += 0.15
        reasons.append("sits deep in the page")

    # --- whose tours are these -------------------------------------------- #
    # The page says so above the block. This replaced a rule that compared the
    # members' link slugs against the page's own, which was a coin flip: the
    # same kind of nahalgasht strip scored 0.85 on one page and 0.10 on
    # another, because its variants differ by hotel name in the slug. A heading
    # is what the site actually wrote, and it reads the same on every site.
    other = names_other_tours(heading_above(container))
    if other:
        score -= 0.6
        reasons.append(f"sits under a heading for other tours ('{other}')")
    elif _is_navigation(container):
        score -= 0.6
        reasons.append("every member links away — a picker, not this page's offers")

    container.score = max(0.0, min(1.0, score))
    container.reasons = reasons
    return container.score


def find_row_containers(
    tree: HTMLParser | Node, page_url: Optional[str] = None
) -> list[RowContainer]:
    """Find every group of look-alike siblings that carries prices.

    Groups come back best-scored first, so the caller can take the head of the
    list and check it against :data:`MIN_ROW_SCORE`. Identical groups are
    reported once: these sites often render the same list twice, one copy for
    mobile and one for desktop, and both would otherwise be returned.

    Pass ``page_url`` whenever it is known — it is what lets scoring tell a
    list of this tour's departure dates from a list of other tours entirely.

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
        >>> [(c.path, c.count, c.with_price, round(c.score, 2)) for c in find_row_containers(tree)]
        [('div.beds > div.bed', 2, 2, 0.55)]
    """
    root = tree.body if isinstance(tree, HTMLParser) else tree
    if root is None:
        return []

    found: list[RowContainer] = []

    for parent in root.traverse(include_text=False):
        if parent.tag in _SKIP_TAGS:
            continue

        groups: dict[str, list[Node]] = defaultdict(list)
        order: list[Node] = []
        child = parent.child
        while child is not None:
            if child.tag not in _SKIP_TAGS:
                groups[_signature(child)].append(child)
                order.append(child)
            child = child.next

        for signature, members in groups.items():
            if len(members) < _MIN_MEMBERS:
                continue
            members = _with_stragglers(members, order)
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

    for container in found:
        score_row_container(container, page_url=page_url)

    # Drop duplicates: same members' text means the same list, however it was
    # reached. Keeps the mobile and desktop copies of one list from both
    # counting as offers.
    unique: list[RowContainer] = []
    seen: set[tuple[str, ...]] = set()
    for container in sorted(found, key=lambda c: (-c.score, -c.with_price, -c.count)):
        fingerprint = tuple(container.texts())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(container)

    return unique


def offer_blocks(
    containers: list[RowContainer], min_score: float = MIN_ROW_SCORE
) -> list[RowContainer]:
    """Pick every block that holds this page's offers, not just the best one.

    A tour page rarely keeps its offers in one list. bisanseir.com gives each
    hotel its own block of room rows and prints seventeen of them down the page;
    taking the best-scoring block alone kept four rows out of sixty-four, and
    one hotel out of sixteen.

    What the wanted blocks share is the element they repeat, so the best block
    names the shape and every other block repeating that same shape joins it.
    That is what keeps the page's own room rows in and leaves out a differently
    built list — a sidebar of related tours — that happens to score well.

    Blocks come back in document order, since the page's order is the order a
    reader sees the hotels in and there is nothing better to sort by.

    Example:
        >>> tree = HTMLParser(
        ...     '<div class="hotel"><div class="beds">'
        ...     '<div class="bed">دو تخته 92,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 99,000,000 تومان</div>'
        ...     '</div></div>'
        ...     '<div class="hotel"><div class="beds">'
        ...     '<div class="bed">دو تخته 82,000,000 تومان</div>'
        ...     '<div class="bed">یک تخته 89,000,000 تومان</div>'
        ...     '</div></div>'
        ... )
        >>> blocks = offer_blocks(find_row_containers(tree))
        >>> [b.texts()[0] for b in blocks]
        ['دو تخته 92,000,000 تومان', 'دو تخته 82,000,000 تومان']
    """
    if not containers:
        return []

    best = containers[0]
    if best.score < min_score:
        return []

    kept = [
        container
        for container in containers
        if container.signature == best.signature and container.score >= min_score
    ]
    # ``containers`` arrives sorted by score, which scrambles blocks that scored
    # the same. Reading order is restored from where each block's first member
    # sits in the markup.
    kept.sort(key=lambda container: _document_position(container.members[0]))
    return kept


def _document_position(node: Node) -> tuple[int, ...]:
    """Where a node sits in the document, as its index at each level.

    Comparing two of these compares document order, which is what restores a
    reading order that sorting by score threw away.

    Example:
        >>> tree = HTMLParser('<div><p>a</p><p>b</p></div>')
        >>> first, second = tree.css("p")
        >>> _document_position(first) < _document_position(second)
        True
    """
    path: list[int] = []
    current: Optional[Node] = node
    while current is not None and current.parent is not None:
        index = 0
        sibling = current.parent.child
        while sibling is not None and sibling != current:
            index += 1
            sibling = sibling.next
        path.append(index)
        current = current.parent
    return tuple(reversed(path))
