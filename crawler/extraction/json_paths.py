"""Reading the shape of a page's embedded JSON, before deciding what it means.

Every site names its JSON keys differently, but the *shape* is readable without
knowing the site: a tour name sits alone near the root, while hotel names sit
inside a repeating array. This module turns a JSON document into that shape —
paths and repeating structures — and stops there. Deciding that
``$.hotels[*].title`` is a hotel name is the mapping layer's job.

Two problems are solved by working in paths rather than bare key names:

* **Ambiguity.** ``title`` alone could be the tour or the hotel. ``$.title``
  and ``$.hotels[*].title`` cannot be confused.
* **Grain.** The ``[*]`` in a path *is* the repeating structure. Finding the
  arrays tells us how many rows the page has and which fields belong to each
  one, without guessing at parallel lists.

Array indices collapse to ``[*]`` so that a thousand hotels produce one path
rather than a thousand, which is what makes the output small enough to reason
about — and later, small enough to show an LLM.
"""

from __future__ import annotations

from typing import Any


# Values a path can end at. Anything else (dict, list) is a container that the
# walk descends into rather than a value to record.
_SCALARS = (str, int, float, bool, type(None))

ROOT = "$"


def _walk(node: Any, prefix: str, out: dict[str, list[Any]]) -> None:
    """Record every scalar under ``node``, keyed by its collapsed path.

    Recurses in document order and appends to ``out`` in place, so a path met
    many times (once per array element) accumulates its values in the order
    they appeared.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{prefix}.{key}", out)
        return

    if isinstance(node, list):
        for item in node:
            # Index collapses to [*]: one path per shape, not per element.
            _walk(item, f"{prefix}[*]", out)
        return

    if isinstance(node, _SCALARS):
        out.setdefault(prefix, []).append(node)


def flatten_json(data: Any) -> dict[str, list[Any]]:
    """Map every scalar in a JSON document to its collapsed path.

    A path repeats once per array element it was found under, so the length of
    each list says how many times that shape occurred — 7 hotels give 7 values
    under one path.

    Example:
        >>> flatten_json({
        ...     "title": "تور کوالالامپور",
        ...     "hotels": [
        ...         {"title": "هتل کورت", "price": 110000000},
        ...         {"title": "هتل آستو", "price": 125000000},
        ...     ],
        ... })
        {'$.title': ['تور کوالالامپور'],
         '$.hotels[*].title': ['هتل کورت', 'هتل آستو'],
         '$.hotels[*].price': [110000000, 125000000]}
    """
    out: dict[str, list[Any]] = {}
    _walk(data, ROOT, out)
    return out


def find_row_paths(data: Any) -> dict[str, int]:
    """Find the arrays of objects that could be the page's repeating rows.

    Only arrays whose elements are objects qualify. An array of plain strings
    (``"tags": ["a", "b"]``) is a multi-valued field, not a set of rows: it has
    no internal fields to keep associated, which is the whole reason rows exist.

    Every qualifying array is returned with its length, including single-element
    ones — a tour with one hotel is a real tour. Choosing between candidates is
    the mapping layer's decision, so this stays a report rather than a pick.

    A nested path is counted across all of its parents, so the number always
    matches how many rows :func:`extract_rows` returns for it: two days holding
    two and one hotels report three, not one.

    Example:
        >>> find_row_paths({
        ...     "title": "تور کوالالامپور",
        ...     "tags": ["مالزی", "آسیا"],
        ...     "hotels": [{"title": "هتل کورت"}, {"title": "هتل آستو"}],
        ...     "dates": [{"start": "۱۴۰۵/۰۵/۲۴"}],
        ... })
        {'$.hotels[*]': 2, '$.dates[*]': 1}
    """
    found: dict[str, int] = {}

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{prefix}.{key}")
            return

        if isinstance(node, list):
            objects = [item for item in node if isinstance(item, dict)]
            if objects:
                path = f"{prefix}[*]"
                # Accumulate: the same collapsed path is reached once per parent.
                found[path] = found.get(path, 0) + len(objects)
            for item in node:
                visit(item, f"{prefix}[*]")

    visit(data, ROOT)
    return found


def _resolve_container(data: Any, path: str) -> list[Any]:
    """Return the list of objects sitting at an array path, or an empty list.

    ``path`` is a row path as produced by :func:`find_row_paths`, e.g.
    ``"$.hotels[*]"``. Nested array paths are followed segment by segment, so
    ``"$.days[*].hotels[*]"`` collects the hotels of every day.
    """
    if not path.startswith(ROOT):
        return []

    # "$.days[*].hotels[*]" -> ["days[*]", "hotels[*]"]
    steps = [step for step in path[len(ROOT):].split(".") if step]
    current: list[Any] = [data]

    for step in steps:
        key = step[:-3] if step.endswith("[*]") else step
        nxt: list[Any] = []
        for node in current:
            if not isinstance(node, dict) or key not in node:
                continue
            value = node[key]
            if step.endswith("[*]"):
                if isinstance(value, list):
                    nxt.extend(value)
            else:
                nxt.append(value)
        current = nxt

    return current


def extract_rows(data: Any, path: str) -> list[dict[str, list[Any]]]:
    """Flatten each object at ``path`` on its own, keeping its fields together.

    This is what makes a row a row: the price and the hotel name in one entry
    are known to belong to each other, unlike the parallel lists
    :func:`flatten_json` produces. Paths inside a row are relative to it, so
    the same key reads the same way for every row.

    Example:
        >>> extract_rows(
        ...     {"hotels": [
        ...         {"title": "هتل کورت", "price": 110000000},
        ...         {"title": "هتل آستو", "price": 125000000},
        ...     ]},
        ...     "$.hotels[*]",
        ... )
        [{'$.title': ['هتل کورت'], '$.price': [110000000]},
         {'$.title': ['هتل آستو'], '$.price': [125000000]}]
    """
    return [
        flatten_json(item)
        for item in _resolve_container(data, path)
        if isinstance(item, dict)
    ]
