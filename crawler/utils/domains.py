"""Access to the list of target sites in ``domains/targets.txt``.

Every stage of this project is checked against real pages before it is
trusted, so loading the target list needs to be one call rather than a path
literal copied into each script.
"""

from __future__ import annotations

import random
from pathlib import Path

from crawler.config import PROJECT_ROOT

DOMAINS_FILE: Path = PROJECT_ROOT / "domains" / "targets.txt"


def load_domains(path: Path = DOMAINS_FILE) -> list[str]:
    """Read the target sites, ignoring blank lines and ``#`` comments.

    Entries are returned exactly as written, since the profiler does its own
    normalization and the file is also meant to be read by a person.

    Example:
        >>> load_domains()
        ['https://nahalgasht.com', 'https://www.alefbatour.com', 'https://avisatravel.com', ...]
    """
    if not path.exists():
        return []

    domains: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            domains.append(entry)
    return domains


def sample_domains(count: int = 3, seed: int | None = None) -> list[str]:
    """Pick ``count`` target sites at random, for spot-checking a new rule.

    Random rather than the first few on purpose: a rule tuned against the same
    two sites every time looks finished long before it is. Pass ``seed`` when a
    check needs to be repeatable.

    The example seeds the picker and counts, rather than naming the sites it
    returns: the answer moves whenever ``domains/targets.txt`` does, and it
    already has — the file went from seven sites to ten.

    Example:
        >>> picked = sample_domains(2, seed=1)
        >>> len(picked), all(p in load_domains() for p in picked)
        (2, True)
        >>> picked == sample_domains(2, seed=1)
        True
    """
    domains = load_domains()
    if count >= len(domains):
        return domains
    return random.Random(seed).sample(domains, count)
