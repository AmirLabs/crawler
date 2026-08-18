"""Remembering what the model already answered, so it is never asked twice.

Tour names repeat far more than tours do. alefbatour publishes 738 pages that
are 108 tours, and every page of a tour carries the same title, so asking about
each page would pay 738 times for 108 answers. A daily refresh makes it worse:
almost every name it meets was already answered yesterday.

The cache is keyed by the *question*, not by the tour — the text asked about,
the model, and the version of the prompt. Leaving the prompt out of the key is
the subtle way a cache goes wrong: improve the wording, and every old answer
keeps coming back, so the improvement never appears and the cache looks like it
is working.

It is a plain JSON file. There are a few thousand names in the whole project,
which is small enough that a database would be more machinery than the problem
deserves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from crawler.config import DATA_DIR

DEFAULT_CACHE_PATH = DATA_DIR / "llm_cache.json"


def cache_key(text: str, *, model: str, prompt_version: str) -> str:
    """The identity of one question, as asked.

    Example:
        >>> cache_key("تور آلانیا", model="gpt-4o-mini", prompt_version="tour-name-1")
        '33b940f63e45a7bc'
    """
    material = f"{prompt_version}|{model}|{text}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


class AnswerCache:
    """Answers already given, stored beside the crawl data.

    Loaded once and written when asked. Nothing here tries to be safe against
    two processes writing at the same time: the crawl is one process, and
    pretending otherwise would add locking for a problem we do not have.

    Example:
        >>> cache = AnswerCache(path=Path("/tmp/example_cache.json"))
        >>> cache.get("تور آلانیا", model="m", prompt_version="v1") is None
        True
        >>> cache.put("تور آلانیا", {"destinations": ["آلانیا"]}, model="m", prompt_version="v1")
        >>> cache.get("تور آلانیا", model="m", prompt_version="v1")
        {'destinations': ['آلانیا']}
    """

    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self._entries: dict[str, Any] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupted cache is not worth failing a crawl over; it costs
            # one round of questions to rebuild.
            self._entries = {}

    def get(self, text: str, *, model: str, prompt_version: str) -> Optional[Any]:
        """The stored answer for this exact question, if there is one.

        Example:
            >>> AnswerCache(path=Path("/tmp/missing.json")).get("x", model="m", prompt_version="v") is None
            True
        """
        return self._entries.get(cache_key(text, model=model, prompt_version=prompt_version))

    def put(self, text: str, answer: Any, *, model: str, prompt_version: str) -> None:
        """Store an answer. Nothing is written to disk until :meth:`save`.

        Example:
            >>> cache = AnswerCache(path=Path("/tmp/example_cache.json"))
            >>> cache.put("تور دبی", {"destinations": ["دبی"]}, model="m", prompt_version="v1")
            >>> cache.get("تور دبی", model="m", prompt_version="v1")
            {'destinations': ['دبی']}
        """
        self._entries[cache_key(text, model=model, prompt_version=prompt_version)] = answer
        self._dirty = True

    def save(self) -> None:
        """Write the cache out, if anything was added since it was read.

        Example:
            >>> cache = AnswerCache(path=Path("/tmp/example_cache.json"))
            >>> cache.put("تور کیش", {"destinations": ["کیش"]}, model="m", prompt_version="v1")
            >>> cache.save()
            >>> Path("/tmp/example_cache.json").exists()
            True
        """
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        self._dirty = False

    def __len__(self) -> int:
        """How many answers are held.

        Example:
            >>> len(AnswerCache(path=Path("/tmp/missing.json")))
            0
        """
        return len(self._entries)
