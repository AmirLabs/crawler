"""The two files the API serves, and how a run replaces them.

The crawl writes into ``logs/``, which it empties at the start of every run.
Serving from there would hand the client half a catalogue whenever they call
while a run is going — and they call at midnight, every night. So a finished
run *publishes*: it writes both files into ``data/published`` through a
temporary name and renames them, which is atomic on one filesystem. A reader
either sees last night's file or tonight's, never half of either.

Nothing here crawls. A request is a file read, which is why response time does
not depend on how long a crawl takes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from crawler.config import DATA_DIR

PUBLISHED_DIR = DATA_DIR / "published"

TOURS_FILE = "tours.json"
PACKAGES_FILE = "packages.json"


def publish(name: str, objects: list[dict], directory: Optional[Path] = None) -> Path:
    """Write one file where the API reads it, replacing the last one atomically.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = publish(TOURS_FILE, [{"tour_code": "31540001"}], Path(folder))
        ...     json.loads(path.read_text(encoding="utf-8"))
        [{'tour_code': '31540001'}]
    """
    folder = directory or PUBLISHED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / name
    writing = folder / f".{name}.writing"
    writing.write_text(
        json.dumps(objects, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(writing, final)
    return final


def read(name: str, directory: Optional[Path] = None) -> list[dict]:
    """The objects of one published file, or none where no run has published yet.

    An empty list rather than an error: a server that has never crawled is a
    server with nothing to say, not a broken one.

    Example:
        >>> read("nothing-here.json", Path("/tmp"))
        []
    """
    path = (directory or PUBLISHED_DIR) / name
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return loaded if isinstance(loaded, list) else []
