"""Run every doctest in the package, and say what failed.

There is no other test here. The doctests are the tests, and until this script
existed they could not all be run at once: ``python -m doctest`` compares output
character for character, so an example whose result was wrapped over two lines
to stay readable failed even though the code was right. Six did, in five files,
which left the only checking mechanism in the project permanently red and
therefore ignored.

``NORMALIZE_WHITESPACE`` is what fixes that. It lets a docstring wrap a long
result the way a person would write it, which is the point of keeping the
examples real — see the note on docstrings in CLAUDE.md.

Run it with the project venv:

    .venv/bin/python scripts/run_doctests.py
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crawler  # noqa: E402

FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

# Modules that reach the network or a browser when imported or exercised. Their
# examples are run by hand against the live sites, which is what the audit
# script is for.
SKIP = {"crawler.extraction.page_extract", "crawler.extraction.extractor",
        "crawler.extraction.tour_api", "crawler.llm.client",
        "crawler.utils.http"}


def main() -> int:
    tried = failed = 0
    for info in pkgutil.walk_packages(crawler.__path__, "crawler."):
        if info.name in SKIP:
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception as error:  # a module that cannot import is a failure
            print(f"IMPORT  {info.name}: {type(error).__name__}: {error}")
            failed += 1
            continue
        result = doctest.testmod(module, optionflags=FLAGS, verbose=False)
        tried += result.attempted
        failed += result.failed
        if result.failed:
            print(f"FAIL    {info.name}: {result.failed} of {result.attempted}")

    print(f"\n{tried - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
