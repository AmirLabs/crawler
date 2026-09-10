"""The run that goes to the client. Every tour page of every site, once a night.

This is the one to schedule. It takes no arguments on purpose: there is nothing
to decide at the point where a machine starts it, and a flag left wrong is how
a night quietly delivers a tenth of the catalogue.

It does what ``scripts/run_client_output.py`` does, over the whole of every site
in ``domains/targets.txt`` rather than a few pages of each. That other script
keeps its ``--domain`` and ``--pages`` for trying one site by hand; this one is
what production runs.

Expect it to take hours the first night. From the second on, a page that came
back exactly as it was is fetched and left unread, which is most of them — see
:mod:`crawler.automation.memory`.

What it leaves behind::

    data/published/     what the API serves
    data/ids.json       the tour numbers, which cannot be rebuilt if lost
    data/seen.json      what last night saw
    data/runs/          everything this printed, kept a fortnight
    logs/client/        the same two files plus a summary, for reading by eye

Run::

    .venv/bin/python scripts/run_nightly.py

It exits non-zero if any site failed, and refuses to start beside a run that
is already going.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.automation.scheduler import AlreadyRunning, only_one_run, run_log
from scripts.run_client_output import main

# No cap: every tour URL the site declares.
EVERY_PAGE = 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(__doc__)
        sys.exit(2)
    try:
        with only_one_run(), run_log() as log:
            print(f"nightly run, every page, logging to {log}", flush=True)
            sys.exit(1 if asyncio.run(main(None, EVERY_PAGE)) else 0)
    except AlreadyRunning as reason:
        print(f"not starting: {reason}")
        sys.exit(1)
