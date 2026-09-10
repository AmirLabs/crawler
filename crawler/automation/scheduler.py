"""Step 6: what a run needs to survive being started by a machine.

Every crawl so far was a person typing a command and watching it. A scheduled
one has nobody watching, which changes three things:

**Only one at a time.** Two runs both read ``data/ids.json``, both write it, and
the second erases the first. That file cannot be rebuilt — nothing can work out
which tour held ``31540001`` yesterday — so a second start refuses rather than
waits. A lock left behind by a run that was killed is taken over, because a
crawl that silently stops running every night is worse than one that overlaps.

**A log that outlives the run.** ``logs/`` is emptied at the start of every
crawl, so the record of a night that went wrong goes in ``data/runs`` and stays
for a fortnight.

**A failure that is visible.** A run that dies at 2am must not be discovered a
week later by the client, so the exit code says whether anything failed and
whatever schedules it can act on that.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Optional, TextIO

from crawler.config import DATA_DIR

LOCK_PATH = DATA_DIR / "crawl.lock"
RUNS_DIR = DATA_DIR / "runs"

# How many nights of logs to keep. Long enough to look back over a weekend.
LOGS_KEPT = 14


class AlreadyRunning(RuntimeError):
    """Raised when another crawl holds the lock and is still alive."""


def _is_alive(pid: int) -> bool:
    """Whether a process id belongs to something still running.

    Example:
        >>> _is_alive(os.getpid()), _is_alive(999_999)
        (True, False)
    """
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _stale(path: Path) -> bool:
    """Whether a lock file was left behind by a run that is no longer there."""
    try:
        return not _is_alive(int(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return True


@contextmanager
def only_one_run(path: Optional[Path] = None) -> Iterator[Path]:
    """Hold the crawl lock, refusing to start beside a run that is still going.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     lock = Path(folder) / "crawl.lock"
        ...     with only_one_run(lock):
        ...         try:
        ...             with only_one_run(lock):
        ...                 print("two at once")
        ...         except AlreadyRunning:
        ...             print("refused")
        ...     lock.exists()
        refused
        False
    """
    lock = path or LOCK_PATH
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and _stale(lock):
        lock.unlink(missing_ok=True)
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise AlreadyRunning(f"another crawl holds {lock}") from None
    with os.fdopen(handle, "w") as writing:
        writing.write(str(os.getpid()))
    try:
        yield lock
    finally:
        lock.unlink(missing_ok=True)


class _Tee(TextIO):
    """Everything written to it goes to both the terminal and the log file."""

    def __init__(self, screen: TextIO, file: TextIO) -> None:
        self._screen, self._file = screen, file

    def write(self, text: str) -> int:
        self._file.write(text)
        self._file.flush()
        return self._screen.write(text)

    def flush(self) -> None:
        self._screen.flush()
        self._file.flush()


@contextmanager
def run_log(directory: Optional[Path] = None) -> Iterator[Path]:
    """Keep a copy of everything a run prints, under tonight's date.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     with run_log(Path(folder)) as written:
        ...         print("crawling nahalgasht")
        ...     "nahalgasht" in written.read_text(encoding="utf-8")
        crawling nahalgasht
        True
    """
    folder = directory or RUNS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{date.today().isoformat()}.log"
    for old in sorted(folder.glob("20*.log"))[:-LOGS_KEPT]:
        old.unlink(missing_ok=True)

    screen = sys.stdout
    with path.open("a", encoding="utf-8") as file:
        sys.stdout = _Tee(screen, file)
        logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True)
        try:
            yield path
        finally:
            sys.stdout = screen
