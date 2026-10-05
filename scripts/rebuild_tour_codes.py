# Lets one tour code sit under every address its tour is sold at.
#
# The table was built with UNIQUE(code), which allowed a code only one key, so
# the code had to be filed under one member of the family. The night that
# member departed and left the family, the row was never found again and the
# tour was given a second code — still holding the prices of the week it left.
#
# Run once per database file. It is safe to run again: it checks first.
#
#     .venv/bin/python scripts/rebuild_tour_codes.py data/crawler.db

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def already_done(db: sqlite3.Connection) -> bool:
    sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='tour_codes'"
    ).fetchone()
    return bool(sql) and "UNIQUE" not in sql[0].upper()


def rebuild(path: Path) -> None:
    db = sqlite3.connect(path)
    try:
        if already_done(db):
            print(f"{path}: already rebuilt, nothing to do")
            return
        before = db.execute("SELECT count(*) FROM tour_codes").fetchone()[0]
        db.executescript(
            """
            PRAGMA foreign_keys=off;
            BEGIN;
            CREATE TABLE tour_codes_new (
                "key" VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY ("key")
            );
            INSERT INTO tour_codes_new ("key", code) SELECT "key", code FROM tour_codes;
            DROP TABLE tour_codes;
            ALTER TABLE tour_codes_new RENAME TO tour_codes;
            CREATE INDEX ix_tour_codes_code ON tour_codes (code);
            COMMIT;
            PRAGMA foreign_keys=on;
            """
        )
        after = db.execute("SELECT count(*) FROM tour_codes").fetchone()[0]
        print(f"{path}: {before} codes before, {after} after")
    finally:
        db.close()


if __name__ == "__main__":
    for name in sys.argv[1:] or ["data/crawler.db"]:
        rebuild(Path(name))
