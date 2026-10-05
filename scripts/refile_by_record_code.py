# Puts every record back under the code it carries.
#
# store_tour filed a whole page under one code, but a page sold across two
# months holds a record per month and a code per month. The client reads the
# code out of the record and asks for its packages, and the answer was none.
#
# The record is believed over the column: the row's key was built from it.
#
#     .venv/bin/python scripts/refile_by_record_code.py data/crawler.db

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def repair(path: Path) -> None:
    db = sqlite3.connect(path)
    try:
        for table in ("packages", "tours"):
            wrong = db.execute(
                f"SELECT count(*) FROM {table} "
                f"WHERE json_extract(record, '$.tour_code') IS NOT NULL "
                f"AND json_extract(record, '$.tour_code') <> tour_code"
            ).fetchone()[0]
            db.execute(
                f"UPDATE {table} SET tour_code = json_extract(record, '$.tour_code') "
                f"WHERE json_extract(record, '$.tour_code') IS NOT NULL "
                f"AND json_extract(record, '$.tour_code') <> tour_code"
            )
            print(f"{path}: {table} refiled {wrong} rows")
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    for name in sys.argv[1:] or ["data/crawler.db"]:
        repair(Path(name))
