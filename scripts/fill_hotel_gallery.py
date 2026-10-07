# Puts each hotel's photograph where the client reads it, and makes the address
# one they can fetch.
#
# It was sent as "logo" alone and their site showed no picture, because the
# gallery is what they display. And every picture after a stay's first hotel
# was left relative to the site it came from, which nobody else can resolve.
#
# Both are already in the database, so nothing has to be crawled again. The
# hash is recomputed to match what the fixed code now builds, so tonight's run
# sees these rows as unchanged and leaves them alone.
#
#     .venv/bin/python scripts/fill_hotel_gallery.py data/crawler.db

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.automation.memory import record_hash


def fill(path: Path) -> None:
    db = sqlite3.connect(path)
    try:
        pages = dict(db.execute(
            "SELECT tour_code, json_extract(record, '$.source.page_url') FROM tours"
        ).fetchall())
        moved = fixed = rows = 0
        for key, code, raw in db.execute(
            "SELECT key, tour_code, record FROM packages"
        ).fetchall():
            record = json.loads(raw)
            hotels = record.get("hotels") or []
            touched = False
            for hotel in hotels:
                picture = (hotel.get("logo") or "").strip()
                page = pages.get(code)
                if picture and not picture.startswith("http") and page:
                    picture = urljoin(page, picture)
                    hotel["logo"] = picture
                    fixed += 1
                    touched = True
                wanted = [{"pic_address": picture}] if picture else []
                if hotel.get("gallery") != wanted:
                    hotel["gallery"] = wanted
                    touched = True
                    if picture:
                        moved += 1
            if not touched:
                continue
            rows += 1
            db.execute(
                "UPDATE packages SET record = ?, record_hash = ? WHERE key = ?",
                (json.dumps(record, ensure_ascii=False), record_hash(record), key),
            )
        db.commit()
        print(f"{path}: {moved} pictures into gallery, {fixed} addresses made "
              f"absolute, across {rows} packages")
    finally:
        db.close()


if __name__ == "__main__":
    for name in sys.argv[1:] or ["data/crawler.db"]:
        fill(Path(name))
