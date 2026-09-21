# Builds a page of every hotel photograph beside its hotel, so they can be
# looked at together instead of opened one address at a time.
#
#     .venv/bin/python tests/hotel_photo_sheet.py

from __future__ import annotations

import asyncio
import html as html_escape
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.extraction.extractor import DEFAULT_HEADERS
from crawler.schema.converter import build_tour
from crawler.schema.identifiers import IdStore
from crawler.schema.safarbank import tour_to_objects

from check_hotel_photos import PAGES, TIMEOUT, read_page

OUT = Path("logs/hotel-photos.html")

CSS = """
body{font:14px system-ui;margin:0;padding:24px;background:#faf9f7;color:#222}
h1{font-size:18px;margin:0 0 4px}
p.sub{color:#666;margin:0 0 24px}
h2{font-size:15px;margin:30px 0 4px}
p.src{margin:0 0 12px;padding-bottom:8px;border-bottom:1px solid #ddd;font-size:12px}
p.src a{color:#1a5fb4;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}
figure{margin:0;background:#fff;border:1px solid #e3e0dc;border-radius:8px;overflow:hidden}
img{width:100%;height:130px;object-fit:cover;display:block;background:#eee}
figcaption{padding:7px 9px;font-size:12px;line-height:1.35}
.name{font-weight:600;word-break:break-word}
.none{color:#999;font-style:italic;padding:44px 9px;text-align:center;font-size:12px}
a{color:#666;font-size:11px;word-break:break-all;text-decoration:none}
"""


async def main() -> int:
    parts = [f"<style>{CSS}</style>", "<h1>عکس هتل‌ها، سایت به سایت</h1>",
             '<p class="sub">هر عکس از همان بلوکی برداشته شده که هتل را نام '
             "می‌برد. صفحهٔ تور هر سایت بالای عکس‌هایش آمده — بازش کن و "
             "مقایسه کن.</p>"]
    found = total = 0
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers=DEFAULT_HEADERS
    ) as client:
        for site, url in PAGES.items():
            page, error = await read_page(client, url)
            if page is None:
                parts.append(f"<h2>{site} — {html_escape.escape(str(error))}</h2>")
                parts.append(f'<p class="src"><a href="{html_escape.escape(url, quote=True)}">{html_escape.escape(url)}</a></p>')
                continue
            rows = tour_to_objects(build_tour(page), IdStore(path=None), site)
            seen: dict[str, str] = {}
            for row in rows:
                for hotel in row.get("hotels") or []:
                    seen.setdefault(hotel.get("name") or "(بی‌نام)",
                                    hotel.get("logo") or "")
            with_photo = sum(1 for a in seen.values() if a)
            found += with_photo
            total += len(seen)
            parts.append(f"<h2>{site} — {with_photo} از {len(seen)} هتل</h2>")
            parts.append(
                f'<p class="src">صفحهٔ تور: '
                f'<a href="{html_escape.escape(url, quote=True)}">'
                f"{html_escape.escape(url)}</a></p>"
            )
            parts.append('<div class="grid">')
            for name, address in seen.items():
                safe = html_escape.escape(name)
                if not address:
                    parts.append(f'<figure><div class="none">عکسی ندارد</div>'
                                 f'<figcaption><span class="name">{safe}</span>'
                                 f"</figcaption></figure>")
                    continue
                link = html_escape.escape(address, quote=True)
                parts.append(
                    f'<figure><img src="{link}" loading="lazy" alt="">'
                    f'<figcaption><span class="name">{safe}</span><br>'
                    f'<a href="{link}">{html_escape.escape(address.rsplit("/", 1)[-1])}</a>'
                    f"</figcaption></figure>"
                )
            parts.append("</div>")
            print(f"{site}: {with_photo} of {len(seen)}", flush=True)

    parts.insert(2, f'<p class="sub">{found} از {total} هتل عکس دارند</p>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
