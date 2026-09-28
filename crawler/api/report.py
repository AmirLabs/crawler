"""The catalogue as pages to look at, for the client to check a site by eye.

Three levels, because the two structures already are two levels and forcing
them into one table repeats every tour once per hotel — 24 times on average,
61 for one of them:

    /report                     the ten sites, with how much each holds
    /report/<site>              that site's tours, one row per departure
    /report/<site>/<api_id>     one tour, with every package under it

Nothing here judges the data. A field the site did not state is shown empty
and left at that; what counts as a fault is Amir's call and the client's, not
this page's. The pages read the same records the API serves, so what is seen
here is what was sent.
"""

from __future__ import annotations

from html import escape
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from crawler.api.queries import as_packed
from crawler.storage.database import Database

router = APIRouter()

_STYLE = """
*{box-sizing:border-box}
body{margin:0;padding:24px;background:#f6f7f9;color:#1c1e21;
 font:15px/1.7 Vazirmatn,Tahoma,system-ui,sans-serif;direction:rtl}
a{color:#1550a0;text-decoration:none}
a:hover{text-decoration:underline}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#6b7280;font-size:13px;margin-bottom:20px}
.crumb{font-size:13px;margin-bottom:14px;color:#6b7280}
table{width:100%;border-collapse:collapse;background:#fff;
 box-shadow:0 1px 2px rgba(0,0,0,.08);border-radius:8px;overflow:hidden}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #eceef1;
 vertical-align:top}
th{background:#f0f2f5;font-weight:600;font-size:13px;color:#4b5563;
 position:sticky;top:0}
tr:last-child td{border-bottom:none}
tr:hover td{background:#fafbfc}
td.n{direction:ltr;text-align:left;font-variant-numeric:tabular-nums}
td.empty{color:#c3c7cc}
td.nw{white-space:nowrap}
.card{background:#fff;border-radius:8px;padding:16px 18px;margin-bottom:16px;
 box-shadow:0 1px 2px rgba(0,0,0,.08)}
.card h2{font-size:15px;margin:0 0 10px;color:#374151}
dl{display:grid;grid-template-columns:220px 1fr;gap:4px 16px;margin:0;
 font-size:13px}
dt{color:#6b7280;direction:ltr;text-align:left}
dd{margin:0;word-break:break-word}
"""


def _page(title: str, body: str) -> HTMLResponse:
    """Wrap a body in the one layout every level shares."""
    return HTMLResponse(
        f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def _cell(value: object, wide: bool = False) -> str:
    """One table cell, with an empty value shown as empty rather than as zero.

    ``wide`` keeps a price on one line; a wrapped "ریال" reads as a second row.

    Example:
        >>> _cell("کیش")
        '<td>کیش</td>'
        >>> _cell("")
        "<td class='empty'>—</td>"
        >>> _cell("4,920,000 ریال", wide=True)
        "<td class='nw'>4,920,000 ریال</td>"
    """
    text = "" if value is None else str(value)
    if not text.strip():
        return "<td class='empty'>—</td>"
    return f"<td class='nw'>{escape(text)}</td>" if wide else f"<td>{escape(text)}</td>"


def _money(price: Optional[dict]) -> str:
    """The starting price as the record carries it: rial, currency, or both.

    Example:
        >>> _money({"price": 4920000, "price_currency": "0.00", "Currency_title": ""})
        '4,920,000 ریال'
        >>> _money({"price": 0, "price_currency": "775.00", "Currency_title": "دلار"})
        '775.00 دلار'
        >>> _money({"price": 0, "price_currency": "0.00", "Currency_title": ""})
        ''
    """
    price = price or {}
    parts = []
    rial = float(price.get("price") or 0)
    if rial > 0:
        parts.append(f"{int(rial):,} ریال")
    foreign = float(price.get("price_currency") or 0)
    if foreign > 0:
        parts.append(f"{price.get('price_currency')} {price.get('Currency_title', '')}".strip())
    return " + ".join(parts)


def _open() -> Database:
    from crawler.api import app as api

    return Database(api.DATABASE_PATH)


@router.get("/report", response_class=HTMLResponse)
def sites() -> HTMLResponse:
    """The ten sites, largest first, with what each one holds tonight.

    Example:
        >>> client = _a_client()
        >>> client.get("/report").status_code
        401
        >>> answer = client.get("/report", auth=("them", "s3cret"))
        >>> answer.status_code, "آژانس" in answer.text
        (200, True)
    """
    db = _open()
    try:
        rows = db.agencies()
    finally:
        db.close()
    body = [
        "<h1>گزارش آژانس‌ها</h1>",
        f"<div class='sub'>{len(rows)} آژانس، "
        f"{sum(r['tours'] for r in rows):,} تور، "
        f"{sum(r['packages'] for r in rows):,} پکیج</div>",
        "<table><tr><th>آژانس</th><th>دامنه</th><th>تور</th><th>پکیج</th></tr>",
    ]
    for row in rows:
        link = f"/report/{escape(row['website'])}"
        body.append(
            f"<tr><td><a href='{link}'>{escape(row['name'] or row['website'])}</a></td>"
            f"<td class='n'>{escape(row['website'])}</td>"
            f"<td class='n'>{row['tours']:,}</td>"
            f"<td class='n'>{row['packages']:,}</td></tr>"
        )
    body.append("</table>")
    return _page("گزارش آژانس‌ها", "".join(body))


@router.get("/report/{website}", response_class=HTMLResponse)
def site(website: str) -> HTMLResponse:
    """One site's tours, one row per departure, in the shape of the second structure.

    Example:
        >>> client = _a_client()
        >>> client.get("/report/nowhere.com", auth=("them", "s3cret")).status_code
        404
    """
    db = _open()
    try:
        rows = db.tours_of(website)
    finally:
        db.close()
    if not rows:
        raise HTTPException(status_code=404, detail="Unknown site")

    body = [
        f"<div class='crumb'><a href='/report'>آژانس‌ها</a> ← {escape(website)}</div>",
        f"<h1>{escape(rows[0].get('agency_name') or website)}</h1>",
        f"<div class='sub'>{len(rows):,} تور</div>",
        "<table><tr><th>نام تور</th><th>تاریخ حرکت</th><th>شب/روز</th>"
        "<th>مقصد</th><th>قیمت شروع</th><th>وسیله</th><th>کد تور</th>"
        "<th>صفحه‌ی سایت</th></tr>",
    ]
    for tour in rows:
        link = f"/report/{escape(website)}/{escape(tour.get('api_id', ''))}"
        nights = f"{tour.get('night') or ''}/{tour.get('day') or ''}".strip("/")
        page = tour.get("tour_detail_address") or ""
        body.append(
            "<tr>"
            f"<td><a href='{link}'>{escape(tour.get('tour_name') or '(بدون نام)')}</a></td>"
            + _cell(tour.get("start_date"))
            + _cell(nights)
            + _cell(tour.get("destination_city_name"))
            + _cell(_money(tour.get("start_price")), wide=True)
            + _cell(tour.get("vehicle_name"))
            + f"<td class='n'>{escape(tour.get('tour_code') or '')}</td>"
            + (f"<td><a href='{escape(page)}' target='_blank' rel='noreferrer'>باز کردن</a></td>"
               if page else "<td class='empty'>—</td>")
            + "</tr>"
        )
    body.append("</table>")
    return _page(rows[0].get("agency_name") or website, "".join(body))


@router.get("/report/{website}/{api_id}", response_class=HTMLResponse)
def tour(website: str, api_id: str) -> HTMLResponse:
    """One tour with everything held about it: its own fields, then its packages.

    The fields keep the names the API sends them under, so a value read here
    can be looked for in their own import without translating anything.

    Example:
        >>> client = _a_client()
        >>> client.get("/report/x.com/nothing", auth=("them", "s3cret")).status_code
        404
    """
    db = _open()
    try:
        record = db.tour_named(api_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown tour")
        wanted = as_packed(record.get("start_date"))
        packages = [
            package for package in db.packages_for(record.get("tour_code", ""))
            if as_packed(package.get("start_date")) == wanted
        ]
    finally:
        db.close()

    body = [
        f"<div class='crumb'><a href='/report'>آژانس‌ها</a> ← "
        f"<a href='/report/{escape(website)}'>{escape(record.get('agency_name') or website)}</a>"
        f" ← {escape(record.get('tour_name') or api_id)}</div>",
        f"<h1>{escape(record.get('tour_name') or '(بدون نام)')}</h1>",
        f"<div class='sub'>{len(packages)} پکیج زیر این تور</div>",
        "<div class='card'><h2>ساختار دوم — خود تور</h2><dl>",
    ]
    for key, value in record.items():
        shown = "" if value in ("", None, [], {}) else str(value)
        body.append(
            f"<dt>{escape(key)}</dt>"
            + (f"<dd>{escape(shown)}</dd>" if shown else "<dd class='empty'>—</dd>")
        )
    body.append("</dl></div>")

    for number, package in enumerate(packages, start=1):
        hotels = package.get("hotels") or []
        rooms = package.get("rooms") or []
        body.append(f"<div class='card'><h2>ساختار اول — پکیج {number}</h2>")
        body.append(
            "<table><tr><th>هتل</th><th>ستاره</th><th>شهر</th><th>شب</th>"
            "<th>سرویس</th></tr>"
        )
        for hotel in hotels:
            body.append(
                "<tr>" + _cell(hotel.get("name")) + _cell(hotel.get("star_code"))
                + _cell(hotel.get("city_name")) + _cell(hotel.get("night"))
                + _cell(hotel.get("room_service")) + "</tr>"
            )
        body.append("</table>")
        body.append(
            "<table style='margin-top:10px'><tr><th>اتاق</th><th>قیمت</th>"
            "<th>قیمت ارزی</th><th>ظرفیت</th></tr>"
        )
        for room in rooms:
            amount = float(room.get("final_price") or 0)
            body.append(
                "<tr>" + _cell(room.get("name"))
                + _cell(f"{int(amount):,}" if amount > 0 else "", wide=True)
                + _cell(f"{room.get('currency_price')} {room.get('currency_name')}".strip()
                        if room.get("currency_price") else "")
                + _cell(room.get("capacity")) + "</tr>"
            )
        body.append("</table>")
        body.append(
            f"<dl style='margin-top:10px'><dt>start_date</dt><dd>"
            f"{escape(package.get('start_date') or '—')}</dd>"
            f"<dt>id</dt><dd>{escape(package.get('id') or '')}</dd>"
            f"<dt>fk_tour_id</dt><dd>{escape(package.get('fk_tour_id') or '')}</dd></dl>"
        )
        body.append("</div>")

    return _page(record.get("tour_name") or api_id, "".join(body))


def _a_client():
    """A client of the whole app with credentials in place, for the examples above."""
    from crawler.api.app import _a_test_client

    return _a_test_client()
