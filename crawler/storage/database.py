"""Everything the crawl remembers between nights, behind one object.

It replaces three things that did this job apart: ``data/ids.json`` (the
numbers), ``data/seen.json`` (what last night saw) and ``data/published/``
(what the API serves). Kept apart, nothing joined them, and a page left
unread overnight had its tours vanish from the catalogue — issue #29.

A tour is dead, and deleted, when its departure has passed or its page has
not been seen for :data:`MISSED_NIGHTS` nights running. Its code is never
deleted: see :mod:`crawler.storage.tables`.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy import create_engine, delete, event, exists, func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from crawler.api.queries import as_packed
from crawler.automation.memory import record_hash, record_key
from crawler.config import DATA_DIR
from crawler.extraction.readers.dates import gregorian_to_jalali
from crawler.schema.identifiers import (
    TOUR_CODE_DIGITS,
    TOUR_CODE_PREFIX,
    hotel_key,
    tour_key,
)
from crawler.storage.tables import (
    Base,
    Counter,
    FkTourId,
    HotelId,
    Package,
    Page,
    RowId,
    Run,
    Tour,
    TourCode,
)

DEFAULT_DATABASE_PATH = DATA_DIR / "crawler.db"

# A site down for a night or two must not take its tours with it.
MISSED_NIGHTS = 3


# One engine per file, shared: the API opens a Database on every request.
_ENGINES: dict[str, Engine] = {}


def _engine(path: Optional[Path]) -> Engine:
    """The engine for one database file, made once."""
    if path is None:
        engine = create_engine("sqlite://", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        return engine
    key = str(path)
    if key not in _ENGINES:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}")

        @event.listens_for(engine, "connect")
        def _wal(connection, _record) -> None:
            # Lets the API read last night's catalogue while tonight's run writes.
            connection.execute("PRAGMA journal_mode=WAL")

        Base.metadata.create_all(engine)
        _ENGINES[key] = engine
    return _ENGINES[key]


def jalali_today(today: Optional[date] = None) -> str:
    """Today in the packed form both structures' dates compare in.

    Example:
        >>> jalali_today(date(2026, 9, 22))
        '14050631'
    """
    today = today or date.today()
    year, month, day = gregorian_to_jalali(today.year, today.month, today.day)
    return f"{year:04d}{month:02d}{day:02d}"


def _code_key(page_url: str, part: str) -> str:
    """What a tour code is remembered under: the page, and which of its tours.

    Example:
        >>> _code_key("https://x.com/tour/1?date=1405-06-11", "")
        'https://x.com/tour/1'
        >>> _code_key("https://x.com/tour/1", "مهر")
        'https://x.com/tour/1|مهر'
    """
    return tour_key(page_url) + (f"|{part}" if part else "")


class Database:
    """The crawl's memory, one SQLite file.

    ``path=None`` keeps it in memory, for the examples and for trying a rule.

    Example:
        >>> db = Database(path=None)
        >>> db.tour_code("https://x.com/tour/1?date=1405-06-11")
        '31540001'
        >>> db.tour_code("https://x.com/tour/1"), db.tour_code("https://x.com/tour/2")
        ('31540001', '31540002')
        >>> db.known_tour_code("https://x.com/tour/3") is None
        True
    """

    def __init__(self, path: Optional[Path] = DEFAULT_DATABASE_PATH) -> None:
        self.session = Session(_engine(path))
        self.run: Optional[int] = None

    def close(self) -> None:
        self.session.close()

    # --- numbers -------------------------------------------------------

    def _take(self, name: str) -> int:
        counter = self.session.get(Counter, name)
        if counter is None:
            counter = Counter(name=name, next=1)
            self.session.add(counter)
        number = counter.next
        counter.next = number + 1
        return number

    def known_tour_code(self, page_url: str, part: str = "") -> Optional[str]:
        """The code this page already holds, minting nothing."""
        found = self.session.get(TourCode, _code_key(page_url, part))
        return found.code if found else None

    def tour_code(self, page_url: str, part: str = "",
                  family: Sequence[str] = ()) -> str:
        """The code for this tour, minted once and reused for ever after.

        ``part`` separates one page that sells several tours. nahalgasht offers
        one page across two months and the client buys each month on its own,
        so each month needs a code of its own.

        ``family`` is every address the same tour is published under, where a
        site uses more than one. radingasht gives each departure its own page
        and each page names all the others, so keyed on the page alone one
        tour became several, each carrying every date of the family: the same
        departure reached the client twice, under two codes and at two
        different prices. One row is kept for the whole family and found from
        any member, so whichever page is read first, the code is the same.

        Where members already hold codes of their own, the oldest is kept and
        the rest are forgotten along with what was stored under them. They are
        written again under the code that stayed, in this same run.

        Example:
            >>> db = Database(path=None)
            >>> db.tour_code("https://x.com/tour/1")
            '31540001'
            >>> db.tour_code("https://x.com/tour/1", "مهر")
            '31540002'
            >>> db.tour_code("https://x.com/tour/1", "مهر")
            '31540002'

            Two pages of one family share a code, whichever is read first,
            and a page that joins the family later joins the code:

            >>> both = ("https://x.com/tour/7", "https://x.com/tour/8")
            >>> db.tour_code("https://x.com/tour/8", family=both)
            '31540003'
            >>> db.tour_code("https://x.com/tour/7", family=both)
            '31540003'
            >>> db.tour_code("https://x.com/tour/9", family=both + ("https://x.com/tour/9",))
            '31540003'

            A member that departs leaves the family, and the code stays:

            >>> first = ("https://x.com/tour/30", "https://x.com/tour/31")
            >>> db.tour_code("https://x.com/tour/31", family=first)
            '31540004'
            >>> later = ("https://x.com/tour/31", "https://x.com/tour/32")
            >>> db.tour_code("https://x.com/tour/32", family=later)
            '31540004'

            Codes minted apart are merged on the oldest:

            >>> db.tour_code("https://x.com/tour/20"), db.tour_code("https://x.com/tour/21")
            ('31540005', '31540006')
            >>> pair = ("https://x.com/tour/20", "https://x.com/tour/21")
            >>> db.tour_code("https://x.com/tour/21", family=pair)
            '31540005'
        """
        keys = list(dict.fromkeys(
            [_code_key(page_url, part)] + [_code_key(one, part) for one in family]
        ))
        held = [row for row in (self.session.get(TourCode, key) for key in keys) if row]

        if not held:
            code = f"{TOUR_CODE_PREFIX}{self._take('tour'):0{TOUR_CODE_DIGITS}d}"
        else:
            code = min(row.code for row in held)
            for gone in {row.code for row in held} - {code}:
                self.forget_tour(gone)
                for row in self.session.scalars(
                    select(TourCode).where(TourCode.code == gone)
                ).all():
                    row.code = code

        # Written against every address, not only one of them. Filed under a
        # single member, the row was lost the night that member departed and
        # left the family: alefbatour's 5-night Armenia was keyed on 184165,
        # which sold out and dropped off the sibling list, and the tour was
        # given a second code still carrying the prices of the week it left.
        for key in keys:
            row = self.session.get(TourCode, key)
            if row is None:
                self.session.add(TourCode(key=key, code=code))
            elif row.code != code:
                row.code = code
        self.session.flush()
        return code

    def forget_tour(self, code: str) -> None:
        """Delete everything stored under one tour code.

        Used when two codes turn out to be one tour. What goes is written
        again under the code that stayed, so nothing is lost to the client.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("31540001", [], [{"api_id": "a", "start_date": "1405-08-01"}])
            >>> db.forget_tour("31540001"); db.catalogue()[1]
            []
        """
        for table in (Package, Tour):
            for row in self.session.scalars(select(table).where(table.tour_code == code)):
                self.session.delete(row)

    def fk_tour_id(self, tour_code: str, start_date: str) -> str:
        """The id of one tour on one departure date.

        Example:
            >>> db = Database(path=None)
            >>> db.fk_tour_id("31540001", "14050604"), db.fk_tour_id("31540001", "14050605")
            ('1', '2')
            >>> db.fk_tour_id("31540001", "14050604")
            '1'
        """
        key = f"{tour_code}|{start_date}"
        found = self.session.get(FkTourId, key)
        if found is None:
            found = FkTourId(key=key, fk_id=str(self._take("fk")))
            self.session.add(found)
        return found.fk_id

    def hotel_id(self, site: str, name: Optional[str]) -> str:
        """The id of one hotel on one site, or ``"0"`` where it has no name.

        Example:
            >>> db = Database(path=None)
            >>> db.hotel_id("tahagasht", "Baiyoke Sky Hotel"), db.hotel_id("tahagasht", "baiyoke sky hotel")
            ('1', '1')
            >>> db.hotel_id("tahagasht", None)
            '0'
        """
        if not name or not name.strip():
            return "0"
        key = hotel_key(site, name)
        found = self.session.get(HotelId, key)
        if found is None:
            found = HotelId(key=key, hotel_id=str(self._take("hotel")))
            self.session.add(found)
        return found.hotel_id

    def row_id(self, key: str) -> str:
        """The id of one row of output, the same on every night.

        Example:
            >>> db = Database(path=None)
            >>> db.row_id("leg|1|1"), db.row_id("leg|1|2"), db.row_id("leg|1|1")
            ('1', '2', '1')
        """
        found = self.session.get(RowId, key)
        if found is None:
            found = RowId(key=key, row_id=str(self._take("row")))
            self.session.add(found)
        return found.row_id

    # --- a night's run -------------------------------------------------

    def start_run(self) -> int:
        """Open tonight's run; every change recorded from here belongs to it."""
        run = Run(started_at=datetime.now())
        self.session.add(run)
        self.session.flush()
        self.run = run.id
        return run.id

    def finish_run(self) -> None:
        """Close tonight's run, which is what makes its changes the ones ``changed=true`` sends."""
        self.session.execute(
            update(Run).where(Run.id == self.run).values(finished_at=datetime.now())
        )
        self.save()

    def _last_finished_run(self) -> Optional[int]:
        return self.session.scalar(
            select(Run.id).where(Run.finished_at.is_not(None)).order_by(Run.id.desc())
        )

    def unchanged(self, url: str, content_hash: Optional[str]) -> bool:
        """Whether this page is exactly what it was the last time we read it."""
        if not content_hash:
            return False
        page = self.session.get(Page, url)
        return page is not None and page.content_hash == content_hash

    def worth_skipping(self, url: str, content_hash: Optional[str]) -> bool:
        """Whether an unchanged page can be left unread tonight.

        Not when any of its records states no day: "مهر ماه" is only retired by
        reading the page again, so skipping it would keep it past its month.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> db.remember_page("https://x.com/tour/1", "abc")
            >>> _ = db.store_tour("31540001", [], [{"api_id": "a", "start_date": ""}])
            >>> db.worth_skipping("https://x.com/tour/1", "abc")
            False
            >>> _ = db.store_tour("31540001", [], [{"api_id": "a", "start_date": "1405-08-01"}])
            >>> db.worth_skipping("https://x.com/tour/1", "abc")
            True
        """
        if not self.unchanged(url, content_hash):
            return False
        code = self.known_tour_code(url) or ""
        return not any(
            self.session.scalar(select(exists().where(table.tour_code == code,
                                                      table.start_date == "")))
            for table in (Package, Tour)
        )

    def remember_page(self, url: str, content_hash: Optional[str]) -> None:
        """Note that this page was seen tonight, and what it looked like.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> db.remember_page("https://x.com/tour/1", "abc")
            >>> db.unchanged("https://x.com/tour/1", "abc"), db.unchanged("https://x.com/tour/1", "def")
            (True, False)
        """
        page = self.session.get(Page, url)
        if page is None:
            page = Page(url=url, tour_code=self.tour_code(url), last_seen_run=self.run)
            self.session.add(page)
        page.content_hash = content_hash or page.content_hash
        page.last_seen_run = self.run
        page.missed = 0

    def store_tour(self, tour_code: str, packages: list[dict], tours: list[dict]) -> int:
        """Replace one tour's records with tonight's, and say how many moved.

        A record whose hash is unchanged keeps the run it last changed on, so
        ``changed=true`` does not send it again. A record the page no longer
        offers is deleted.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> tour = {"api_id": "31540001-1-14050701", "tour_code": "31540001",
            ...         "start_date": "1405-07-01", "price": 39}
            >>> db.store_tour("31540001", [], [tour])
            1
            >>> db.store_tour("31540001", [], [tour])
            0
            >>> db.store_tour("31540001", [], [dict(tour, price=45)])
            1
            >>> db.store_tour("31540001", [], [])
            0
            >>> db.catalogue()[1]
            []
        """
        moved = 0
        for table, records in ((Package, packages), (Tour, tours)):
            key_column = Package.key if table is Package else Tour.api_id
            keep = {record_key(record) for record in records}
            self.session.execute(
                delete(table).where(table.tour_code == tour_code, key_column.not_in(keep))
            )
            for record in records:
                key = record_key(record)
                fingerprint = record_hash(record)
                row = self.session.get(table, key)
                if row is not None and row.record_hash == fingerprint:
                    continue
                if row is None:
                    row = table(**{"key" if table is Package else "api_id": key})
                    self.session.add(row)
                row.tour_code = tour_code
                row.start_date = as_packed(record.get("start_date")) or ""
                row.record = json.dumps(record, ensure_ascii=False)
                row.record_hash = fingerprint
                row.changed_run = self.run
                moved += 1
        return moved

    def retire(self, domains: list[str], today: Optional[date] = None) -> dict[str, int]:
        """Delete what cannot be bought any more, and count what went.

        Pages of ``domains`` not seen tonight go one night further towards
        :data:`MISSED_NIGHTS`. Pass only the sites that were crawled in full:
        a run over three pages of one site did not look for the rest.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> db.remember_page("https://x.com/tour/1", "a")
            >>> _ = db.store_tour("31540001", [], [{"api_id": "a", "start_date": "1405-06-01"},
            ...                                    {"api_id": "b", "start_date": "1405-08-01"}])
            >>> db.retire(["x.com"], today=date(2026, 9, 22))
            {'departed': 1, 'missing': 0}
            >>> for _ in range(3):
            ...     _ = db.start_run()
            ...     gone = db.retire(["x.com"], today=date(2026, 9, 22))
            >>> gone, db.catalogue()[1], db.known_tour_code("https://x.com/tour/1")
            ({'departed': 0, 'missing': 1}, [], '31540001')
        """
        now = jalali_today(today)
        departed = 0
        for table in (Package, Tour):
            departed += self.session.execute(
                delete(table).where(table.start_date != "", table.start_date < now)
            ).rowcount

        missing = 0
        for page in self.session.scalars(select(Page)).all():
            if page.last_seen_run == self.run or not any(d in page.url for d in domains):
                continue
            page.missed += 1
            if page.missed >= MISSED_NIGHTS:
                for table in (Package, Tour):
                    self.session.execute(delete(table).where(table.tour_code == page.tour_code))
                self.session.delete(page)
                missing += 1
        return {"departed": departed, "missing": missing}

    def save(self) -> None:
        """Commit everything recorded so far; a run stopped after this keeps it."""
        self.session.commit()

    # --- what the API reads --------------------------------------------

    def codes_of(self, sites: Optional[list[str]]) -> Optional[set[str]]:
        """The tour codes belonging to these sites, or ``None`` for all of them.

        The site lives inside the record, not in a column, so it is read back
        out of there — the same place :meth:`agencies` reads it.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [], [{"api_id": "a",
            ...     "source": {"website": "bisanseir.com"}}])
            >>> _ = db.store_tour("2", [], [{"api_id": "b",
            ...     "source": {"website": "vanrosh.com"}}])
            >>> sorted(db.codes_of(["bisanseir.com"]))
            ['1']
            >>> db.codes_of(None) is None
            True
        """
        if not sites:
            return None
        website = func.json_extract(Tour.record, "$.source.website")
        return set(self.session.scalars(
            select(Tour.tour_code).where(website.in_(sites))
        ))

    def catalogue(self, sites: Optional[list[str]] = None) -> tuple[list[dict], list[dict]]:
        """Every live package and tour, for ``changed=false``.

        ``sites`` narrows it to those websites, which is how one site is put in
        front of the client while the others are still being worked on.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [{"tour_code": "1", "number_package": "1"}],
            ...                   [{"api_id": "a", "source": {"website": "bisanseir.com"}}])
            >>> _ = db.store_tour("2", [{"tour_code": "2", "number_package": "1"}],
            ...                   [{"api_id": "b", "source": {"website": "vanrosh.com"}}])
            >>> [len(part) for part in db.catalogue()]
            [2, 2]
            >>> [len(part) for part in db.catalogue(["bisanseir.com"])]
            [1, 1]
        """
        codes = self.codes_of(sites)
        packages = select(Package.record)
        tours = select(Tour.record)
        if codes is not None:
            packages = packages.where(Package.tour_code.in_(codes))
            tours = tours.where(Tour.tour_code.in_(codes))
        return (
            [json.loads(r) for r in self.session.scalars(packages)],
            [json.loads(r) for r in self.session.scalars(tours)],
        )

    def changes(self, run: Optional[int] = None,
                sites: Optional[list[str]] = None) -> tuple[list[dict], list[dict]]:
        """The packages and tours the last finished run found new or different, for ``changed=true``.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [], [{"api_id": "a"}]); db.finish_run()
            >>> _ = db.start_run()
            >>> _ = db.store_tour("2", [], [{"api_id": "b"}]); db.finish_run()
            >>> [t["api_id"] for t in db.changes()[1]], len(db.catalogue()[1])
            (['b'], 2)
        """
        last = run or self._last_finished_run()
        codes = self.codes_of(sites)
        packages = select(Package.record).where(Package.changed_run == last)
        tours = select(Tour.record).where(Tour.changed_run == last)
        if codes is not None:
            packages = packages.where(Package.tour_code.in_(codes))
            tours = tours.where(Tour.tour_code.in_(codes))
        return (
            [json.loads(r) for r in self.session.scalars(packages)],
            [json.loads(r) for r in self.session.scalars(tours)],
        )

    def packages_for(
        self, tour_code: str, start_date: Optional[str] = None,
        end_date: Optional[str] = None, changed: bool = False,
        sites: Optional[list[str]] = None,
    ) -> list[dict]:
        """One tour's packages, narrowed to a window of departure dates, both ends inclusive.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [{"tour_code": "1", "start_date": d, "number_package": "1"}
            ...                         for d in ("14050619", "14050625")], [])
            >>> [p["start_date"] for p in db.packages_for("1", "1405-06-20")]
            ['14050625']
        """
        codes = self.codes_of(sites)
        if codes is not None and tour_code not in codes:
            return []
        query = select(Package.record).where(Package.tour_code == tour_code)
        if changed:
            query = query.where(Package.changed_run == self._last_finished_run())
        since, until = as_packed(start_date), as_packed(end_date)
        if since or until:
            query = query.where(Package.start_date != "")
        if since:
            query = query.where(Package.start_date >= since)
        if until:
            query = query.where(Package.start_date <= until)
        return [json.loads(r) for r in self.session.scalars(query.order_by(Package.start_date))]

    # --- what the report reads -----------------------------------------

    def agencies(self) -> list[dict]:
        """One row per site in the catalogue: its name, and how much it holds.

        The site is read out of the record rather than kept in a column of its
        own, because the record is what the client is served and the two can
        then never disagree.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> tour = {"api_id": "a", "agency_name": "نهال گشت",
            ...         "source": {"website": "nahalgasht.com"}}
            >>> _ = db.store_tour("1", [{"tour_code": "1", "number_package": "1"}], [tour])
            >>> db.agencies()
            [{'website': 'nahalgasht.com', 'name': 'نهال گشت', 'tours': 1, 'packages': 1}]
        """
        website = func.json_extract(Tour.record, "$.source.website")
        name = func.json_extract(Tour.record, "$.agency_name")
        sites = self.session.execute(
            select(website, func.min(name), func.count()).group_by(website)
        ).all()
        rows = self.session.execute(
            select(Tour.tour_code, func.min(website)).group_by(Tour.tour_code)
        ).all()
        of_code = {code: site for code, site in rows}
        counted: dict[str, int] = {}
        for code, number in self.session.execute(
            select(Package.tour_code, func.count()).group_by(Package.tour_code)
        ).all():
            counted[of_code.get(code, "")] = counted.get(of_code.get(code, ""), 0) + number
        return [
            {"website": site or "", "name": title or "", "tours": number,
             "packages": counted.get(site or "", 0)}
            for site, title, number in sorted(sites, key=lambda row: -row[2])
        ]

    def tours_of(self, website: str) -> list[dict]:
        """Every live tour of one site, newest departure last.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [], [{"api_id": "a", "start_date": "1405-07-02",
            ...     "source": {"website": "x.com"}}])
            >>> [t["start_date"] for t in db.tours_of("x.com")]
            ['1405-07-02']
            >>> db.tours_of("other.com")
            []
        """
        site = func.json_extract(Tour.record, "$.source.website")
        return [
            json.loads(record) for record in self.session.scalars(
                select(Tour.record).where(site == website).order_by(Tour.start_date)
            )
        ]

    def tour_named(self, api_id: str) -> Optional[dict]:
        """One tour by the id the report links to, or ``None`` when it has gone.

        Example:
            >>> db = Database(path=None); _ = db.start_run()
            >>> _ = db.store_tour("1", [], [{"api_id": "a-1", "tour_name": "تور کیش"}])
            >>> db.tour_named("a-1")["tour_name"]
            'تور کیش'
            >>> db.tour_named("gone") is None
            True
        """
        found = self.session.get(Tour, api_id)
        return json.loads(found.record) if found else None

    # --- moving over from the JSON files -------------------------------

    def import_numbers(self, ids_path: Path) -> int:
        """Take over the numbers ``data/ids.json`` handed out, once, and say how many codes came.

        Only the numbers: the client already holds them, and a tour that came
        back under a new code would be counted twice. The catalogue is not
        taken over — the first night crawls everything, and a record carried
        in with no page behind it could never be retired.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     path = Path(folder) / "ids.json"
            ...     _ = path.write_text(json.dumps({"tour_code": {"https://x.com/tour/1": "31540007"},
            ...                                     "next": {"tour": 8}}))
            ...     db = Database(path=None)
            ...     db.import_numbers(path)
            1
            >>> db.tour_code("https://x.com/tour/1"), db.tour_code("https://x.com/tour/2")
            ('31540007', '31540008')
        """
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        for key, code in ids.get("tour_code", {}).items():
            self.session.merge(TourCode(key=key, code=code))
        for key, number in ids.get("hotel_id", {}).items():
            self.session.merge(HotelId(key=key, hotel_id=number))
        for key, number in ids.get("fk_tour_id", {}).items():
            self.session.merge(FkTourId(key=key, fk_id=number))
        for key, number in ids.get("row_id", {}).items():
            self.session.merge(RowId(key=key, row_id=number))
        for name, number in ids.get("next", {}).items():
            self.session.merge(Counter(name=name, next=number))
        self.save()
        return len(ids.get("tour_code", {}))
