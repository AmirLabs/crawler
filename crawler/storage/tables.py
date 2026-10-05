"""The tables the crawl keeps between nights, in one SQLite file.

Two kinds of table live here, and they are treated differently on purpose:

* **Numbers** — ``tour_codes``, ``hotel_ids``, ``fk_tour_ids``, ``row_ids``,
  ``counters``.
  Never deleted. A site revives a dead page (avisatravel reused a نوروز 1404
  page for مهر 1405), and a revived URL must get back the code the client
  already holds, or they count one tour twice.
* **The catalogue** — ``pages``, ``tours``, ``packages``. What is live tonight.
  A tour that has gone is deleted, so nothing that reads these has to remember
  to filter the dead out.

``runs`` says which night a record last changed on, which is all
``changed=true`` asks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TourCode(Base):
    """One tour's page, minus the date asked for, and the code it was given.

    A tour sold under several addresses holds one row per address, all naming
    the same code, so the code survives any one of them departing.
    """

    __tablename__ = "tour_codes"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    code: Mapped[str] = mapped_column(String, index=True)


class HotelId(Base):
    """One hotel on one site, and its number."""

    __tablename__ = "hotel_ids"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    hotel_id: Mapped[str] = mapped_column(String, unique=True)


class FkTourId(Base):
    """One tour on one departure date, and its number."""

    __tablename__ = "fk_tour_ids"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    fk_id: Mapped[str] = mapped_column(String, unique=True)


class RowId(Base):
    """One row of output — a package, a leg of a journey — and its number."""

    __tablename__ = "row_ids"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    row_id: Mapped[str] = mapped_column(String, unique=True)


class Counter(Base):
    """The next number to hand out, one row per kind of number."""

    __tablename__ = "counters"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    next: Mapped[int] = mapped_column(Integer, default=1)


class Run(Base):
    """One night's crawl."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column()
    finished_at: Mapped[Optional[datetime]] = mapped_column(default=None)


class Page(Base):
    """A tour page as last seen: what it looked like, and how many nights it has been missing."""

    __tablename__ = "pages"

    url: Mapped[str] = mapped_column(String, primary_key=True)
    tour_code: Mapped[str] = mapped_column(String, index=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String, default=None)
    last_seen_run: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    missed: Mapped[int] = mapped_column(Integer, default=0)


class Tour(Base):
    """One record of the second structure: a tour on one departure, with its starting price."""

    __tablename__ = "tours"

    api_id: Mapped[str] = mapped_column(String, primary_key=True)
    tour_code: Mapped[str] = mapped_column(String, index=True)
    start_date: Mapped[str] = mapped_column(String, default="")
    record: Mapped[str] = mapped_column(Text)
    record_hash: Mapped[str] = mapped_column(String)
    changed_run: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)


class Package(Base):
    """One record of the first structure: a hotel and room on one date."""

    __tablename__ = "packages"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    tour_code: Mapped[str] = mapped_column(String)
    start_date: Mapped[str] = mapped_column(String, default="")
    record: Mapped[str] = mapped_column(Text)
    record_hash: Mapped[str] = mapped_column(String)
    changed_run: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)

    __table_args__ = (Index("ix_packages_tour_date", "tour_code", "start_date"),)
