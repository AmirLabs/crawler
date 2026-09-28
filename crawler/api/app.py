"""The two routes the client calls, over the database the nightly run keeps.

Their developer asked for the split and our two structures already were it: a
*tour* is one departure with a starting price, a *package* is one hotel and
room on one date. Measured on the run of 2026-09-09, 93 tours carried 2236
packages — about 24 each, and 61 for one of them — which is why sending both
together timed out on his side.

    GET /api/v1/tours
    GET /api/v1/packages?tour_code=31540001&start_date=…&end_date=…

Both answer the envelope their own sample uses. Two rules about it are worth
stating, because getting either wrong is silent:

* **The status and the code always agree.** A failure that answered 200 with an
  error body would be read as an empty success, and their import would clear
  what it already holds.
* **Nothing to send is not a failure.** An unknown tour_code, or a window with
  no departures, answers 200 with an empty list.

Credentials come from the environment, never from the code. Basic auth is only
as safe as the connection under it, so this belongs behind HTTPS: base64 is not
encryption.

Run::

    CRAWLER_API_USER=… CRAWLER_API_PASSWORD=… \\
        .venv/bin/uvicorn crawler.api.app:app --port 8000
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from crawler.storage.database import DEFAULT_DATABASE_PATH, Database

# Where the nightly run's database is; the examples point it at a scratch file.
DATABASE_PATH = DEFAULT_DATABASE_PATH

USER_VARIABLE = "CRAWLER_API_USER"
PASSWORD_VARIABLE = "CRAWLER_API_PASSWORD"

# Which sites the feed carries. Unset means all of them.
SITES_VARIABLE = "CRAWLER_API_SITES"

# Below this a response is smaller than the cost of compressing it.
_GZIP_FROM = 1024

# The documentation FastAPI writes for itself is served by hand below, behind
# the same password. Left to itself it publishes the shape of the whole API to
# anyone who knows the address.
app = FastAPI(title="Tour feed", version="1.0.0", docs_url=None, redoc_url=None,
              openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=_GZIP_FROM)
_basic = HTTPBasic(auto_error=False)


def envelope(data: list[dict], code: int = 200, message: str = "") -> dict:
    """Their wrapper, with the status word following the code.

    Example:
        >>> envelope([{"tour_code": "31540001"}])["message"]
        'Data Fetched Successfully'
        >>> envelope([], 401, "Unauthorized")["status"]
        'error'
    """
    ok = code < 400
    return {
        "status": "success" if ok else "error",
        "code": code,
        "message": message or ("Data Fetched Successfully" if ok else "Error"),
        "data": data,
    }


def wanted_sites() -> Optional[list[str]]:
    """The websites the feed is allowed to carry, or ``None`` for every one.

    The sites are finished one at a time, so the client is given the ones that
    are ready instead of waiting for all ten. Set
    ``CRAWLER_API_SITES=bisanseir.com,vanrosh.com`` to publish those two;
    leaving it out publishes everything, which is what it did before.

    Example:
        >>> os.environ[SITES_VARIABLE] = " bisanseir.com , vanrosh.com "
        >>> wanted_sites()
        ['bisanseir.com', 'vanrosh.com']
        >>> del os.environ[SITES_VARIABLE]
        >>> wanted_sites() is None
        True
    """
    written = os.environ.get(SITES_VARIABLE, "")
    sites = [part.strip() for part in written.split(",") if part.strip()]
    return sites or None


def _matches(given: Optional[str], wanted: str) -> bool:
    """Compare a credential without letting the time taken give it away."""
    return secrets.compare_digest((given or "").encode(), wanted.encode())


def signed_in(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic),
) -> str:
    """Refuse anyone whose credentials do not match the ones in the environment.

    A server started without them refuses everybody. An open feed of ten
    agencies' prices is worse than a feed nobody can reach, and a default
    password is how one becomes the other.
    """
    user = os.environ.get(USER_VARIABLE, "")
    password = os.environ.get(PASSWORD_VARIABLE, "")
    if not user or not password:
        raise HTTPException(status_code=401, detail="Server has no credentials")
    if credentials is None or not (
        _matches(credentials.username, user)
        and _matches(credentials.password, password)
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


@app.get("/api/v1/tours")
def tours(_: str = Depends(signed_in), changed: bool = False) -> dict:
    """Every live tour, one per departure.

    ``changed=true`` narrows it to what that run found new or different. The
    full list stays available so a night that was missed can be caught up.

    Example:
        >>> client = _a_test_client()
        >>> client.get("/api/v1/tours").status_code
        401
        >>> answer = client.get("/api/v1/tours", auth=("them", "s3cret")).json()
        >>> answer["status"], answer["code"], answer["message"]
        ('success', 200, 'Data Fetched Successfully')
    """
    db = Database(DATABASE_PATH)
    try:
        sites = wanted_sites()
        return envelope(
            (db.changes(sites=sites) if changed else db.catalogue(sites))[1]
        )
    finally:
        db.close()


@app.get("/api/v1/packages")
def packages(
    _: str = Depends(signed_in),
    tour_code: str = Query(..., min_length=1),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    changed: bool = False,
) -> dict:
    """The packages of one tour, narrowed to a window of departure dates.

    ``tour_code`` is the code the tours route carries. Dates may be written
    either way the two structures write them.

    Example:
        >>> client = _a_test_client()
        >>> them = ("them", "s3cret")
        >>> client.get("/api/v1/packages", auth=them).status_code
        400
        >>> answer = client.get("/api/v1/packages?tour_code=31540001", auth=them)
        >>> answer.status_code, answer.json()["data"]
        (200, [])
    """
    db = Database(DATABASE_PATH)
    try:
        return envelope(db.packages_for(tour_code, start_date, end_date,
                                        changed, wanted_sites()))
    finally:
        db.close()


@app.exception_handler(HTTPException)
def _handled(request: Request, error: HTTPException) -> JSONResponse:
    """Answer a refusal in their envelope, with the same code in both places."""
    return JSONResponse(
        status_code=error.status_code,
        content=envelope([], error.status_code, str(error.detail)),
        headers=(
            {"WWW-Authenticate": "Basic"} if error.status_code == 401 else None
        ),
    )


@app.get("/openapi.json", include_in_schema=False)
def schema(_: str = Depends(signed_in)) -> dict:
    """The API's own description, for their developer to read instead of asking.

    Example:
        >>> client = _a_test_client()
        >>> client.get("/openapi.json").status_code
        401
        >>> paths = client.get("/openapi.json", auth=("them", "s3cret")).json()
        >>> sorted(paths["paths"])
        ['/api/v1/packages', '/api/v1/tours']
    """
    return app.openapi()


@app.get("/docs", include_in_schema=False)
def documentation(_: str = Depends(signed_in)):
    """The clickable version of the same thing."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Tour feed")


def _wire_report() -> None:
    """Hang the report pages off the same app, behind the same password.

    They are kept out of the schema: that document describes the feed their
    developer imports, and pages to look at are not part of it.
    """
    from crawler.api.report import router

    app.include_router(router, dependencies=[Depends(signed_in)],
                       include_in_schema=False)


_wire_report()


@app.exception_handler(RequestValidationError)
def _bad_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """A missing or malformed parameter is the caller's mistake, not a crash."""
    return JSONResponse(
        status_code=400, content=envelope([], 400, "Bad request")
    )


@app.exception_handler(Exception)
def _unhandled(request: Request, error: Exception) -> JSONResponse:
    """Never let a crash answer 200. Their import reads that as an empty feed."""
    return JSONResponse(status_code=500, content=envelope([], 500, "Server error"))


def _a_test_client():
    """A client of this app with credentials in place, for the examples above."""
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    global DATABASE_PATH
    DATABASE_PATH = Path(tempfile.mkdtemp()) / "crawler.db"
    os.environ[USER_VARIABLE] = "them"
    os.environ[PASSWORD_VARIABLE] = "s3cret"
    return TestClient(app)
