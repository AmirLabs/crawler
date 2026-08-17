"""Asking for a page again when the first attempt failed for a passing reason.

A crawl over seven sites makes hundreds of requests, and some of them fail for
reasons that have nothing to do with the page: a server busy for a moment, a
network hiccup, a gateway restarting. liyaseir.com dropped one request during
an audit run and answered normally minutes later — without a second attempt,
that single timeout removed the entire site from the day's output and looked
exactly like a bug in our own rules.

Retrying is only right for failures that might not happen again, so the two
kinds are kept apart:

* **Worth another try** — timeouts, connection errors, and the 5xx statuses a
  server returns when it is overloaded or restarting.
* **Not worth it** — 404, because the page will still be missing, and 403,
  because the site has said no and asking harder is how a crawler earns a
  permanent block.

Waits double between attempts. A server that just failed under load is made
worse by an immediate retry, and the point is to give it room rather than to
ask faster.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from crawler.config import HTTP_MAX_ATTEMPTS, HTTP_RETRY_BASE_DELAY

# Statuses that mean "not now" rather than "no". 429 is included because the
# right answer to being asked to slow down is to wait longer, which is what
# the doubling delay does.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Failures that happen before any status arrives. Every one of these is about
# the connection rather than the page.
RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


def is_retryable_status(status_code: int) -> bool:
    """Whether a status is worth asking about again.

    Example:
        >>> is_retryable_status(503), is_retryable_status(404), is_retryable_status(403)
        (True, False, False)
    """
    return status_code in RETRY_STATUSES


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    attempts: int = HTTP_MAX_ATTEMPTS,
    base_delay: float = HTTP_RETRY_BASE_DELAY,
    **kwargs: Any,
) -> tuple[Optional[httpx.Response], Optional[str]]:
    """GET ``url``, trying again when the failure looks like a passing one.

    Returns the response and ``None``, or ``None`` and a description of what
    went wrong. The reason is returned rather than raised so a caller crawling
    many pages can record why one of them is missing and carry on.

    A response that arrives is returned even when its status is an error, as
    long as retrying would not help — the caller decides what a 404 means.

    Example:
        >>> async with httpx.AsyncClient() as client:
        ...     response, error = await get_with_retry(client, "https://example.com")
        >>> response.status_code, error
        (200, None)
    """
    last_error = "no attempt was made"

    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = await client.get(url, **kwargs)
        except RETRY_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}".strip()
        except Exception as exc:  # anything else will not improve on a retry
            return None, f"{type(exc).__name__}: {exc}".strip()
        else:
            if not is_retryable_status(response.status_code):
                return response, None
            last_error = f"status {response.status_code}"

        if attempt < attempts:
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    return None, f"{last_error} (after {attempts} attempts)"
