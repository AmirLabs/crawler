# Tour crawler

Reads tour offers from ten Persian travel-agency sites every night and serves
them over two HTTP routes, in the two JSON shapes the client asked for.

Every rule in it is keyed on the *shape* of a page — a price beside a currency
word, a city under a country in a breadcrumb — never on a domain name. Sites
change; the shapes do not. Where a page does not say something, the field is
sent empty rather than guessed: a missing field shows up in the fill rate, a
wrong one is trusted.

## What it produces

Two structures over the same catalogue, at different grains.

| | one record is | routes as |
|---|---|---|
| **tour** | one tour on one departure date, with a starting price | `/api/v1/tours` |
| **package** | one hotel and one room type on one date | `/api/v1/packages` |

Measured on a real run: 93 tours carried 2236 packages, about 24 each. That
ratio is why the two are fetched separately.

## Pipeline

```text
1. profiler/    what a site is and how to fetch it
2. filtering/   which URLs are tour pages
3. extraction/  rows out of a page          ← most of the work lives here
     readers/     text → value (prices, dates, places, countries, transport)
     structure/   page → repeating blocks, rates tables, JSON paths
4. schema/      rows → the client's output models
5. llm/         only what rules cannot reach; answers are checked against the
                source text before they are kept
6. automation/  the nightly run: one at a time, logged, and a memory of what
                last night saw
7. api/         the two routes, over what the last finished run published
```

## Requirements

- Python 3.13
- Chromium, for the pages that are drawn by JavaScript

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
playwright install chromium
```

`playwright install chromium` is not optional. Three of the ten sites leave
their hotel list for JavaScript to fill, and without a browser those pages
yield nothing at all.

## Running it

The nightly run. It takes no arguments on purpose — there is nothing to decide
where a machine starts it, and a flag left wrong is how a night quietly
delivers a tenth of the catalogue.

```bash
.venv/bin/python scripts/run_nightly.py
```

The API, which serves what that run published and never crawls:

```bash
CRAWLER_API_USER=... CRAWLER_API_PASSWORD=... \
    .venv/bin/uvicorn crawler.api.app:app --host 0.0.0.0 --port 8000
```

For trying one site by hand:

```bash
.venv/bin/python scripts/run_client_output.py --domain nahalgasht --pages 3
```

## The two routes

Both answer `{"status", "code", "message", "data"}`, and the status always
agrees with the HTTP code. A failure never answers 200: read as an empty
success, it would clear what the caller already holds. Nothing to send is not
a failure — an unknown tour, or a date window with no departures, answers 200
with an empty list.

```http
GET /api/v1/tours
GET /api/v1/tours?changed=true

GET /api/v1/packages?tour_code=31540001
GET /api/v1/packages?tour_code=31540001&start_date=1405-06-19&end_date=1405-07-19
GET /api/v1/packages?tour_code=31540001&changed=true
```

`tour_code` is the code carried in the tours response. Dates may be written
either way the two structures write them — `1405-06-19` or `14050619`, in
Persian or Latin digits.

`changed=true` narrows a response to what the last run found new or different.
The full lists stay available, so a night that was missed can still be caught
up.

Errors are `400` for a missing or malformed parameter, `401` for wrong or
absent credentials, and `500` when the server itself failed.

`GET /docs` is the API's own documentation page, behind the same password.

## Deployment

### Environment

```dotenv
CRAWLER_API_USER=...          # required; the server refuses everyone without it
CRAWLER_API_PASSWORD=...      # required
CRAWLER_TIMEOUT_SECONDS=15
CRAWLER_USER_AGENT="TourCrawler/1.0 (+contact: you@example.com)"
```

Keep them in the untracked `.env`, or in the platform's own secret store. There
is no default password: a server started without these answers `401` to
everything, which is the safe way to fail.

**Basic auth is only as safe as the connection under it.** Base64 is not
encryption, so this belongs behind HTTPS. On plain HTTP the password travels in
the clear.

### Schedule

One run a night, early enough to have finished before the feed is read:

```cron
0 3 * * *  cd /srv/crawler && .venv/bin/python scripts/run_nightly.py
```

The first night takes hours — ten catalogues, one second of politeness between
requests, and some sites list several hundred tour pages. Later nights are much
shorter: a page that comes back exactly as it was is fetched and then left
unread, and reading a page costs about 11 seconds against 1 to 3 for the fetch.

A second run started while one is going refuses rather than waits, and a run
that failed exits non-zero.

### What lives on disk

```text
data/ids.json        the number each tour is known by. Cannot be rebuilt.
data/seen.json       what last night saw. Losing it costs one slow night.
data/agencies.json   the ten agencies' contact details, written by hand.
data/published/      what the API serves. Replaced atomically at the end of a run.
data/runs/           everything a run printed, kept a fortnight.
logs/                emptied at the start of every run.
```

**`data/ids.json` needs a daily backup that is not on the same disk.** The
client keys their records on `api_id`, which is built on the number in this
file. Nothing can work out which tour held `31540001` yesterday, so losing it
gives every tour a new number and fills their database with duplicates of what
it already has. The 14 dated copies the crawler keeps sit beside it, on the
same disk, and go with it.

`data/` must survive a restart. On a plain server this is free; in a container
it needs a mounted volume, or every id is lost on the first redeploy.

The user the schedule runs as needs write access to `data/`.

### Outbound access

The crawl reaches ten external sites, thousands of requests a night. A host
that blocks outbound HTTP cannot run it.

## Testing

Doctests are the tests. Every public function carries a real output example and
the measurement behind the decision it makes.

```bash
.venv/bin/python scripts/run_doctests.py
```

There is no pytest suite. A change is verified by the doctests plus, for
anything that reads a page, a run against the saved or live pages with the
before and after numbers reported.

## How to work on it

- **Shapes, never sites.** A rule keyed on `nahalgasht.com` is the wrong answer
  even when it works.
- **Empty beats wrong.** Return nothing rather than a guess.
- **Measure, never assume.** Every threshold here came from counting the real
  sites, and several confident assumptions turned out backwards when measured.
- Respect `robots.txt` and crawl delays. Do not implement anti-bot bypasses;
  flag a blocked target for review instead.
