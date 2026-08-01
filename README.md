# AI-Powered Web Crawler

A production-oriented Python crawler for discovering, filtering, extracting, and periodically refreshing structured information from websites with different technical characteristics.

The system is designed around explicit pipeline stages rather than a one-size-fits-all crawl. It profiles a target first, respects crawl controls, and chooses an appropriate strategy for static sites, JavaScript-rendered pages, sitemap-led discovery, and blocked targets.

## Goals

- Profile a target website before any crawl work begins.
- Discover URLs from sitemaps and crawlable pages.
- Filter discovered URLs into a focused, extensible work set.
- Extract page content, metadata, and structured data reliably.
- Convert raw content into validated structured records through an LLM.
- Refresh the resulting dataset on a schedule with observable, recoverable execution.

## Architecture

```text
Target domain
    │
    ▼
1. Profiler ──► crawl strategy and discovery inputs
    │
    ▼
2. URL filtering ──► valuable, deduplicated URLs
    │
    ▼
3. Extraction ──► raw page content and metadata
    │
    ▼
4. Schema ──► validated normalized records
    │
    ▼
5. LLM pipeline ──► structured enriched output
    │
    ▼
6. Automation ──► scheduled refresh, logs, retries, recovery
```

## Project layout

```text
crawler/
├── config.py                 # environment-backed settings and shared constants
├── profiler/                 # step 1: reconnaissance and crawl strategy
├── filtering/                # step 2: URL selection rules
├── extraction/               # step 3: page-content extraction
├── schema/                   # step 4: validated output models
├── llm/                      # step 5: prompts and LLM processing
├── automation/               # step 6: scheduling and pipeline execution
└── utils/                    # shared concerns, including logging
scripts/                      # command-line entry points
data/                         # generated local data; excluded from Git
tests/                        # tests grouped by pipeline stage
```

## Current status

Step 1 is implemented. `crawler.profiler` builds a `SiteProfile` containing:

- reachability, final URL, and HTTP status;
- `robots.txt` availability, permissions, crawl delay, and sitemap declarations;
- sitemap existence, type, URL count, and sample URLs;
- static/dynamic rendering signals, framework hints, and link counts;
- Cloudflare, CAPTCHA, and HTTP blocking signals;
- basic server/CMS technology hints; and
- a recommended strategy: `sitemap`, `static-crawl`, `dynamic-crawl`, `blocked`, or `unreachable`.

Steps 2–6 have their package boundaries in place and will be implemented incrementally.

## Requirements

- Python 3.10 or newer
- Chromium browser binaries for rendering-enabled profiling

Install the Python packages and browser runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
playwright install chromium
```

## Configuration

Runtime configuration is read from environment variables in `crawler/config.py`.
Keep secrets in the untracked `.env` file; never commit API keys.

```dotenv
OPENAI_API_KEY=your_key_here
CRAWLER_TIMEOUT_SECONDS=15
CRAWLER_USER_AGENT="TourCrawler/0.1 (+crawler; contact: you@example.com)"
```

The profiler does not require an OpenAI key. Before later LLM commands are introduced, load variables into the shell (or use your deployment platform's secret manager):

```bash
set -a
source .env
set +a
```

## Usage

Profile a site without browser rendering (fast heuristic):

```bash
python3 scripts/run_profiler.py --no-render https://example.com
```

Profile with Playwright rendering enabled (detects client-side generated content more reliably):

```bash
python3 scripts/run_profiler.py https://example.com
```

The command prints a JSON `SiteProfile` and its `recommended_strategy`.

## Development principles

- Respect `robots.txt`, crawl-delay directives, HTTP failures, and site terms before collecting data.
- Treat site behavior as variable; do not assume a common HTML structure or rendering model.
- Keep each stage independently testable with typed inputs and outputs.
- Prefer deterministic filtering and extraction first; add AI classification only as an optional extension.
- Validate LLM outputs against schema models and retain raw extraction data for traceability.
- Use bounded retries, timeouts, structured logs, and idempotent writes in automated jobs.
- Do not implement anti-bot bypasses. Flag blocked targets for review instead.

## Incremental roadmap

1. Add URL discovery adapters and deterministic URL filtering rules.
2. Implement HTTP and Playwright extraction with normalized page records.
3. Define validated output models and normalization rules.
4. Add LLM prompts, structured responses, retries, and evaluation fixtures.
5. Build scheduling, persistent job state, monitoring, and failure recovery.
6. Expand tests for static, dynamic, sitemap, restricted, and malformed-site cases.

## Testing

Tests belong under `tests/`, grouped by pipeline stage. Run the suite with:

```bash
python3 -m pytest
```

Add fixtures for real-world variations without placing private data or credentials in the repository.
