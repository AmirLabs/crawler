"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CRAWLER_TIMEOUT_SECONDS", "15"))

# How long to wait between two requests to the same host.

DEFAULT_CRAWL_DELAY_SECONDS = float(os.getenv("CRAWLER_CRAWL_DELAY_SECONDS", "1.0"))
MIN_CRAWL_DELAY_SECONDS = float(os.getenv("CRAWLER_MIN_CRAWL_DELAY_SECONDS", "0.5"))
MAX_CRAWL_DELAY_SECONDS = float(os.getenv("CRAWLER_MAX_CRAWL_DELAY_SECONDS", "10.0"))

# Each wait is multiplied by a random factor in this range. A perfectly regular
# interval is itself a bot signature; the jitter keeps the average rate while
# removing that pattern.
CRAWL_DELAY_JITTER = (0.5, 1.5)
USER_AGENT = os.getenv(
    "CRAWLER_USER_AGENT", "TourCrawler/0.1 (+crawler; contact: you@example.com)"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
