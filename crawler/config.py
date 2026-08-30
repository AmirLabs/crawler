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

# Each wait is multiplied by a random factor in this range, so the rate holds
# without the perfectly regular interval that is itself a bot signature.
CRAWL_DELAY_JITTER = (0.5, 1.5)

# One timeout in hundreds of requests is ordinary, so a page is asked for again
# before it is given up on. The wait doubles each time.
HTTP_MAX_ATTEMPTS = int(os.getenv("CRAWLER_HTTP_MAX_ATTEMPTS", "3"))
HTTP_RETRY_BASE_DELAY = float(os.getenv("CRAWLER_HTTP_RETRY_BASE_DELAY", "2.0"))

# lachinseir.com builds its price table only for what it reads as a browser,
# and refuses Googlebot too; robots.txt and the crawl delay are still obeyed.
USER_AGENT = os.getenv(
    "CRAWLER_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Naming a destination in a short Persian title is a small job, and answers are
# cached, so the cheap model does it.
LLM_MODEL = os.getenv("CRAWLER_LLM_MODEL", "gpt-4o-mini")

# OpenAI direct; only worth setting if the calls are routed through something else.
LLM_BASE_URL = os.getenv("CRAWLER_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# A ceiling, not a cost per page: the wait ends as soon as the placeholders are
# gone, and eight seconds cleared every alefbatour tour page tested.
RENDER_SETTLE_SECONDS = float(os.getenv("CRAWLER_RENDER_SETTLE_SECONDS", "8.0"))
