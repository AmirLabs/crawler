"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CRAWLER_TIMEOUT_SECONDS", "15"))
USER_AGENT = os.getenv(
    "CRAWLER_USER_AGENT", "TourCrawler/0.1 (+crawler; contact: you@example.com)"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
