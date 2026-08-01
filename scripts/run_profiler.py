"""Command-line runner for the Site Profiler.

Usage:
    python scripts/run_profiler.py https://example.com [more-domains...]
    python scripts/run_profiler.py --no-render https://example.com

Prints a JSON profile for each domain. Domain-agnostic: pass any site.
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path

# Make direct execution from the ``scripts`` directory reliable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.profiler.profiler import profile_site


async def _main(domains: list[str], render: bool) -> None:
    for domain in domains:
        print(f"\n{'=' * 70}\nProfiling: {domain}\n{'=' * 70}")
        profile = await profile_site(domain, render=render)
        print(profile.to_json())
        print(f"\n>>> recommended_strategy: {profile.recommended_strategy}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    render = True
    if "--no-render" in args:
        render = False
        args.remove("--no-render")

    if not args:
        print("usage: python scripts/run_profiler.py [--no-render] <domain> [domain...]")
        sys.exit(1)

    asyncio.run(_main(args, render))


if __name__ == "__main__":
    main()
