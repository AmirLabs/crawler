"""Manual profiler check against a real target site.

Prints the full SiteProfile to the screen. Nothing is written to disk — this is
a look-and-see script, not an automated test.

Usage:
    python tests/manual_profile_check.py                      # default target
    python tests/manual_profile_check.py https://example.com  # any domain
    python tests/manual_profile_check.py --no-render <domain> # skip the browser
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.profiler.profiler import profile_site

DEFAULT_TARGET = "https://nahalgasht.com"


def _summarize(profile) -> None:
    """Print the handful of fields that drive later pipeline stages."""
    r, s, rd, ab, t = (
        profile.robots,
        profile.sitemap,
        profile.rendering,
        profile.anti_bot,
        profile.tech,
    )

    print("\n" + "-" * 70)
    print("WHAT THIS MEANS FOR THE PIPELINE")
    print("-" * 70)
    print(f"  strategy          : {profile.recommended_strategy}")
    print(f"  reachable         : {profile.reachable} (HTTP {profile.status_code})")
    print(f"  robots allows     : {r.allows_crawling}   crawl delay: {r.crawl_delay}")

    pages = f"{s.url_count}{'+' if s.url_count_is_partial else ''}"
    print(f"  sitemap           : {s.exists} type={s.type} pages={pages}")
    if s.child_sitemap_count:
        print(f"                      {s.child_sitemap_count} child sitemaps")

    print(f"  needs browser     : {rd.needs_playwright} ({rd.type}, {rd.js_framework_hint})")
    print(f"  links raw/rendered: {rd.links_in_raw_html} / {rd.links_after_render}")
    print(f"  blocked           : {ab.blocks_bots}  captcha={ab.captcha_detected} cf={ab.cloudflare}")
    print(f"  cms               : {t.cms}")
    print(f"  json api          : {t.has_json_api} {t.json_api_url or ''}")

    if s.sample_urls:
        print("\n  sample URLs from the sitemap (do any look like tour pages?):")
        for u in s.sample_urls:
            print(f"    - {u}")
    if ab.notes:
        print(f"\n  anti-bot notes: {ab.notes}")
    if rd.notes:
        print(f"  rendering notes: {rd.notes}")


async def _main(domain: str, render: bool) -> None:
    print("=" * 70)
    print(f"Profiling: {domain}   (render={render})")
    print("=" * 70)

    profile = await profile_site(domain, render=render)

    print(profile.to_json())
    _summarize(profile)


def main() -> None:
    args = sys.argv[1:]
    render = True
    if "--no-render" in args:
        render = False
        args.remove("--no-render")

    domain = args[0] if args else DEFAULT_TARGET
    asyncio.run(_main(domain, render))


if __name__ == "__main__":
    main()
