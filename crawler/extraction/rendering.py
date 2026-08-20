"""Telling a page that finished building from one that never started.

Some of these sites serve a template rather than a document. alefbatour.com's
tour page arrives with its departure dates written into the HTML and its hotel
list left as ``{{ accommodation.name }}`` — a placeholder its JavaScript fills
in later. Read without a browser, the page looks complete and simply has no
hotels, which is why every alefbatour row came out with an empty hotel name.

The profiler already asks whether a *site* needs JavaScript, and answers by
counting links on its home page. That answer is right about alefbatour's home
page and wrong about its tour pages: part of a page can need JavaScript while
the rest does not. So the question is asked again here, per page, of the markup
actually in hand.

The test is the leftover template syntax itself. ``{{ ... }}`` is what Vue,
Angular, Mustache and Handlebars all leave behind when their script has not
run, so it needs no framework detection. Measured over 60 saved pages it fires
on every alefbatour, avisatravel, nahalgasht and tahagasht page, and on no
bisanseir or liyaseir page — which matches exactly which sites hide content
behind JavaScript and which do not.

A template stays in the document after its data is filled in, so this test
says "this page has client-side content", not "this page failed to render".
That is the right question to ask of markup fetched over plain HTTP, where
nothing has run at all.
"""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser

# A moustache placeholder: an expression between double braces. Kept tight
# enough not to match a stray "{{" in prose — the body may not contain another
# brace, and the run has a length limit.
_PLACEHOLDER = re.compile(r"\{\{\s*[\w.$][^{}]{0,80}\}\}")

# Below this many placeholders the page is treated as built. One stray
# expression is usually a widget that failed rather than the content missing,
# and re-fetching every such page through a browser costs more than it returns.
MIN_PLACEHOLDERS = 2


def markup_without_scripts(html: str) -> str:
    """The page's markup with executable code removed, templates kept.

    ``<script>`` is dropped because a framework's own source legitimately
    contains braces. ``<template>`` is deliberately **kept**: it is the thing
    being looked for. A page that ships a ``<template>`` full of placeholders
    is a page whose content is built in the browser, which is the whole
    question here.

    Example:
        >>> markup_without_scripts('<body><script>var a={{}}</script>'
        ...                        '<template><p>{{ x.y }}</p></template></body>')
        '<body><template><p>{{ x.y }}</p></template></body>'
    """
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript"):
        node.decompose()
    body = tree.body
    return (body.html or "") if body is not None else ""


def count_placeholders(html: str) -> int:
    """How many template expressions the markup carries outside its scripts.

    Example:
        >>> count_placeholders('<body><template><p>{{ accommodation.name }}</p>'
        ...                    '<span>{{ tour.title }}</span></template></body>')
        2
        >>> count_placeholders('<body><p>هتل باکسوز</p></body>')
        0
    """
    return len(_PLACEHOLDER.findall(markup_without_scripts(html)))


def needs_rendering(html: str) -> bool:
    """Whether this page builds part of itself in the browser.

    Note what this does *not* claim: that the page failed to render. A template
    stays in the document after its data is filled in, so this cannot tell a
    built page from an unbuilt one — and does not need to. It is asked of
    markup fetched over plain HTTP, where nothing has run, so "this page has
    client-side content" and "we are missing it" are the same statement.

    Example:
        >>> needs_rendering('<body><template><div>{{ accommodation.name }}</div>'
        ...                 '<div>{{ accommodation.stars }}</div></template></body>')
        True
        >>> needs_rendering('<body><div>هتل باکسوز ارمنستان</div></body>')
        False
    """
    return count_placeholders(html) >= MIN_PLACEHOLDERS


## Waiting for a rendered page
##
## Two finish lines were tried and dropped, both worth recording.
##
## *Navigating with ``wait_until="networkidle"``* never completes on these
## pages and failed every alefbatour fetch with a timeout — the page keeps a
## connection open, so the network is never idle by that definition.
##
## *Waiting for the document to stop changing* fires too early. alefbatour
## paints its dates at about 3 seconds, goes quiet, and only fetches its hotels
## at 5. A 900ms quiet window called the page finished at 3 and we saved a copy
## with no hotels in it.
##
## What works is to navigate on ``domcontentloaded`` — which always arrives —
## and then wait for the network to settle *as a bounded, optional step*. The
## wait is allowed to time out: whatever has rendered by then is kept, and the
## timeout is recorded on the page rather than raised.
