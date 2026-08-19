"""Sending a question to the model, with the answer's shape guaranteed.

Two things shape this module.

**Batching.** The instructions and worked examples cost far more tokens than a
tour name does — around 750 against 30 — so asking about one name per call pays
for the instructions once per name. Fifty names in one call pays for them once
per fifty, which is most of the difference between two cents and fifty for the
same work. Batching is the caller's job; what this provides is a call that can
carry a batch.

**A typed answer.** ``with_structured_output`` constrains the reply to a
Pydantic model, so a malformed or half-written answer cannot reach us: the shape
is the API's problem. What it cannot check is whether the content is *true* —
a reply naming Dubai for a tour to Kish is perfectly well-formed — and that
check belongs to whoever asked the question.

Nothing here knows what a tour is. It sends messages and returns a parsed
model; :mod:`crawler.llm.tour_names` is what knows the question.
"""

from __future__ import annotations

import warnings
from typing import Optional, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from crawler.config import LLM_MODEL, OPENAI_API_KEY

AnswerModel = TypeVar("AnswerModel", bound=BaseModel)

# Long enough for a batch of fifty short answers, and a ceiling rather than a
# target: a reply cut off halfway is unusable, and headroom we do not use costs
# nothing.
DEFAULT_MAX_TOKENS = 4000

# Naming a place in a title has one right answer, so there is nothing to gain
# from letting the model vary.
TEMPERATURE = 0.0

# How many times the model is asked before giving up. Rate limits and gateway
# errors are worth another try; a malformed request would fail the same way.
MAX_RETRIES = 2

# langchain-openai passes the parsed model back through a field typed as
# ``None``, and Pydantic warns about it on every call. The warning says nothing
# about our data and would fill a nightly crawl's log.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic.main",
)


class LLMUnavailable(RuntimeError):
    """Raised when there is no key, so a caller can carry on without the model.

    The deterministic rules fill most fields on their own, and a crawl that
    stops because an API key is missing is worse than one that runs with a few
    fields empty.
    """


def is_available() -> bool:
    """Whether the model can be reached at all.

    Example:
        >>> is_available()
        True
    """
    return bool(OPENAI_API_KEY)


def ask(
    messages: list[tuple[str, str]],
    answer_model: type[AnswerModel],
    *,
    model: str = LLM_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
) -> AnswerModel:
    """Ask one question and return the answer as ``answer_model``.

    Raises :class:`LLMUnavailable` when no key is configured. Anything the
    model or the network raises is left to propagate: a caller crawling many
    batches decides whether one failure is worth stopping for.

    Example:
        >>> from pydantic import BaseModel
        >>> class Reply(BaseModel):
        ...     answer: str
        >>> ask([("human", "reply with the single word ok")], Reply).answer.lower()[:2]
        'ok'
    """
    if not OPENAI_API_KEY:
        raise LLMUnavailable("OPENAI_API_KEY is not set")

    chat = ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        max_completion_tokens=max_tokens,
        timeout=timeout,
        max_retries=MAX_RETRIES,
    )
    structured = chat.with_structured_output(answer_model, method="json_schema", strict=True)
    return structured.invoke(messages)
