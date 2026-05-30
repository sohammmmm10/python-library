from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def _delay_for_attempt(
    attempt: int, *, backoff_seconds: float, max_backoff_seconds: float, jitter: bool
) -> float:
    delay = min(max_backoff_seconds, backoff_seconds * (2 ** (attempt - 1)))
    if jitter:
        delay *= random.uniform(0.75, 1.25)
    return max(0.0, delay)


def run_with_retry(
    operation: Callable[[], T],
    *,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    jitter: bool,
    retry_on: tuple[type[Exception], ...],
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    max_attempts = max(1, retries + 1)
    attempt = 0
    last_error: Exception | None = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return operation()
        except retry_on as exc:  # type: ignore[misc]
            last_error = exc
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_attempts:
                raise
            time.sleep(
                _delay_for_attempt(
                    attempt,
                    backoff_seconds=backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    jitter=jitter,
                )
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("run_with_retry reached an unexpected state.")


async def arun_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    jitter: bool,
    retry_on: tuple[type[Exception], ...],
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    max_attempts = max(1, retries + 1)
    attempt = 0
    last_error: Exception | None = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return await operation()
        except retry_on as exc:  # type: ignore[misc]
            last_error = exc
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(
                _delay_for_attempt(
                    attempt,
                    backoff_seconds=backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    jitter=jitter,
                )
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("arun_with_retry reached an unexpected state.")
