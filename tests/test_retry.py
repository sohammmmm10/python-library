from ai_bridge_kit.retry import run_with_retry


def test_run_with_retry_recovers_after_failures() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = run_with_retry(
        flaky,
        retries=3,
        backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        jitter=False,
        retry_on=(RuntimeError,),
    )
    assert result == "ok"
    assert attempts["count"] == 3


def test_run_with_retry_raises_after_exhaustion() -> None:
    attempts = {"count": 0}

    def always_fail() -> str:
        attempts["count"] += 1
        raise RuntimeError("boom")

    try:
        run_with_retry(
            always_fail,
            retries=2,
            backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            jitter=False,
            retry_on=(RuntimeError,),
        )
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass

    assert attempts["count"] == 3
