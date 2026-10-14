"""
Тесты LLM-фильтра на моках httpx (monkeypatch).
Без реального ключа и сети.
Запуск: python tests/test_llm_filter_mocks.py
        (pytest опционален; тест самодостаточен)
"""
import json
import sys
import os

try:
    import pytest  # noqa: F401  # optional
except ImportError:
    pytest = None  # type: ignore

from newsbot.llm.client import DeepSeekClient, LLMError, BalanceInfo
from newsbot.llm.filter import LLMFilter, FilterResult, route
from newsbot.collector import NewsItem


# ─────────────────────────────────────────────────────────────────
# Fixtures — мок-клиент
# ─────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        """Имитирует httpx: поднимает HTTPStatusError при 4xx/5xx."""
        if 400 <= self.status_code < 600:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,
                response=self,
            )


class FakeChatCompletion:
    """Захватывает последний POST-body для assertions."""

    last_request_body: dict = {}

    @classmethod
    def install(cls, text_response: str, *, status_code: int = 200):
        import httpx

        def fake_post(url, **kwargs):
            cls.last_request_body = kwargs.get("json", {})
            return FakeResponse(
                {"choices": [{"message": {"content": text_response}}]},
                status_code=status_code,
            )

        import newsbot.llm.client as client_module
        client_module.httpx.Client = FakeClientPost(text_response, status_code)
        client_module.httpx.Client.post = fake_post

    @classmethod
    def uninstall(cls):
        import httpx
        import newsbot.llm.client as client_module
        # restore real class
        client_module.httpx.Client = httpx.Client


class FakeClientPost:
    """Подменяет httpx.Client для POST в DeepSeekClient."""

    def __init__(self, text_response: str, status_code: int = 200):
        self.text_response = text_response
        self.status_code = status_code

    def post(self, url, **kwargs):
        FakeChatCompletion.last_request_body = kwargs.get("json", {})
        return FakeResponse(
            {"choices": [{"message": {"content": self.text_response}}]},
            status_code=self.status_code,
        )


def make_client(model_response: str, *, status_code: int = 200) -> DeepSeekClient:
    """
    Возвращает DeepSeekClient, у которого POST подменён на мок.
    """
    import httpx
    import newsbot.llm.client as client_module

    original_client_class = client_module.httpx.Client

    class _MockClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {"choices": [{"message": {"content": model_response}}]},
                status_code=status_code,
            )

        def get(s, url, **kwargs):
            return FakeResponse({"balance_infos": []}, status_code=200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockClient  # type: ignore
    client = DeepSeekClient(
        api_key="mock-key-not-logged",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )
    return client


def unmock():
    import httpx
    import newsbot.llm.client as client_module
    client_module.httpx.Client = httpx.Client


# ─────────────────────────────────────────────────────────────────
# Tests: route()
# ─────────────────────────────────────────────────────────────────

class TestRoute:
    def test_push_high_score(self):
        assert route(10, push_threshold=7, digest_threshold=5) == "push"
        assert route(7, push_threshold=7, digest_threshold=5) == "push"

    def test_digest_mid_score(self):
        assert route(6, push_threshold=7, digest_threshold=5) == "digest"
        assert route(5, push_threshold=7, digest_threshold=5) == "digest"

    def test_skip_low_score(self):
        assert route(4, push_threshold=7, digest_threshold=5) == "skip"
        assert route(0, push_threshold=7, digest_threshold=5) == "skip"

    def test_boundary_values(self):
        # пороги из конфига по умолчанию: push=7, digest=5
        assert route(7, 7, 5) == "push"
        assert route(6, 7, 5) == "digest"
        assert route(5, 7, 5) == "digest"
        assert route(4, 7, 5) == "skip"


# ─────────────────────────────────────────────────────────────────
# Tests: FilterResult dataclass
# ─────────────────────────────────────────────────────────────────

class TestFilterResult:
    def test_ok_true_has_fields(self):
        r = FilterResult(ok=True, score=8, summary="тест", reason="ok")
        assert r.ok is True
        assert r.score == 8
        assert r.summary == "тест"
        assert r.reason == "ok"

    def test_ok_false_defaults(self):
        r = FilterResult(ok=False)
        assert r.ok is False
        assert r.score is None
        assert r.summary == ""
        assert r.reason == ""


# ─────────────────────────────────────────────────────────────────
# Tests: DeepSeekClient.chat_json — валидный ответ
# ─────────────────────────────────────────────────────────────────

def test_chat_json_valid_response():
    valid_json = '{"score":8,"summary":"Сводка: сбито 12 БПЛА","reason":"попадание в тему"}'
    client = make_client(valid_json)

    try:
        result = client.chat_json(
            system="ты фильтр",
            user="заголовок: Атака БПЛА",
        )

        assert result["score"] == 8
        assert "сбито" in result["summary"]
        assert result["reason"] == "попадание в тему"
    finally:
        unmock()


# ─────────────────────────────────────────────────────────────────
# Tests: DeepSeekClient.chat_json — невалидный JSON
# ─────────────────────────────────────────────────────────────────

def test_chat_json_invalid_json_raises_llm_error():
    # использует standalone runner: вместо pytest.raises — try/except внутри
    bad_content = "Это не JSON вообще"

    import httpx
    import newsbot.llm.client as client_module

    class _MockClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {"choices": [{"message": {"content": bad_content}}]},
                status_code=200,
            )

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="test",
        max_retries=1,
    )

    try:
        try:
            client.chat_json("sys", "user")
            raise AssertionError("ожидалось LLMError")
        except LLMError as exc:
            assert "json" in str(exc).lower() or "попыт" in str(exc).lower()
    finally:
        unmock()


# ─────────────────────────────────────────────────────────────────
# Tests: DeepSeekClient.chat_json — HTTP 500, ретраи, затем ошибка
# ─────────────────────────────────────────────────────────────────

def test_chat_json_http_500_retry_then_fail():
    """Проверяем: при 5xx ретраим max_retries раз, затем LLMError."""
    import httpx
    import newsbot.llm.client as client_module

    call_count = [0]  # list чтобы обойти область видимости замыкания

    class _MockClient500:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            call_count[0] += 1
            return FakeResponse({"error": "internal error"}, status_code=500)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockClient500  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="test",
        max_retries=3,
    )

    try:
        try:
            client.chat_json("sys", "user")
            raise AssertionError("ожидалось LLMError")
        except LLMError as exc:
            assert call_count[0] == 3, f"expected 3 calls, got {call_count[0]}"
            assert "500" in str(exc) or "попыток" in str(exc)
    finally:
        unmock()


# ─────────────────────────────────────────────────────────────────
# Tests: DeepSeekClient.get_balance — парсинг balance_infos
# ─────────────────────────────────────────────────────────────────

def test_get_balance_parsed():
    import httpx
    import newsbot.llm.client as client_module

    balance_response = {
        "balance_infos": [
            {"currency": "CNY", "total_balance": 8.5, "granted_balance": 0.0, "topped_up_balance": 8.5},
            {"currency": "USD", "total_balance": 1.25, "granted_balance": 1.0, "topped_up_balance": 0.25},
        ]
    }

    class _MockBalanceClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def get(s, url, **kwargs):
            return FakeResponse(balance_response, status_code=200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockBalanceClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="test",
    )

    try:
        balance = client.get_balance()
        assert balance.is_available is True
        assert balance.currency == "CNY"
        assert balance.total_balance == 8.5
    finally:
        unmock()


def test_get_balance_empty_infos():
    """Пустой balance_infos → is_available=False (НЕ валидный нулевой баланс)."""
    import httpx
    import newsbot.llm.client as client_module

    class _MockClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def get(s, url, **kwargs):
            return FakeResponse({"balance_infos": []}, status_code=200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockClient  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")

    try:
        balance = client.get_balance()
        # пустой balance_infos ≠ ошибка, но и ≠ валидный баланс → is_available=False
        # чтобы pipeline не слал ложный алерт о «баланс 0.0»
        assert balance.is_available is False
        assert balance.total_balance == 0.0
    finally:
        unmock()


# ─────────────────────────────────────────────────────────────────
# Tests: LLMFilter.evaluate — пороги и решения
# ─────────────────────────────────────────────────────────────────

def make_item(title="Тестовая новость", summary="Краткое содержание"):
    return NewsItem(
        source="TestSource",
        url="https://example.com/1",
        title=title,
        summary=summary,
        published="2024-01-01T12:00:00",
        published_parsed=None,
    )


def make_filter_config():
    return {
        "enabled": True,
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key": "mock-key",
        "timeout_sec": 60,
        "max_retries": 3,
        "push_threshold": 7,
        "digest_threshold": 5,
        "system_prompt": "ты фильтр",
    }


def test_filter_score_9_push():
    import httpx
    import newsbot.llm.client as client_module

    resp = {"score": 9, "summary": "Факт: атака БПЛА", "reason": "тема"}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _Mock  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is True
        assert res.score == 9
        assert route(res.score, flt.push_threshold, flt.digest_threshold) == "push"
    finally:
        unmock()


def test_filter_score_6_digest():
    import httpx
    import newsbot.llm.client as client_module

    resp = {"score": 6, "summary": "Слабо релевантно", "reason": "边缘"}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _Mock  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is True
        assert res.score == 6
        assert route(res.score, flt.push_threshold, flt.digest_threshold) == "digest"
    finally:
        unmock()


def test_filter_score_4_skip():
    import httpx
    import newsbot.llm.client as client_module

    resp = {"score": 4, "summary": "Мнение автора", "reason": "не факт"}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _Mock  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is True
        assert res.score == 4
        assert route(res.score, flt.push_threshold, flt.digest_threshold) == "skip"
    finally:
        unmock()


def test_filter_llm_error_returns_ok_false():
    """При LLMError фильтр возвращает ok=False (fallback: слать непрофильтрованным)."""
    import httpx
    import newsbot.llm.client as client_module

    class _MockError:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            raise httpx.TimeoutException("timeout")

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockError  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model", max_retries=1)
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is False
        assert res.score is None
        assert res.summary == ""
    finally:
        unmock()


def test_filter_score_out_of_range_returns_ok_false():
    """Score вне 0..10 → ok=False."""
    import httpx
    import newsbot.llm.client as client_module

    resp = {"score": 99, "summary": "Что-то странное", "reason": ""}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _Mock  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is False
    finally:
        unmock()


def test_filter_empty_summary_returns_ok_false():
    """Пустой summary → ok=False."""
    import httpx
    import newsbot.llm.client as client_module

    resp = {"score": 8, "summary": "", "reason": "ok"}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _Mock  # type: ignore
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = make_filter_config()
    flt = LLMFilter(cfg, client=client)

    try:
        res = flt.evaluate(make_item())
        assert res.ok is False
    finally:
        unmock()


def test_filter_disabled_returns_high_score():
    """disabled=True → сразу ok=True score=10."""
    cfg = make_filter_config()
    cfg["enabled"] = False
    flt = LLMFilter(cfg)
    item = make_item()
    res = flt.evaluate(item)
    assert res.ok is True
    assert res.score == 10


# ─────────────────────────────────────────────────────────────────
# Main runner (standalone, without pytest)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (test_fn, description)
        (lambda: TestRoute().test_push_high_score(), "test_push_high_score"),
        (lambda: TestRoute().test_digest_mid_score(), "test_digest_mid_score"),
        (lambda: TestRoute().test_skip_low_score(), "test_skip_low_score"),
        (lambda: TestRoute().test_boundary_values(), "test_boundary_values"),
        (lambda: TestFilterResult().test_ok_true_has_fields(), "test_ok_true_has_fields"),
        (lambda: TestFilterResult().test_ok_false_defaults(), "test_ok_false_defaults"),
        (lambda: test_chat_json_valid_response(), "test_chat_json_valid_response"),
        (lambda: test_chat_json_invalid_json_raises_llm_error(), "test_chat_json_invalid_json_raises_llm_error"),
        (lambda: test_chat_json_http_500_retry_then_fail(), "test_chat_json_http_500_retry_then_fail"),
        (lambda: test_get_balance_parsed(), "test_get_balance_parsed"),
        (lambda: test_get_balance_empty_infos(), "test_get_balance_empty_infos"),
        (lambda: test_filter_score_9_push(), "test_filter_score_9_push"),
        (lambda: test_filter_score_6_digest(), "test_filter_score_6_digest"),
        (lambda: test_filter_score_4_skip(), "test_filter_score_4_skip"),
        (lambda: test_filter_llm_error_returns_ok_false(), "test_filter_llm_error_returns_ok_false"),
        (lambda: test_filter_score_out_of_range_returns_ok_false(), "test_filter_score_out_of_range_returns_ok_false"),
        (lambda: test_filter_empty_summary_returns_ok_false(), "test_filter_empty_summary_returns_ok_false"),
        (lambda: test_filter_disabled_returns_high_score(), "test_filter_disabled_returns_high_score"),
    ]

    passed = 0
    failed = 0
    for fn, name in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1
        finally:
            # на случай если тест не дошёл до unmock
            try:
                unmock()
            except Exception:
                pass

    print()
    total = passed + failed
    print(f"Result: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed")
        sys.exit(1)
    else:
        print(" -- all green!")
        sys.exit(0)