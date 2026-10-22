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

from newsbot.llm.client import DeepSeekClient, LLMError, BalanceInfo, UsageStats, _safe_int
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

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockClient  # type: ignore
    client = DeepSeekClient(
        api_key="mock-key-not-logged",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
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

    # ── urgent + push_urgent_only ──────────────────────────────────────────────

    def test_urgent_only_urgent_high_score_push(self):
        """urgent + score>=7 + push_urgent_only=True → push."""
        assert route(9, 7, 5, urgent=True, push_urgent_only=True) == "push"
        assert route(7, 7, 5, urgent=True, push_urgent_only=True) == "push"

    def test_urgent_only_urgent_low_score_digest(self):
        """urgent + score<7 + push_urgent_only=True → digest (не push)."""
        assert route(6, 7, 5, urgent=True, push_urgent_only=True) == "digest"
        assert route(5, 7, 5, urgent=True, push_urgent_only=True) == "digest"

    def test_urgent_only_not_urgent_high_score_not_push(self):
        """Не-urgent + score>=7 + push_urgent_only=True → digest (НЕ push)."""
        assert route(9, 7, 5, urgent=False, push_urgent_only=True) == "digest"
        assert route(7, 7, 5, urgent=False, push_urgent_only=True) == "digest"

    def test_urgent_only_not_urgent_low_score_skip(self):
        """Не-urgent + score<digest + push_urgent_only=True → skip."""
        assert route(4, 7, 5, urgent=False, push_urgent_only=True) == "skip"
        assert route(0, 7, 5, urgent=False, push_urgent_only=True) == "skip"

    def test_legacy_mode_score_ge_push_threshold_push(self):
        """push_urgent_only=False → прежнее поведение: score>=7 → push."""
        assert route(9, 7, 5, urgent=False, push_urgent_only=False) == "push"
        assert route(7, 7, 5, urgent=False, push_urgent_only=False) == "push"
        assert route(9, 7, 5, urgent=True, push_urgent_only=False) == "push"  # urgent игнорируется

    def test_legacy_mode_score_ge_digest_threshold_digest(self):
        """push_urgent_only=False → score 5–6 → digest."""
        assert route(6, 7, 5, urgent=False, push_urgent_only=False) == "digest"
        assert route(5, 7, 5, urgent=False, push_urgent_only=False) == "digest"

    def test_legacy_mode_score_lt_digest_skip(self):
        """push_urgent_only=False → score<5 → skip."""
        assert route(4, 7, 5, urgent=True, push_urgent_only=False) == "skip"
        assert route(0, 7, 5, urgent=True, push_urgent_only=False) == "skip"


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
        assert r.urgent is False  # дефолт

    def test_ok_false_defaults(self):
        r = FilterResult(ok=False)
        assert r.ok is False
        assert r.score is None
        assert r.summary == ""
        assert r.reason == ""
        assert r.urgent is False

    def test_urgent_field(self):
        r = FilterResult(ok=True, score=9, summary="срочное", reason="ok", urgent=True)
        assert r.urgent is True
        assert r.ok is True
        assert r.score == 9


# ─────────────────────────────────────────────────────────────────
# Tests: LLMFilter._parse_bool
# ─────────────────────────────────────────────────────────────────

class TestLLMFilterParseBool:
    def test_parse_bool_false(self):
        assert LLMFilter._parse_bool(False) is False

    def test_parse_bool_true(self):
        assert LLMFilter._parse_bool(True) is True

    def test_parse_bool_str_true_values(self):
        assert LLMFilter._parse_bool("true") is True
        assert LLMFilter._parse_bool("True") is True
        assert LLMFilter._parse_bool("TRUE") is True
        assert LLMFilter._parse_bool("1") is True
        assert LLMFilter._parse_bool("yes") is True
        assert LLMFilter._parse_bool("on") is True

    def test_parse_bool_str_false_values(self):
        assert LLMFilter._parse_bool("false") is False
        assert LLMFilter._parse_bool("False") is False
        assert LLMFilter._parse_bool("FALSE") is False
        assert LLMFilter._parse_bool("0") is False
        assert LLMFilter._parse_bool("no") is False
        assert LLMFilter._parse_bool("off") is False
        assert LLMFilter._parse_bool("") is False

    def test_parse_bool_str_mixed(self):
        assert LLMFilter._parse_bool("Yes") is True
        assert LLMFilter._parse_bool("OFF") is False

    def test_parse_bool_non_str_defaults(self):
        assert LLMFilter._parse_bool(None) is False
        assert LLMFilter._parse_bool(123) is False
        assert LLMFilter._parse_bool(0) is False


# ─────────────────────────────────────────────────────────────────
# Tests: _parse_response — парсинг urgent
# ─────────────────────────────────────────────────────────────────

class TestParseResponseUrgent:
    """Тесты парсинга поля urgent в _parse_response."""

    def _parse(self, raw: dict) -> FilterResult:
        """Прямой вызов _parse_response через LLMFilter.evaluate."""
        import httpx
        import newsbot.llm.client as client_module

        resp_str = json.dumps(raw)

        class _Mock:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return FakeResponse(
                    {"choices": [{"message": {"content": resp_str}}]},
                    status_code=200,
                )

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        client_module.httpx.Client = _Mock  # type: ignore
        client = DeepSeekClient("key", "https://api.deepseek.com", "model")
        cfg = make_filter_config()
        flt = LLMFilter(cfg, client=client)

        try:
            return flt.evaluate(make_item())
        finally:
            unmock()

    def test_urgent_true_bool(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": True})
        assert r.ok is True
        assert r.urgent is True

    def test_urgent_false_bool(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": False})
        assert r.ok is True
        assert r.urgent is False

    def test_urgent_true_str_lower(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": "true"})
        assert r.ok is True
        assert r.urgent is True

    def test_urgent_true_str_upper(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": "TRUE"})
        assert r.ok is True
        assert r.urgent is True

    def test_urgent_false_str(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": "false"})
        assert r.ok is True
        assert r.urgent is False

    def test_urgent_one_int(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": 1})
        assert r.ok is True
        assert r.urgent is True

    def test_urgent_zero_int(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": 0})
        assert r.ok is True
        assert r.urgent is False

    def test_urgent_missing_defaults_false(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r"})
        assert r.ok is True
        assert r.urgent is False, "отсутствующий urgent должен дефолтить в False"

    def test_urgent_garbage_defaults_false(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": "maybe"})
        assert r.ok is True
        assert r.urgent is False, "мусорный urgent должен дефолтить в False"

    def test_urgent_null_defaults_false(self):
        r = self._parse({"score": 8, "summary": "ok", "reason": "r", "urgent": None})
        assert r.ok is True
        assert r.urgent is False, "null urgent должен дефолтить в False"


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

        def close(s):
            pass

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

        def close(s):
            pass

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
# Tests: usage-логирование
# ─────────────────────────────────────────────────────────────────

def test_chat_json_usage_accumulated():
    """
    usage в ответе API → аккумулируется в client.usage_stats,
    отсутствие usage не ломает вызов.
    """
    import httpx
    import newsbot.llm.client as client_module

    valid_json = '{"score":8,"summary":"ok","reason":"test"}'

    # OpenAI-формат: prompt_tokens_details.cached_tokens
    usage_data = {
        "prompt_tokens": 1100,
        "completion_tokens": 200,
        "total_tokens": 1300,
        "prompt_tokens_details": {
            "cached_tokens": 900,
        },
    }

    class _MockUsageClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {
                    "choices": [{"message": {"content": valid_json}}],
                    "usage": usage_data,
                },
                status_code=200,
            )

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockUsageClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    try:
        result = client.chat_json("sys", "user")
        assert result["score"] == 8

        stats = client.usage_stats
        assert stats.calls == 1
        assert stats.prompt_tokens == 1100
        assert stats.completion_tokens == 200
        assert stats.total_tokens == 1300
        assert stats.prompt_cache_hit_tokens == 900
        assert stats.prompt_cache_miss_tokens == 200
    finally:
        unmock()


def test_chat_json_usage_absent_no_crash():
    """
    usage отсутствует в ответе → вызов завершается успешно,
    usage_stats.calls == 1 (вызов прошёл), токены = 0 (fallback).
    """
    import httpx
    import newsbot.llm.client as client_module

    valid_json = '{"score":8,"summary":"ok","reason":"test"}'

    class _MockNoUsageClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            # ответ БЕЗ поля usage
            return FakeResponse(
                {"choices": [{"message": {"content": valid_json}}]},
                status_code=200,
            )

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockNoUsageClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    try:
        result = client.chat_json("sys", "user")
        assert result["score"] == 8

        stats = client.usage_stats
        # вызов прошёл, но usage отсутствует → инкремент calls,
        # но токены остаются 0 (безопасный fallback)
        assert stats.calls == 1
        assert stats.prompt_tokens == 0
        assert stats.total_tokens == 0
    finally:
        unmock()


def test_chat_json_usage_deepseek_top_level():
    """
    DeepSeek top-level формат (prompt_cache_hit_tokens / prompt_cache_miss_tokens)
    корректно парсится: cached_hit=900, cached_miss=200.
    """
    import httpx
    import newsbot.llm.client as client_module

    valid_json = '{"score":8,"summary":"ok","reason":"test"}'

    # DeepSeek top-level формат (НЕ prompt_tokens_details)
    usage_data = {
        "prompt_tokens": 1100,
        "completion_tokens": 200,
        "total_tokens": 1300,
        "prompt_cache_hit_tokens": 900,
        "prompt_cache_miss_tokens": 200,
    }

    class _MockUsageClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {
                    "choices": [{"message": {"content": valid_json}}],
                    "usage": usage_data,
                },
                status_code=200,
            )

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockUsageClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    try:
        result = client.chat_json("sys", "user")
        assert result["score"] == 8

        stats = client.usage_stats
        assert stats.calls == 1
        assert stats.prompt_tokens == 1100
        assert stats.completion_tokens == 200
        assert stats.total_tokens == 1300
        assert stats.prompt_cache_hit_tokens == 900
        assert stats.prompt_cache_miss_tokens == 200
    finally:
        unmock()


def test_chat_json_usage_only_cached_hit():
    """
    Формат с одним cached_hit (без cached_miss): miss вычисляется
    как prompt - cached_hit, неотрицательный.
    """
    import httpx
    import newsbot.llm.client as client_module

    valid_json = '{"score":8,"summary":"ok","reason":"test"}'

    # Только cached_hit, cached_miss отсутствует
    usage_data = {
        "prompt_tokens": 1100,
        "completion_tokens": 200,
        "total_tokens": 1300,
        "prompt_cache_hit_tokens": 700,
    }

    class _MockUsageClient:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {
                    "choices": [{"message": {"content": valid_json}}],
                    "usage": usage_data,
                },
                status_code=200,
            )

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    client_module.httpx.Client = _MockUsageClient  # type: ignore

    client = DeepSeekClient(
        api_key="mock-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    try:
        result = client.chat_json("sys", "user")
        assert result["score"] == 8

        stats = client.usage_stats
        assert stats.prompt_cache_hit_tokens == 700
        assert stats.prompt_cache_miss_tokens == 400  # 1100 - 700
        assert stats.prompt_cache_miss_tokens >= 0
    finally:
        unmock()


def test_usage_stats_reset_clears_counters():
    """
    reset() обнуляет все счётчики, model-имя сохраняется.
    """
    stats = UsageStats(model="deepseek-flash")
    # накопим данные
    stats.calls = 3
    stats.prompt_tokens = 3300
    stats.prompt_cache_hit_tokens = 2700
    stats.prompt_cache_miss_tokens = 600
    stats.completion_tokens = 600
    stats.total_tokens = 3900

    stats.reset()

    assert stats.model == "deepseek-flash"
    assert stats.calls == 0
    assert stats.prompt_tokens == 0
    assert stats.prompt_cache_hit_tokens == 0
    assert stats.prompt_cache_miss_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0


def test_usage_stats_safe_int_non_numeric():
    """
    _safe_int: нечисловое значение → default, не падает.
    """
    from newsbot.llm.client import _safe_int

    assert _safe_int("123", 0) == 123
    assert _safe_int("abc", 0) == 0      # ValueError → default
    assert _safe_int(None, 99) == 99      # None → default
    assert _safe_int(["list"], 0) == 0   # TypeError → default
    assert _safe_int(42.5, 0) == 42      # float → int


def test_usage_stats_safe_int_non_finite():
    """Major-регресс: int(inf) → OverflowError не должен выбрасывать ответ LLM."""
    from newsbot.llm.client import _safe_int

    assert _safe_int(float("inf"), 0) == 0
    assert _safe_int(float("-inf"), 7) == 7
    assert _safe_int(float("nan"), 0) == 0
    assert _safe_int("Infinity", 0) == 0   # ValueError → default


def test_usage_stats_infinite_usage_values_do_not_crash():
    """Major-регресс: usage с Infinity/NaN не роняет add_response (контракт клиента)."""
    stats = UsageStats(model="deepseek-flash")
    stats.add_response({
        "usage": {
            "prompt_tokens": float("inf"),
            "completion_tokens": float("nan"),
            "total_tokens": float("inf"),
            "prompt_cache_hit_tokens": float("inf"),
            "prompt_cache_miss_tokens": float("nan"),
        }
    })
    assert stats.calls == 1
    assert stats.prompt_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0
    assert stats.prompt_cache_hit_tokens == 0
    assert stats.prompt_cache_miss_tokens == 0


def test_sanitize_item_field_neutralizes_fullwidth_delimiters():
    """Minor-регресс: fullwidth ＜｜ (U+FF1C/U+FF5C) не проходят sanitize."""
    from newsbot.llm.prompts import sanitize_item_field

    result = sanitize_item_field("\uff1c\uff5c\u0434\u0430\u043d\u043d\u044b\u0435:1\uff5c>")
    assert "\uff1c" not in result
    assert "\uff5c" not in result
    assert "<|" not in result
    assert "< |" in result
    assert "| >" in result


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

        def close(s):
            pass

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

        def close(s):
            pass

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
        "model": "deepseek-flash",
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

        def close(s):
            pass

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

    resp = {"score": 6, "summary": "Слабо релевантно", "reason": "edge case"}

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse({"choices": [{"message": {"content": json.dumps(resp)}}]}, 200)

        def close(s):
            pass

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

        def close(s):
            pass

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

        def close(s):
            pass

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

        def close(s):
            pass

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

        def close(s):
            pass

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
        (lambda: TestRoute().test_urgent_only_urgent_high_score_push(), "test_urgent_only_urgent_high_score_push"),
        (lambda: TestRoute().test_urgent_only_urgent_low_score_digest(), "test_urgent_only_urgent_low_score_digest"),
        (lambda: TestRoute().test_urgent_only_not_urgent_high_score_not_push(), "test_urgent_only_not_urgent_high_score_not_push"),
        (lambda: TestRoute().test_urgent_only_not_urgent_low_score_skip(), "test_urgent_only_not_urgent_low_score_skip"),
        (lambda: TestRoute().test_legacy_mode_score_ge_push_threshold_push(), "test_legacy_mode_score_ge_push_threshold_push"),
        (lambda: TestRoute().test_legacy_mode_score_ge_digest_threshold_digest(), "test_legacy_mode_score_ge_digest_threshold_digest"),
        (lambda: TestRoute().test_legacy_mode_score_lt_digest_skip(), "test_legacy_mode_score_lt_digest_skip"),
        (lambda: TestFilterResult().test_ok_true_has_fields(), "test_ok_true_has_fields"),
        (lambda: TestFilterResult().test_ok_false_defaults(), "test_ok_false_defaults"),
        (lambda: TestFilterResult().test_urgent_field(), "test_urgent_field"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_false(), "test_parse_bool_false"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_true(), "test_parse_bool_true"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_str_true_values(), "test_parse_bool_str_true_values"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_str_false_values(), "test_parse_bool_str_false_values"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_str_mixed(), "test_parse_bool_str_mixed"),
        (lambda: TestLLMFilterParseBool().test_parse_bool_non_str_defaults(), "test_parse_bool_non_str_defaults"),
        (lambda: TestParseResponseUrgent().test_urgent_true_bool(), "test_urgent_true_bool"),
        (lambda: TestParseResponseUrgent().test_urgent_false_bool(), "test_urgent_false_bool"),
        (lambda: TestParseResponseUrgent().test_urgent_true_str_lower(), "test_urgent_true_str_lower"),
        (lambda: TestParseResponseUrgent().test_urgent_true_str_upper(), "test_urgent_true_str_upper"),
        (lambda: TestParseResponseUrgent().test_urgent_false_str(), "test_urgent_false_str"),
        (lambda: TestParseResponseUrgent().test_urgent_one_int(), "test_urgent_one_int"),
        (lambda: TestParseResponseUrgent().test_urgent_zero_int(), "test_urgent_zero_int"),
        (lambda: TestParseResponseUrgent().test_urgent_missing_defaults_false(), "test_urgent_missing_defaults_false"),
        (lambda: TestParseResponseUrgent().test_urgent_garbage_defaults_false(), "test_urgent_garbage_defaults_false"),
        (lambda: TestParseResponseUrgent().test_urgent_null_defaults_false(), "test_urgent_null_defaults_false"),
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
        (lambda: test_chat_json_usage_accumulated(), "test_chat_json_usage_accumulated"),
        (lambda: test_chat_json_usage_absent_no_crash(), "test_chat_json_usage_absent_no_crash"),
        (lambda: test_chat_json_usage_deepseek_top_level(), "test_chat_json_usage_deepseek_top_level"),
        (lambda: test_chat_json_usage_only_cached_hit(), "test_chat_json_usage_only_cached_hit"),
        (lambda: test_usage_stats_reset_clears_counters(), "test_usage_stats_reset_clears_counters"),
        (lambda: test_usage_stats_safe_int_non_numeric(), "test_usage_stats_safe_int_non_numeric"),
        (lambda: test_usage_stats_safe_int_non_finite(), "test_usage_stats_safe_int_non_finite"),
        (lambda: test_usage_stats_infinite_usage_values_do_not_crash(), "test_usage_stats_infinite_usage_values_do_not_crash"),
        (lambda: test_sanitize_item_field_neutralizes_fullwidth_delimiters(), "test_sanitize_item_field_neutralizes_fullwidth_delimiters"),
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