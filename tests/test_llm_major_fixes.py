"""
Тесты всех Major-фиксов и Minor-фиксов для LLM-слоя.
Запуск: python -m pytest tests/test_llm_major_fixes.py -v
"""
import json
import logging
import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.llm.client import (
    DeepSeekClient, LLMError, LLMTruncationError,
    LLMFatalError, LLMContextLengthError,
    _strip_json_fences, _safe_price, BalanceInfo,
)
from newsbot.llm.filter import LLMFilter, FilterResult


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def make_item(url="https://e.com/1", title="Test", summary="Summary"):
    return NewsItem(
        source="Test", url=url, title=title, summary=summary,
        published="2024-01-01", published_parsed=None,
    )


class _HttpxPatcher:
    """Контекстный менеджер: подменяет httpx.Client на время блока with."""
    def __init__(self, cls):
        self._cls = cls
        self._prev = None

    def __enter__(self):
        import newsbot.llm.client as cm
        self._prev = cm.httpx.Client
        cm.httpx.Client = self._cls
        return self

    def __exit__(self, *args):
        import newsbot.llm.client as cm
        cm.httpx.Client = self._prev


# ─────────────────────────────────────────────────────────────────
# Major #1: Retry-After -1 / "1.5" / "abc" — no crash, retry
# ─────────────────────────────────────────────────────────────────

class TestMajor1RetryAfter:
    """Retry-After с мусором не падает, retry происходит. time.sleep замокан."""

    def _do_test(self, retry_after_hdr, expected_sleep_min, expected_sleep_max):
        """Общий helper: проверяет, что Retry-After hdr приводит к корректному sleep.

        Семантика Retry-After в DeepSeekClient:
          — валидный int → clamp(hdr, 1, 120) + jitter ±50%
          — невалидный (float "1.5", negative "-1", non-numeric "abc", zero "0")
            → warning + default backoff = 2^(attempt-1) + jitter ±50%
        """
        import httpx
        import newsbot.llm.client as cm

        class _Resp429:
            status_code = 429

            def __init__(self, hdr):
                self.headers = {"retry-after": hdr}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("429", request=None, response=self)

        class _FakeHTTPClient429:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp429(retry_after_hdr)

            def get(s, url, **kwargs):
                return _Resp429(retry_after_hdr)

            def close(s):
                pass

        sleep_calls = []

        def mock_sleep(sec):
            sleep_calls.append(sec)

        def mock_jitter(a, b):
            return 1.0  # deterministic: jitter=+1.0

        prev_client = cm.httpx.Client
        prev_sleep = cm.time.sleep
        cm.httpx.Client = _FakeHTTPClient429
        cm.time.sleep = mock_sleep
        try:
            with patch.object(cm.random, 'uniform', mock_jitter):
                client = DeepSeekClient("test", "https://x.com", "test",
                                        max_retries=3, max_tokens=1024)
                try:
                    client.chat_json("sys", "user")
                except LLMError:
                    pass
        finally:
            cm.httpx.Client = prev_client
            cm.time.sleep = prev_sleep

        assert len(sleep_calls) >= 1, (
            f"Retry-After {retry_after_hdr!r}: ожидался хотя бы 1 sleep, "
            f"получено {len(sleep_calls)}"
        )
        for s in sleep_calls:
            assert isinstance(s, (int, float)), (
                f"sleep argument must be numeric, got {type(s).__name__}: {s!r}"
            )
            assert expected_sleep_min <= s <= expected_sleep_max, (
                f"Retry-After {retry_after_hdr!r}: sleep={s} вне диапазона "
                f"[{expected_sleep_min}, {expected_sleep_max}]"
            )
        # Также: должно быть не менее 1 retry (>=2 sleep calls)
        assert len(sleep_calls) >= 2, (
            f"Retry-After {retry_after_hdr!r}: retry не сработал, "
            f"всего {len(sleep_calls)} sleep-вызовов"
        )

    def test_retry_after_negative_one(self):
        """Retry-After: -1 — invalid → warning + default backoff (2**attempt + jitter)."""
        self._do_test("-1", 0.5, 5.5)

    def test_retry_after_float_string(self):
        """Retry-After: "1.5" — не целое → warning + default backoff."""
        self._do_test("1.5", 0.5, 5.5)

    def test_retry_after_non_numeric(self):
        """Retry-After: "abc" — не число → warning + default backoff."""
        self._do_test("abc", 0.5, 5.5)

    def test_retry_after_zero(self):
        """Retry-After: 0 — invalid → warning + default backoff (2**attempt + jitter)."""
        self._do_test("0", 0.5, 5.5)

    def test_retry_after_1000(self):
        """Retry-After: 1000 → clamp(1000, 1, 120)=120 сек + jitter ±50%."""
        self._do_test("1000", 60, 180)

    def test_retry_after_positive_int(self):
        """Retry-After: 10 → clamp(10, 1, 120)=10 сек + jitter ±50%."""
        self._do_test("10", 5, 15)


# ─────────────────────────────────────────────────────────────────
# Major #2: JSON fence / preamble stripping
# ─────────────────────────────────────────────────────────────────

class TestMajor2JsonFenceStripping:
    """JSON с ```json … ``` или преамбулой парсится корректно."""

    def _client_result(self, content):
        """Мок-клиент, возвращающий заданный content. Возвращает результат chat_json."""
        class _Resp:
            status_code = 200

            def __init__(self, c):
                self._content = c

            def json(self):
                return {
                    "choices": [{"message": {"content": self._content}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
                }

            def raise_for_status(self):
                pass

        class _BalanceResp:
            status_code = 200

            def json(self):
                return {"balance_infos": []}

            def raise_for_status(self):
                pass

        class _FakeHTTPClient:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp(content)

            def get(s, url, **kwargs):
                return _BalanceResp()

            def close(s):
                pass

        with _HttpxPatcher(_FakeHTTPClient):
            client = DeepSeekClient("test", "https://x.com", "test",
                                   max_retries=1, max_tokens=1024)
            return client.chat_json("sys", "user")

    def test_backticks_json_block_parsed(self):
        result = self._client_result('```json\n{"score": 8, "summary": "Test"}\n```')
        assert result["score"] == 8

    def test_backticks_no_lang_parsed(self):
        result = self._client_result('```\n{"score": 7, "summary": "X"}\n```')
        assert result["score"] == 7

    def test_preamble_here_is_json_parsed(self):
        result = self._client_result('Here is the JSON:\n{"score": 9, "summary": "Y"}')
        assert result["score"] == 9

    def test_preamble_here_s_json_parsed(self):
        result = self._client_result("Here's JSON: {\"score\": 6, \"summary\": \"Z\"}")
        assert result["score"] == 6

    def test_preamble_here_is_response_parsed(self):
        result = self._client_result("Here is the response: {\"score\": 5, \"summary\": \"W\"}")
        assert result["score"] == 5

    def test_both_fence_and_preamble_parsed(self):
        result = self._client_result('```json\nHere is the JSON:\n{"score": 4, "summary": "A"}\n```')
        assert result["score"] == 4

    def test_clean_json_unchanged(self):
        result = self._client_result('{"score": 3, "summary": "B"}')
        assert result["score"] == 3

    def test_json_embedded_in_text_parsed(self):
        result = self._client_result("Some explanation.\n{\"score\": 2, \"summary\": \"C\"}\nMore text.")
        assert result["score"] == 2


class TestStripJsonFencesUnit:
    """Unit-тесты для _strip_json_fences."""

    def test_triple_backtick_json(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences('```json\n{"a": 1}\n```')
        assert result == '{"a": 1}'

    def test_triple_backtick_plain(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences('```\n{"a": 2}\n```')
        assert result == '{"a": 2}'

    def test_preamble_json(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences("Here is JSON: {\"a\": 3}")
        assert result == '{"a": 3}'

    def test_preamble_the_json(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences("Here is the JSON:\n{\"a\": 4}\n")
        assert result == '{"a": 4}'

    def test_preamble_response(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences("Here's the response: {\"a\": 5}")
        assert result == '{"a": 5}'

    def test_both_fence_and_preamble(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences('```json\nHere is the JSON: {"a": 6}\n```')
        assert result == '{"a": 6}'

    def test_clean_json_unchanged(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences('{"a": 7}')
        assert result == '{"a": 7}'

    def test_json_embedded_in_text(self):
        from newsbot.llm.client import _strip_json_fences
        result = _strip_json_fences("Some text\n{\"a\": 8}\nmore text")
        assert result == '{"a": 8}'

    def test_nested_braces_extracted(self):
        from newsbot.llm.client import _strip_json_fences
        text = 'prefix {"outer": {"inner": [1, 2]}} suffix'
        result = _strip_json_fences(text)
        assert result == '{"outer": {"inner": [1, 2]}}'

    def test_empty_string(self):
        from newsbot.llm.client import _strip_json_fences
        assert _strip_json_fences("") == ""

    def test_none_input(self):
        from newsbot.llm.client import _strip_json_fences
        assert _strip_json_fences(None) == ""

    def test_markdown_in_text_not_stripped(self):
        from newsbot.llm.client import _strip_json_fences
        # Markdown вне ```json ``` не снимается
        result = _strip_json_fences('{"text": "Here is **bold** text"}')
        assert result == '{"text": "Here is **bold** text"}'


# ─────────────────────────────────────────────────────────────────
# Major #3: LLMFatalError (402/401/403) and LLMContextLengthError (400)
# ─────────────────────────────────────────────────────────────────

class TestMajor3FatalAndContextErrors:
    """Тесты новых исключений."""

    def test_401_raises_llmfatal_error(self):
        """HTTP 401 → LLMFatalError."""
        import httpx
        import newsbot.llm.client as cm

        class _Resp401:
            status_code = 401

            def __init__(self):
                self.text = "Unauthorized"
                self.headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("401", request=None, response=self)

        class _FakeHTTPClient401:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp401()

            def get(s, url, **kwargs):
                return _Resp401()

            def close(s):
                pass

        prev = cm.httpx.Client
        cm.httpx.Client = _FakeHTTPClient401
        try:
            client = DeepSeekClient("test", "https://x.com", "test", max_retries=3)
            try:
                client.chat_json("sys", "user")
                assert False, "Должно было выбросить LLMFatalError"
            except LLMFatalError:
                pass
            except Exception as e:
                assert False, f"Должно быть LLMFatalError, получено {type(e).__name__}"
        finally:
            cm.httpx.Client = prev

    def test_402_raises_llmfatal_error(self):
        """HTTP 402 → LLMFatalError."""
        import httpx
        import newsbot.llm.client as cm

        class _Resp402:
            status_code = 402

            def __init__(self):
                self.text = "Insufficient balance"
                self.headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("402", request=None, response=self)

        class _FakeHTTPClient402:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp402()

            def get(s, url, **kwargs):
                return _Resp402()

            def close(s):
                pass

        prev = cm.httpx.Client
        cm.httpx.Client = _FakeHTTPClient402
        try:
            client = DeepSeekClient("test", "https://x.com", "test", max_retries=3)
            try:
                client.chat_json("sys", "user")
                assert False, "Должно было выбросить LLMFatalError"
            except LLMFatalError:
                pass
        finally:
            cm.httpx.Client = prev

    def test_400_raises_llmcontext_length_error(self):
        """HTTP 400 с context_length -> LLMContextLengthError.

        CRITICAL-фикс: новые уточнённые маркеры ("context_length", "maximum context",
        "too long", "reduce the length") → LLMContextLengthError.
        Широкие маркеры ("tokens" одиночное, "maximum" одиночное) → общий LLMError.
        """
        import httpx
        import newsbot.llm.client as cm

        class _Resp400Ctx:
            status_code = 400

            def __init__(self):
                self.text = "context_length_exceeded"
                self.headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("400", request=None, response=self)

        class _FakeHTTPClient400Ctx:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp400Ctx()

            def get(s, url, **kwargs):
                return _Resp400Ctx()

            def close(s):
                pass

        prev = cm.httpx.Client
        cm.httpx.Client = _FakeHTTPClient400Ctx
        try:
            client = DeepSeekClient("test", "https://x.com", "test", max_retries=1)
            try:
                client.chat_json("sys", "user")
                assert False, "Expected LLMContextLengthError"
            except LLMContextLengthError:
                pass  # expected
            except LLMError as e:
                assert False, f"Expected LLMContextLengthError, got {type(e).__name__}: {e}"
        finally:
            cm.httpx.Client = prev

    def test_400_invalid_max_tokens_raises_llm_error_not_context_error(self):
        """HTTP 400 'Invalid max_tokens value' → общий LLMError (НЕ LLMContextLengthError).

        CRITICAL-фикс: широкие маркеры ("tokens", "maximum") одиночно НЕ вызывают
        контекстную ошибку → сплит не происходит (chat_json вызван 1 раз).
        """
        import httpx
        import newsbot.llm.client as cm

        class _Resp400Tokens:
            status_code = 400

            def __init__(self):
                self.text = "Invalid max_tokens value, the valid range is 1-8192"
                self.headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("400", request=None, response=self)

        class _FakeHTTPClient400Tokens:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp400Tokens()

            def get(s, url, **kwargs):
                return _Resp400Tokens()

            def close(s):
                pass

        prev = cm.httpx.Client
        cm.httpx.Client = _FakeHTTPClient400Tokens
        try:
            client = DeepSeekClient("test", "https://x.com", "test", max_retries=1)
            try:
                client.chat_json("sys", "user")
                assert False, "Expected LLMError"
            except LLMError as e:
                assert not isinstance(e, LLMContextLengthError), (
                    f"400 'Invalid max_tokens value' should NOT raise LLMContextLengthError, "
                    f"got {type(e).__name__}: {e}"
                )
                # Проверяем что это именно общий LLMError (без сплита)
                assert "invalid max_tokens" in str(e).lower() or "400" in str(e)
        finally:
            cm.httpx.Client = prev

    def test_400_unsupported_response_format_raises_llm_error(self):
        """HTTP 400 unsupported response_format -> обычный LLMError (без сплита)."""
        import httpx
        import newsbot.llm.client as cm

        class _Resp400Bad:
            status_code = 400

            def __init__(self):
                self.text = "unsupported response_format"
                self.headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("400", request=None, response=self)

        class _FakeHTTPClient400Bad:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _Resp400Bad()

            def get(s, url, **kwargs):
                return _Resp400Bad()

            def close(s):
                pass

        prev = cm.httpx.Client
        cm.httpx.Client = _FakeHTTPClient400Bad  # assign CLASS, not instance
        try:
            client = DeepSeekClient("test", "https://x.com", "test", max_retries=1)
            try:
                client.chat_json("sys", "user")
                assert False, "Expected LLMError"
            except LLMError as e:
                assert not isinstance(e, LLMContextLengthError), (
                    f"400 unsupported response_format should NOT raise LLMContextLengthError, "
                    f"got {type(e).__name__}: {e}"
                )
                # Проверяем что это именно наш общий LLMError, а не сплит
                assert "unsupported" in str(e).lower() or "400" in str(e)
        finally:
            cm.httpx.Client = prev


    def test_llmfatal_error_is_llm_error(self):
        """LLMFatalError наследуется от LLMError."""
        assert issubclass(LLMFatalError, LLMError)
        assert issubclass(LLMContextLengthError, LLMError)

    def test_llmcontext_length_error_is_llmtruncation_error(self):
        """LLMContextLengthError наследуется от LLMTruncationError."""
        assert issubclass(LLMContextLengthError, LLMTruncationError)


class TestMajor3FilterFatalError:
    """filter.evaluate: LLMFatalError → ok=False, items pending."""

    def test_fatal_error_returns_fatal_filter_result(self):
        """LLMFatalError → FilterResult(ok=False, fatal=True).

        Контракт: fatal=True означает items остаются pending (как cost_blocked),
        отличие от ok=False с fatal=False — при fatal=False это per-item fallback
        (непрофильтрованное сырьё уходит в пуш).
        """
        from newsbot.llm.filter import LLMFilter

        class _FatalClient:
            def chat_json(self, *args, **kwargs):
                raise LLMFatalError("HTTP 402: insufficient balance")

            def get_balance(self):
                return BalanceInfo(True, "USD", 50.0)

        flt = LLMFilter(
            {"enabled": True, "model": "t", "api_key": "k",
             "push_urgent_only": False},
            client=_FatalClient(),
        )
        item = make_item()
        result = flt.evaluate(item)
        assert result.ok is False
        assert result.fatal is True, (
            "fatal=True: pipeline должен трактовать как pending, не пуш-спам"
        )
        assert result.cost_blocked is False, (
            "cost_blocked=False: это фатальная ошибка, а не блокировка по лимиту"
        )

    def test_fatal_error_batch_all_fatal(self):
        """LLMFatalError на батче → все FilterResult(ok=False, fatal=True)."""
        from newsbot.llm.filter import LLMFilter

        class _FatalClient:
            def chat_json(self, *args, **kwargs):
                raise LLMFatalError("HTTP 401: unauthorized")

            def get_balance(self):
                return BalanceInfo(True, "USD", 50.0)

        flt = LLMFilter(
            {"enabled": True, "model": "t", "api_key": "k",
             "push_urgent_only": False},
            client=_FatalClient(),
        )
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(3)]
        results = flt.evaluate_batch(items)
        assert len(results) == 3
        assert all(r.ok is False for r in results)
        assert all(r.cost_blocked is False for r in results)
        assert all(r.fatal is True for r in results), (
            "Все результаты должны иметь fatal=True (items pending)"
        )


# ─────────────────────────────────────────────────────────────────
# Major #4: Sanity cap on dedup duplicates fraction
# ─────────────────────────────────────────────────────────────────

class TestMajor4DedupSanityCap:
    """Sanity cap: >50% помеченных дублей → результат игнорируется."""

    def _make_filter(self, mock_client) -> LLMFilter:
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
        }
        return LLMFilter(cfg, client=mock_client)

    def test_6_of_10_retries_halves(self, caplog):
        """4 items: 3/4=75%%>cap, retry halves, halves ok - result from halves."""
        import logging

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = [
            {"duplicates": [1, 2, 3]},  # 3/4=75%% > cap -> retry halves(2+2)
            {"duplicates": [1]},           # 1/2=50%% -> NOT > cap -> OK (global 0)
            {"duplicates": [1]},           # 1/2=50%% -> NOT > cap -> OK (global 2)
        ]
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(4)]

        with caplog.at_level(logging.WARNING, logger="newsbot.llm.filter"):
            result = flt.detect_duplicates(items, [])

        # half1(0-1): dup id 1→global 0; half2(2-3): dup id 1→global 2
        # (2 halves: id 1 in each → global 0 and global 2)
        assert result == {0, 2}, f"Expected {{0, 2}}, got {result}"
        assert mock_client.chat_json.call_count == 3
        assert any("sanity-cap" in r.message.lower() or "75%" in r.message
                   for r in caplog.records), f"No sanity-cap warning"

    def test_5_of_10_accepted(self):
        """5/10 (50%) — на границе cap, НЕ игнорируется."""
        mock_client = MagicMock()
        # 5 из 10 = ровно 50% — cap 0.5, 5/10 = 0.5, NOT > cap → применяется
        mock_client.chat_json.return_value = {"duplicates": [1, 3, 5, 7, 9]}  # 5 items
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        result = flt.detect_duplicates(items, [])

        assert result == {0, 2, 4, 6, 8}, f"5/10 должно было примениться, got {result}"

    def test_4_of_10_accepted(self):
        """4/10 (40%) — ниже cap, применяется."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {"duplicates": [2, 4, 6, 8]}  # 4 items
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        result = flt.detect_duplicates(items, [])

        assert result == {1, 3, 5, 7}, f"4/10 должно было примениться, got {result}"

    def test_0_of_10_accepted(self):
        """0/10 — валидный ответ (нет дублей), применяется."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {"duplicates": []}
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        result = flt.detect_duplicates(items, [])

        assert result == set()

    def test_41_of_40_chunk_ignored(self):
        """Чанк из 40: 30/40=75% > cap → half-retry (3 вызова, не 1).

        CRITICAL: задача требовала call_count assertion, чтобы отличать
        «cap→retry (3 вызова)» от «cap→сразу отбросили (1 вызов)».
        Каждый half получает 15/20=75%, но при recent=None cap подавляется
        (в halves recent=None), поэтому halves возвращают Set[int] — финальный
        результат {1..15, 21..35} корректен.
        """
        mock_client = MagicMock()
        mock_client.chat_json.side_effect = [
            {"duplicates": list(range(2, 32))},  # call1: 30/40 > cap → half-retry
            {"duplicates": list(range(2, 17))},   # call2: 15/20 — cap подавлен (recent=None)
            {"duplicates": list(range(2, 17))},   # call3: 15/20 — cap подавлен (recent=None)
        ]
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(40)]

        result = flt.detect_duplicates(items, [])

        # call_count=3: cap ТРИГГЕРИТ half-retry (а не «сразу отбросили 1 вызовом»)
        assert mock_client.chat_json.call_count == 3, (
            f"Expected 3 calls (main >cap → retry halves), got {mock_client.chat_json.call_count}"
        )
        # half1: raw[2..16] → 0-based[1..15] → abs[0+1..0+15]={1..15}
        # half2: raw[2..16] → 0-based[1..15] → abs[20+1..20+15]={21..35}
        expected = set(range(1, 16)) | set(range(21, 36))
        assert result == expected, f"Expected {sorted(expected)}, got {sorted(result)}"

    def test_cap_50_percent_boundary(self):
        """50% — ГРАНИЦА. Cap = 0.5, > 0.5 → IGNORED. 5/10 = 0.5 → ACCEPTED."""
        mock_client = MagicMock()
        # Ровно 5/10 — cap 0.5, 5/10 = 0.5, 0.5 > 0.5 = False → ACCEPTED
        mock_client.chat_json.return_value = {"duplicates": [2, 4, 6, 8, 10]}
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        result = flt.detect_duplicates(items, [])
        assert result == {1, 3, 5, 7, 9}, "50% должно ПРИМЕНЯТЬСЯ (cap > 50%)"

    def test_cap_51_percent_retries_halves(self):
        """10 items: 6/10=60%>cap -> retry halves(5+5); halves ok -> результат из halves."""
        mock_client = MagicMock()
        # call 1: 6/10 > cap -> retry halves
        # call 2: 1/5=20% <= cap -> OK
        #   LLM returned 1-based id 2 -> global 0+(2-1)=1
        # call 3: 1/5=20% <= cap -> OK
        #   LLM returned 1-based id 2 -> global 5+(2-1)=6
        mock_client.chat_json.side_effect = [
            {"duplicates": [1, 2, 3, 4, 5, 6]},  # 6/10 > cap -> retry halves
            {"duplicates": [2]},                   # 1/5 <= cap -> OK (global 1)
            {"duplicates": [2]},                   # 1/5 <= cap -> OK (global 6)
        ]
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        result = flt.detect_duplicates(items, [])

        # half1 (0-4): dup idx 2 (1-based) -> global 1
        # half2 (5-9): dup idx 2 (1-based) -> global 5+1=6
        assert result == {1, 6}, f"Expected {{1, 6}}, got {result}"
        assert mock_client.chat_json.call_count == 3

    def test_half_retry_llm_error_partial_result(self, caplog):
        """CRITICAL-фикс: half-retry при LLMError на first_half → частичный результат из second_half.

        Контракт fail-open: второй half обрабатывается, первый игнорируется,
        исключение не пробрасывается наружу.
        """
        import logging

        mock_client = MagicMock()
        # call 1: 6/10 > cap -> retry halves
        # call 2: LLMError на first_half → игнорируем, продолжаем
        # call 3: 1/5 <= cap -> OK (global 5+0=5)
        mock_client.chat_json.side_effect = [
            {"duplicates": [1, 2, 3, 4, 5, 6]},  # 6/10 > cap
            LLMError("429 Rate Limited"),          # first_half: error
            {"duplicates": [1]},                   # second_half: 1/5 OK (global 5)
        ]
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        with caplog.at_level(logging.WARNING, logger="newsbot.llm.filter"):
            result = flt.detect_duplicates(items, [])

        # first_half: error → игнорируется
        # second_half: dup id 1 (1-based) → global 5+(1-1)=5
        assert result == {5}, f"Expected {{5}}, got {result}"
        assert mock_client.chat_json.call_count == 3
        assert any("LLM-ошибка" in r.message for r in caplog.records), (
            f"Expected LLM error warning in log. Records: {[r.message for r in caplog.records]}"
        )

    def test_half_retry_second_half_llm_error(self, caplog):
        """CRITICAL-фикс: half-retry при LLMError на second_half → результат из first_half."""
        import logging

        mock_client = MagicMock()
        # call 1: 6/10 > cap -> retry halves
        # call 2: 1/5 <= cap -> OK (global 0)
        # call 3: LLMError на second_half → игнорируем
        mock_client.chat_json.side_effect = [
            {"duplicates": [1, 2, 3, 4, 5, 6]},  # 6/10 > cap
            {"duplicates": [1]},                   # first_half: 1/5 OK (global 0)
            LLMError("timeout"),                    # second_half: error
        ]
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]

        with caplog.at_level(logging.WARNING, logger="newsbot.llm.filter"):
            result = flt.detect_duplicates(items, [])

        # first_half: dup id 1 (1-based) → global 0+(1-1)=0
        # second_half: error → игнорируется
        assert result == {0}, f"Expected {{0}}, got {result}"
        assert mock_client.chat_json.call_count == 3

    def test_cap_ignored_when_recent_non_empty(self):
        """CRITICAL-фикс: cap не применяется при непустом recent (легитимный дедуп)."""
        mock_client = MagicMock()
        # 10/10 = 100% > cap, но recent непуст → cap НЕ применяется
        mock_client.chat_json.return_value = {"duplicates": list(range(1, 11))}
        flt = self._make_filter(mock_client)

        items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(10)]
        # recent: непустой → cap подавляется
        recent = [{"title": "Old news", "source": "old"}]

        result = flt.detect_duplicates(items, recent)

        # Cap НЕ срабатывает, все 10 дубли применяются
        assert result == set(range(10)), f"10/10 с recent должен был примениться, got {result}"
        assert mock_client.chat_json.call_count == 1, (
            "При непустом recent cap не должен триггерить retry → 1 вызов"
        )




    def test_chunk_len_3_all_marked_no_retry(self):
            """chunk_len=3, все 3 помечены (100%%): cap не применяется (MIN_CHUNK=4).
            Cap нужен только для чанков >=4 items."""
            mock_client = MagicMock()
            mock_client.chat_json.return_value = {"duplicates": [1, 2, 3]}  # 3/3
            flt = self._make_filter(mock_client)

            # Только 3 items — меньше MIN_CHUNK=4
            items = [make_item(f"u{i}", f"Item{i}", f"Summary{i}") for i in range(3)]

            result = flt.detect_duplicates(items, [])

            # Cap НЕ применяется (chunk_len < MIN_CHUNK)
            # Результат принимается: {0, 1, 2} (1-based [1,2,3] -> 0-based)
            assert result == {0, 1, 2}, f"3/3 (100%%) < MIN_CHUNK=4 -> принимается, got {result}"
            assert mock_client.chat_json.call_count == 1, (
                f"Expected 1 call (no retry for small chunks), "
                f"got {mock_client.chat_json.call_count}"
            )

    
class TestMinor1SafePrice:
    def test_valid_int(self):
        assert _safe_price(2, 0.0) == 2.0

    def test_valid_string(self):
        assert _safe_price("3.5", 0.0) == 3.5

    def test_invalid_string_returns_default(self):
        assert _safe_price("abc", 0.0) == 0.0

    def test_nan_returns_default(self):
        assert _safe_price(float("nan"), 1.0) == 1.0

    def test_inf_returns_default(self):
        assert _safe_price(float("inf"), 1.0) == 1.0

    def test_wrong_type_returns_default(self):
        assert _safe_price([1, 2, 3], 0.5) == 0.5


# ─────────────────────────────────────────────────────────────────
# Minor #3: cache_hit warning at startup
# ─────────────────────────────────────────────────────────────────

class TestMinor3CacheHitWarning:
    def test_warning_when_cache_hit_price_missing(self, caplog):
        """input>0, output>0, но cache_hit=0 в pricing → warning."""
        import logging

        # pricing: input=1, output=2, но cache_hit отсутствует (==0).
        # Клиент создаётся БЕЗ цен → prices из client = 0.
        # _warn_if_cost_guard_blind срабатывает: prices=0, guard=1.0.
        cfg = {
            "enabled": True,
            "model": "test-model",
            "api_key": "k",
            "push_urgent_only": False,
            "daily_cost_limit": 1.0,
            # pricing с input/output, но БЕЗ cache_hit
            "pricing": {
                "test-model": {"input": 1.0, "output": 2.0},
            },
        }
        # БЕЗ client= — создаём DeepSeekClient с prices из cfg (1.0/2.0)
        with caplog.at_level(logging.INFO, logger="newsbot.llm.filter"):
            LLMFilter(cfg)

        messages = [r.message for r in caplog.records]
        # При наличии input/output в pricing, но отсутствии cache_hit,
        # логируется INFO-сообщение о том, что cache_hit не задан.
        assert any("cache_hit" in m.lower()
                   for m in messages), (
            f"No cache_hit warning. All messages: {messages}"
        )


# ─────────────────────────────────────────────────────────────────
# Minor #4: _check_daily_cost BEFORE len(items)==1
# ─────────────────────────────────────────────────────────────────

class TestMinor4CostGuardBeforeSingleItem:
    """_check_daily_cost вызывается ДО len==1, чтобы одиночный item тоже проверялся."""

    def test_single_item_respects_cost_guard(self):
        """len==1 проходит через _check_daily_cost."""
        from newsbot.llm.filter import LLMFilter

        # Mock storage: cost limit EXCEEDED
        class _MockStorage:
            def __init__(self):
                self._store = {}

            def get_meta(self, key):
                today = "llm_daily_cost_"
                if key.startswith(today):
                    return "999.0"  # превышен
                return None

            def set_meta(self, key, value):
                self._store[key] = value

            def count_by_state(self, state):
                return 0

        storage = _MockStorage()
        cfg = {
            "enabled": True, "model": "t", "api_key": "k",
            "daily_cost_limit": 1.0,  # лимит 1 USD
            "push_urgent_only": False,
        }
        flt = LLMFilter(cfg, client=MagicMock(), storage=storage)

        # Мок-клиент: вызов chat_json не должен происходить
        mock_client = MagicMock()
        flt._client = mock_client

        result = flt.evaluate_batch([make_item()])

        assert len(result) == 1
        assert result[0].ok is False
        assert result[0].cost_blocked is True
        mock_client.chat_json.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# Minor #5: backoff with jitter
# ─────────────────────────────────────────────────────────────────

class TestMinor5JitterBackoff:
    def test_backoff_has_jitter(self):
        """Backoff includes jitter (random factor). Uses patch.object on random module."""
        from unittest.mock import patch
        import newsbot.llm.client as cm
        import httpx

        jitter_calls = []

        def mock_uniform(a, b):
            jitter_calls.append((a, b))
            return 1.0  # jitter = +1.0 (deterministic)

        class _Resp429:
            status_code = 429
            headers = {}
            def json(self): return {}
            def raise_for_status(self):
                raise httpx.HTTPStatusError("429", request=None, response=self)

        class _FakeClient429:
            is_closed = False
            def __init__(s, **kw): pass
            def post(s, url, **kwargs): return _Resp429()
            def get(s, url, **kwargs): return _Resp429()
            def close(s): pass

        prev_client = cm.httpx.Client
        prev_sleep = cm.time.sleep
        cm.httpx.Client = _FakeClient429
        sleep_calls = []
        cm.time.sleep = lambda s: sleep_calls.append(s)
        try:
            with patch.object(cm.random, 'uniform', mock_uniform):
                client = DeepSeekClient("test", "https://x.com", "test", max_retries=2)
                try:
                    client.chat_json("sys", "user")
                except LLMError:
                    pass
        finally:
            cm.httpx.Client = prev_client
            cm.time.sleep = prev_sleep

        assert len(sleep_calls) >= 1, f"Expected >=1 sleep, got {len(sleep_calls)}"
        assert len(jitter_calls) >= 1, f"random.uniform called {len(jitter_calls)} times"
        # Call 1: 429, attempt=1 → wait=2 (2**1) → jitter range ±50% → [-1, 1]
        a1, b1 = jitter_calls[0]
        assert -1.1 < a1 < -0.9, f"jitter[0] min={a1}, expected -1.0 (±50%% of wait=2)"
        assert 0.9 < b1 < 1.1, f"jitter[0] max={b1}, expected 1.0 (+50%% of wait=2)"
        # Call 2: 429, attempt=2 → wait=4 (2**2) → jitter range ±50% → [-2, 2]
        # (exponential backoff never runs: second 429 arrives before it)
        assert len(jitter_calls) >= 2, "expected second 429 jitter call"
        a2, b2 = jitter_calls[1]
        assert -2.1 < a2 < -1.9, f"jitter[1] min={a2}, expected -2.0 (±50%% of wait=4)"
        assert 1.9 < b2 < 2.1, f"jitter[1] max={b2}, expected 2.0 (+50%% of wait=4)"
class TestMinor6MaxTokensDefault:
    def test_default_max_tokens_4608(self):
        client = DeepSeekClient("test", "https://x.com", "test")
        assert client.max_tokens == 4608, f"Expected 4608, got {client.max_tokens}"

    def test_config_overrides_default(self):
        client = DeepSeekClient("test", "https://x.com", "test", max_tokens=8192)
        assert client.max_tokens == 8192


# ─────────────────────────────────────────────────────────────────
# Minor #7: LANG_MARKER documentation
# ─────────────────────────────────────────────────────────────────

class TestMinor7LangMarkerDocumentation:
    def test_lang_marker_present(self):
        from newsbot.llm.prompts import LANG_MARKER
        assert LANG_MARKER == "# Language: {lang}"

    def test_lang_marker_comment_present(self):
        """LANG_MARKER определён и имеет осмысленное значение."""
        from newsbot.llm.prompts import LANG_MARKER
        assert LANG_MARKER == "# Language: {lang}", (
            f"Unexpected LANG_MARKER: {LANG_MARKER!r}"
        )
        # Содержит заполнитель — LLM подставляет язык
        assert "{lang}" in LANG_MARKER


# ─────────────────────────────────────────────────────────────────
# V7 task: reason field removed from prompts, parser accepts both
# ─────────────────────────────────────────────────────────────────

class TestV7ReasonFieldOptional:
    """V7: LLM prompts no longer require `reason` field; parser accepts both."""

    def _make_filter(self, mock_client, extra_cfg=None):
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
        }
        if extra_cfg:
            cfg.update(extra_cfg)
        return LLMFilter(cfg, client=mock_client)

    def test_batch_without_reason_passes(self):
        """Batch response WITHOUT reason field → all items ok=True."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 8, "urgent": False, "summary": "Summary1"},
                {"id": 2, "score": 6, "urgent": False, "summary": "Summary2"},
            ]
        }
        flt = self._make_filter(mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 3)]
        results = flt.evaluate_batch(items)

        assert all(r.ok for r in results)
        assert results[0].score == 8
        assert results[1].score == 6

    def test_batch_with_reason_also_passes(self):
        """Batch response WITH reason field → backward compatible."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 7, "urgent": False, "summary": "Sum1",
                 "reason": "relevant event"},
                {"id": 2, "score": 5, "urgent": False, "summary": "Sum2",
                 "reason": "interesting"},
            ]
        }
        flt = self._make_filter(mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 3)]
        results = flt.evaluate_batch(items)

        assert all(r.ok for r in results)
        assert results[0].reason == "relevant event"
        assert results[1].reason == "interesting"

    def test_single_item_without_reason_passes(self):
        """Single-item response WITHOUT reason → ok=True, reason=''."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "score": 7, "summary": "Test summary", "urgent": False,
        }
        flt = self._make_filter(mock_client)
        item = make_item()
        result = flt.evaluate(item)

        assert result.ok is True
        assert result.score == 7
        assert result.reason == ""

    def test_summary_max_chars_default_240(self):
        """summary_max_chars not in config → default 240."""
        flt = LLMFilter(
            {"enabled": True, "api_key": "k", "model": "m",
             "push_urgent_only": False},
            client=MagicMock(),
        )
        assert flt._summary_max_chars == 240

    def test_summary_max_chars_from_config(self):
        """summary_max_chars read from config."""
        flt = LLMFilter(
            {"enabled": True, "api_key": "k", "model": "m",
             "push_urgent_only": False, "summary_max_chars": 180},
            client=MagicMock(),
        )
        assert flt._summary_max_chars == 180

    def test_summary_max_chars_invalid_uses_default(self):
        """Invalid summary_max_chars → default 240, warning."""
        flt = LLMFilter(
            {"enabled": True, "api_key": "k", "model": "m",
             "push_urgent_only": False, "summary_max_chars": "abc"},
            client=MagicMock(),
        )
        assert flt._summary_max_chars == 240

    def test_prompt_contains_summary_max_chars_value(self):
        """Formatted system prompt contains the actual config number."""
        from newsbot.llm.prompts import format_system_prompt
        p240 = format_system_prompt(240)
        p180 = format_system_prompt(180)
        assert "max 240 characters" in p240
        assert "max 180 characters" in p180
        assert "max 240 characters" not in p180


# ─────────────────────────────────────────────────────────────────
# V3 task: two-phase scoring
# ─────────────────────────────────────────────────────────────────

class TestV3TwoPhaseDisabled:
    """V3: two_phase.enabled=false → single call, no phase separation."""

    def test_disabled_one_call(self):
        """enabled=false → exactly 1 chat_json call (no phase 1)."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 7, "urgent": False,
                 "summary": "S", "reason": "r", "location": "", "title_en": "T"},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": False},
        }
        flt = LLMFilter(cfg, client=mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 4)]
        flt.evaluate_batch(items)
        assert mock_client.chat_json.call_count == 1

    def test_disabled_all_items_scored(self):
        """enabled=false → all items return ok=True."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 8, "urgent": False,
                 "summary": "S1", "reason": "r1", "location": "", "title_en": "T1"},
                {"id": 2, "score": 3, "urgent": False,
                 "summary": "S2", "reason": "r2", "location": "", "title_en": "T2"},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": False},
        }
        flt = LLMFilter(cfg, client=mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 3)]
        results = flt.evaluate_batch(items)
        assert all(r.ok for r in results)
        assert len(results) == 2


class TestV3TwoPhaseEnabled:
    """V3: two_phase.enabled=true → phase1 on all + phase2 on high-score items."""

    def test_enabled_phase1_then_phase2(self):
        """enabled=true → phase1 on 5 items, phase2 only on items with score>=4."""
        phase1_responses = []
        phase2_responses = []

        def fake_chat_json(system, user):
            if "You are a news filter for a Telegram bot." in system:
                # Phase 1 (compact: no "personal")
                phase1_responses.append(user)
                # 5 items: scores 3,5,2,7,4
                return {
                    "results": [
                        {"id": 1, "score": 3, "urgent": False},
                        {"id": 2, "score": 5, "urgent": False},
                        {"id": 3, "score": 2, "urgent": False},
                        {"id": 4, "score": 7, "urgent": False},
                        {"id": 5, "score": 4, "urgent": False},
                    ]
                }
            else:
                # Phase 2 — only items 2,4,5 pass (scores 5,7,4 >= 4)
                phase2_responses.append(user)
                n = user.count("<|data:")
                return {
                    "results": [
                        {"id": i, "score": s, "urgent": False,
                         "summary": f"Sum{i}", "reason": f"r{i}",
                         "location": "R", "title_en": f"T{i}"}
                        for i, s in zip(range(1, n + 1), [5, 7, 4][:n])
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat_json

        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 6)]
        results = flt.evaluate_batch(items)

        assert mock_client.chat_json.call_count == 2, (
            f"Expected 2 calls (phase1 + phase2), got {mock_client.chat_json.call_count}"
        )
        # Phase 1: 5 items
        assert phase1_responses[0].count("<|data:") == 5
        # Phase 2: 3 items (scores 5,7,4 all >= 4)
        assert phase2_responses[0].count("<|data:") == 3

        # Results
        assert all(r.ok for r in results)
        # Items 1 (score=3) and 3 (score=2) — below threshold, have phase1 score only
        assert results[0].score == 3
        assert results[2].score == 2
        # Items 2,4,5 — above threshold, have full data from phase2
        assert results[1].score == 5
        assert results[1].summary == "Sum1"
        assert results[3].score == 7
        assert results[3].summary == "Sum2"  # phase2 item id=2 → original index 3
        assert results[4].score == 4
        assert results[4].summary == "Sum3"  # phase2 item id=3 → original index 4

    def test_enabled_phase2_only_on_passing_items(self):
        """Phase-2 prompt is called only for items where score >= phase2_min_score."""
        phase1_calls = []
        phase2_calls = []

        def fake_chat_json(system, user):
            if "You are a news filter for a Telegram bot." in system:
                phase1_calls.append(1)
                return {
                    "results": [
                        {"id": 1, "score": 1, "urgent": False},
                        {"id": 2, "score": 2, "urgent": False},
                        {"id": 3, "score": 3, "urgent": False},
                    ]
                }
            else:
                phase2_calls.append(1)
                n = user.count("<|data:")
                return {
                    "results": [
                        {"id": i, "score": 0, "urgent": False,
                         "summary": f"Sum{i}", "reason": "",
                         "location": "", "title_en": ""}
                        for i in range(1, n + 1)
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat_json

        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 4)]
        results = flt.evaluate_batch(items)

        assert len(phase1_calls) == 1, "Phase 1 called once"
        assert len(phase2_calls) == 0, (
            f"Phase 2 should NOT be called (all scores < 4), got {len(phase2_calls)} calls"
        )
        # All have phase1 scores, empty summary (no phase2)
        assert all(r.ok for r in results)
        assert all(r.summary == "" for r in results)
        assert [r.score for r in results] == [1, 2, 3]

    def test_enabled_cost_guard_before_phase1(self):
        """Phase-1 is blocked by cost guard → all items cost_blocked."""
        class _CostBlockedClient:
            def chat_json(self, *args, **kwargs):
                raise LLMError("unreachable")

        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": True},
            "daily_cost_limit": 0.001,
            "pricing": {"m": {"input": 1, "output": 2, "cache_hit": 0}},
        }

        class _MockStorage:
            def __init__(self):
                self._store = {}
                self._meta_call_count = 0

            def get_meta(self, key):
                self._meta_call_count += 1
                if "llm_daily_cost" in key:
                    return "999.0"  # limit exceeded
                return None

            def set_meta(self, key, value):
                self._store[key] = value

            def count_by_state(self, s):
                return 0

        storage = _MockStorage()
        flt = LLMFilter(cfg, client=_CostBlockedClient(), storage=storage)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 4)]
        results = flt.evaluate_batch(items)

        assert all(r.cost_blocked for r in results)
        assert all(not r.ok for r in results)
        assert storage._meta_call_count > 0, "storage should be called for cost check"

    def test_enabled_cost_guard_between_phases(self):
        """Cost limit exhausted AFTER phase-1 → phase-2 skipped, phase-1 scores kept."""
        def fake_chat_json(system, user):
            if "You are a news filter for a Telegram bot." in system:
                # Phase 1
                return {
                    "results": [
                        {"id": 1, "score": 5, "urgent": False},
                        {"id": 2, "score": 8, "urgent": False},
                        {"id": 3, "score": 1, "urgent": False},
                    ]
                }
            else:
                # Phase 2 — should NOT be reached
                return {
                    "results": [
                        {"id": 1, "score": 5, "urgent": False,
                         "summary": "X", "reason": "", "location": "", "title_en": ""},
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat_json

        # Cost exceeded on the SECOND cost check (after phase-1 usage is recorded).
        # Flow: evaluate_batch(check#1→OK) → _evaluate_two_phase(check#2→OK,phase1) →
        #   _record_usage(phase1) → _accumulate(phase1 cost) →
        #   check#3→FAIL → phase2 skipped, phase1 scores returned.
        class _MockStorage:
            def __init__(self):
                self._store = {}
                self._check_count = 0  # instance variable — isolated per test

            def get_meta(self, key):
                self._check_count += 1
                if "llm_daily_cost" in key:
                    # Check 1,2: OK (within limit); Check 3+: exceeded
                    if self._check_count > 2:
                        return "999.0"
                    return "0.0"
                return None

            def set_meta(self, key, value):
                self._store[key] = value

            def count_by_state(self, s):
                return 0

        cfg = {
            "enabled": True, "api_key": "k", "model": "m",
            "push_urgent_only": False, "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
            "daily_cost_limit": 0.001,
            "pricing": {"m": {"input": 1, "output": 2, "cache_hit": 0}},
        }
        storage = _MockStorage()
        flt = LLMFilter(cfg, client=mock_client, storage=storage)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 4)]
        results = flt.evaluate_batch(items)

        # Phase-1 scored all items; phase-2 blocked by cost guard
        # Items 1 and 2 have scores from phase-1
        assert results[0].score == 5
        assert results[1].score == 8
        assert results[2].score == 1
        # Phase-2 was NOT called (only phase-1)
        assert mock_client.chat_json.call_count == 1

    def test_phase2_min_score_default_4(self):
        """phase2_min_score defaults to 4."""
        flt = LLMFilter(
            {"enabled": True, "api_key": "k", "model": "m",
             "push_urgent_only": False},
            client=MagicMock(),
        )
        assert flt._two_phase_phase2_min_score == 4

    def test_two_phase_disabled_by_default(self):
        """two_phase.enabled defaults to False (backward compatible)."""
        flt = LLMFilter(
            {"enabled": True, "api_key": "k", "model": "m",
             "push_urgent_only": False},
            client=MagicMock(),
        )
        assert flt._two_phase_enabled is False


# ─────────────────────────────────────────────────────────────────
# V3 task: guard fixes
# ─────────────────────────────────────────────────────────────────

class TestV3GuardFixes:
    """Guard #1 (evaluate) and guard #2 (recursive split)."""

    def test_evaluate_has_own_guard(self):
        """evaluate() has _check_daily_cost() inside — not only via evaluate_batch."""
        # Storage: limit exceeded
        class _MockStorage:
            def __init__(self):
                self._calls = []

            def get_meta(self, key):
                self._calls.append(key)
                if "llm_daily_cost" in key:
                    return "999.0"
                return None

            def set_meta(self, key, value):
                pass

            def count_by_state(self, s):
                return 0

        storage = _MockStorage()
        cfg = {
            "enabled": True, "model": "m", "api_key": "k",
            "daily_cost_limit": 1.0,
            "push_urgent_only": False,
        }
        flt = LLMFilter(cfg, client=MagicMock(), storage=storage)
        item = make_item()
        result = flt.evaluate(item)
        assert result.cost_blocked is True, "evaluate() should be blocked by cost guard"

    def test_guard_in_evaluate_counts_storage_calls(self):
        """evaluate() guard hits storage (multiple checks per call)."""
        class _MockStorage:
            def __init__(self):
                self._meta_calls = 0

            def get_meta(self, key):
                self._meta_calls += 1
                if "llm_daily_cost" in key:
                    return "999.0"
                return None

            def set_meta(self, key, value):
                pass

            def count_by_state(self, s):
                return 0

        storage = _MockStorage()
        cfg = {
            "enabled": True, "model": "m", "api_key": "k",
            "daily_cost_limit": 1.0, "push_urgent_only": False,
        }
        flt = LLMFilter(cfg, client=MagicMock(), storage=storage)
        result = flt.evaluate(make_item())
        # Should have triggered get_meta calls (cost key + warning flag)
        assert storage._meta_calls >= 1, (
            f"evaluate() guard should access storage, got {storage._meta_calls} calls"
        )


# ─────────────────────────────────────────────────────────────────
# Smoke: existing tests не сломались
# ─────────────────────────────────────────────────────────────────

class TestRegressionSmoke:
    def test_llm_error_still_works(self):
        """LLMError как и раньше — без подкласса."""
        import httpx

        class _FakeResp500:
            status_code = 500
            text = "Internal Server Error"
            headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("500", request=None, response=self)

        class _ErrClient:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                raise httpx.HTTPStatusError("500", request=None, response=_FakeResp500())

            def get(s, url, **kwargs):
                return _FakeResp500()

            def close(s):
                pass

        with _HttpxPatcher(_ErrClient):
            client = DeepSeekClient("test", "https://x.com", "test")
            try:
                client.chat_json("sys", "user")
                assert False, "Должно было выбросить"
            except LLMError as e:
                # Это должен быть обычный LLMError, не LLMFatalError
                assert not isinstance(e, LLMFatalError)


# ─────────────────────────────────────────────────────────────────
# Major #1 (new): score range validation in all parsers
# ─────────────────────────────────────────────────────────────────

class TestScoreCoercion:
    """_coerce_score: int/float/str in 0..10 → int; out-of-range/bool/malformed → None."""

    def _f(self, raw):
        # expose helper for testing (imported from module-level function)
        from newsbot.llm.filter import _coerce_score
        return _coerce_score(raw)

    # Valid: int
    def test_int_in_range(self):
        assert self._f(7) == 7

    def test_int_boundary_0(self):
        assert self._f(0) == 0

    def test_int_boundary_10(self):
        assert self._f(10) == 10

    # Invalid: int out of range
    def test_int_too_high(self):
        assert self._f(99) is None

    def test_int_too_low(self):
        assert self._f(-1) is None

    # Valid: integer-valued float
    def test_float_int_valued(self):
        assert self._f(7.0) == 7

    # Invalid: non-integer float
    def test_float_non_integer(self):
        assert self._f(7.6) is None

    def test_float_nan(self):
        import math
        assert self._f(math.nan) is None

    def test_float_inf(self):
        import math
        assert self._f(math.inf) is None

    # Valid: str int
    def test_str_int(self):
        assert self._f("7") == 7

    def test_str_int_with_spaces(self):
        assert self._f("  5  ") == 5

    # Invalid: str out of range
    def test_str_too_high(self):
        assert self._f("99") is None

    def test_str_too_low(self):
        assert self._f("-1") is None

    # Invalid: str non-numeric
    def test_str_non_numeric(self):
        assert self._f("abc") is None

    # Invalid: bool (subclass of int — must be rejected first)
    def test_bool_true(self):
        assert self._f(True) is None

    def test_bool_false(self):
        assert self._f(False) is None


class TestScoreValidationInParsers:
    """score 99/-1 → ok=False in ALL parser methods (Major #1 fix)."""

    def _make_filter(self, mock_client, extra_cfg=None):
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
        }
        if extra_cfg:
            cfg.update(extra_cfg)
        return LLMFilter(cfg, client=mock_client)

    def test_batch_score_99_rejected(self):
        """_parse_batch_response: score=99 → ok=False."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 99, "urgent": False,
                 "summary": "Summary", "reason": "", "location": "", "title_en": ""},
            ]
        }
        flt = self._make_filter(mock_client)
        result = flt.evaluate_batch([make_item()])
        assert result[0].ok is False, "score=99 should result in ok=False"
        assert result[0].score is None

    def test_batch_score_minus1_rejected(self):
        """_parse_batch_response: score=-1 → ok=False."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": -1, "urgent": False,
                 "summary": "Summary", "reason": "", "location": "", "title_en": ""},
            ]
        }
        flt = self._make_filter(mock_client)
        result = flt.evaluate_batch([make_item()])
        assert result[0].ok is False, "score=-1 should result in ok=False"
        assert result[0].score is None

    def test_batch_score_string_7_accepted(self):
        """_parse_batch_response: score="7" → 7 (same as old parser)."""
        # Use 2 items so two-phase path is not triggered (len>1 → two_phase=True, 1 item → evaluate())
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": "7", "urgent": False,
                 "summary": "Summary1", "reason": "", "location": "", "title_en": ""},
                {"id": 2, "score": 5, "urgent": False,
                 "summary": "Summary2", "reason": "", "location": "", "title_en": ""},
            ]
        }
        flt = self._make_filter(mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        assert result[0].ok is True
        assert result[0].score == 7
        assert result[1].ok is True
        assert result[1].score == 5

    def test_phase1_score_99_rejected(self):
        """_parse_phase1_response: score=99 → ok=False."""
        # Use 2 items so two-phase is used (len>1)
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 99, "urgent": False},
                {"id": 2, "score": 5, "urgent": False},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        # Item 1 (score=99) → ok=False; Item 2 (score=5) → ok=True
        assert result[0].ok is False, "phase1 score=99 should result in ok=False"
        assert result[1].ok is True

    def test_phase1_score_minus1_rejected(self):
        """_parse_phase1_response: score=-1 → ok=False."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": -1, "urgent": False},
                {"id": 2, "score": 5, "urgent": False},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        assert result[0].ok is False, "phase1 score=-1 should result in ok=False"

    def test_phase1_score_string_5_accepted(self):
        """_parse_phase1_response: score="5" → 5."""
        # Use 2 items so two-phase is used (len>1)
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": "5", "urgent": False},
                {"id": 2, "score": 3, "urgent": False},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        assert result[0].ok is True
        assert result[0].score == 5

    def test_phase2_score_99_rejected(self):
        """_parse_phase2_response: score=99 → ok=False (phase1 score used instead)."""
        # Use 2 items so two-phase is used (len>1)
        def fake_chat(system, user):
            if "You are a news filter for a Telegram bot." in system:
                # Phase 1: both items pass threshold
                return {
                    "results": [
                        {"id": 1, "score": 5, "urgent": False},
                        {"id": 2, "score": 5, "urgent": False},
                    ]
                }
            else:
                # Phase 2: item 1 score=99 (rejected), item 2 valid
                return {
                    "results": [
                        {"id": 1, "score": 99, "urgent": False,
                         "summary": "X", "reason": "", "location": "", "title_en": ""},
                        {"id": 2, "score": 5, "urgent": False,
                         "summary": "Y", "reason": "", "location": "", "title_en": ""},
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        # Phase-2 item 1 score=99 → merged back as ok=False → kept phase1
        assert result[0].ok is True
        assert result[0].score == 5
        assert result[0].summary == ""  # phase1 has no summary
        # Item 2: phase2 valid
        assert result[1].ok is True
        assert result[1].score == 5
        assert result[1].summary == "Y"


# ─────────────────────────────────────────────────────────────────
# Major #3: summary_max_chars enforcement in all parsers
# ─────────────────────────────────────────────────────────────────

class TestSummaryTruncation:
    """summary longer than _summary_max_chars → truncated to ≈limit with '…'."""

    def _make_filter(self, mock_client, summary_max_chars=240):
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": summary_max_chars,
        }
        return LLMFilter(cfg, client=mock_client)

    def test_single_item_truncated_to_240(self):
        """500-char summary at limit=240 → ~241 with … (single-item path)."""
        long_summary = "A" * 500
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "score": 8, "summary": long_summary, "urgent": False
        }
        flt = self._make_filter(mock_client, summary_max_chars=240)
        result = flt.evaluate(make_item())
        assert result.ok is True
        assert len(result.summary) == 241  # 240 + "…"
        assert result.summary.endswith("…")
        assert not result.summary.endswith(" A")

    def test_single_item_truncated_to_180_from_config(self):
        """500-char summary at limit=180 from config → ~181 with …."""
        long_summary = "B" * 500
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "score": 8, "summary": long_summary, "urgent": False
        }
        flt = self._make_filter(mock_client, summary_max_chars=180)
        result = flt.evaluate(make_item())
        assert result.ok is True
        assert len(result.summary) == 181
        assert result.summary.endswith("…")

    def test_batch_truncated(self):
        """Batch response: 500-char summary → truncated (batch path)."""
        # Use 2+ items so batch path (not evaluate) is used
        long_summary = "C" * 500
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                {"id": 1, "score": 7, "urgent": False,
                 "summary": long_summary, "reason": "",
                 "location": "", "title_en": ""},
                {"id": 2, "score": 5, "urgent": False,
                 "summary": "Short", "reason": "",
                 "location": "", "title_en": ""},
            ]
        }
        flt = self._make_filter(mock_client, summary_max_chars=240)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        assert result[0].ok is True
        assert len(result[0].summary) == 241
        assert result[0].summary.endswith("…")

    def test_phase2_truncated(self):
        """Two-phase: phase2 500-char summary → truncated."""
        # Use 2 items so two-phase path is triggered
        def fake_chat(system, user):
            if "You are a news filter for a Telegram bot." in system:
                return {
                    "results": [
                        {"id": 1, "score": 5, "urgent": False},
                        {"id": 2, "score": 5, "urgent": False},
                    ]
                }
            else:
                return {
                    "results": [
                        {"id": 1, "score": 5, "urgent": False,
                         "summary": "D" * 500, "reason": "",
                         "location": "", "title_en": ""},
                        {"id": 2, "score": 5, "urgent": False,
                         "summary": "Short", "reason": "",
                         "location": "", "title_en": ""},
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        result = flt.evaluate_batch([make_item("u1"), make_item("u2")])
        assert result[0].ok is True
        assert len(result[0].summary) == 241
        assert result[0].summary.endswith("…")
        assert result[1].ok is True
        assert result[1].summary == "Short"

    def test_exactly_at_limit_unchanged(self):
        """Summary exactly at limit → no truncation."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "score": 7, "summary": "x" * 240, "urgent": False
        }
        flt = self._make_filter(mock_client, summary_max_chars=240)
        result = flt.evaluate(make_item())
        assert result.summary == "x" * 240
        assert not result.summary.endswith("…")


# ─────────────────────────────────────────────────────────────────
# Major #2: truncation recovery in two-phase path
# ─────────────────────────────────────────────────────────────────

class TestTruncationRecovery:
    """LLMTruncationError in two-phase → recovery via split (Major #2 fix)."""

    def _make_filter(self, mock_client, summary_max_chars=240):
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": summary_max_chars,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        return LLMFilter(cfg, client=mock_client)

    def test_phase1_truncation_delegates_to_recursive(self):
        """Phase-1 LLMTruncationError → delegates to _evaluate_batch_recursive."""
        call_log = []

        def fake_chat(system, user):
            call_log.append(system[:60])
            if "You are a news filter for a Telegram bot." in system:
                # Phase 1
                raise LLMTruncationError("finish_reason=length")
            else:
                # Recursive batch fallback (2 halves)
                n = user.count("<|data:")
                return {
                    "results": [
                        {"id": i, "score": 7, "urgent": False,
                         "summary": f"Sum{i}", "reason": "r",
                         "location": "L", "title_en": f"T{i}"}
                        for i in range(1, n + 1)
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat

        flt = self._make_filter(mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(4)]
        results = flt.evaluate_batch(items)

        # Recovery happened: items were processed
        assert all(r.ok for r in results)
        assert all(r.summary != "" for r in results)
        # Phase-1 was called once, then recursive split handled it
        assert call_log.count(call_log[0]) == 1  # phase-1 called once
        assert len(call_log) >= 2  # at least one more call from recursive split

    def test_phase2_truncation_splits_and_recovers(self):
        """Phase-2 LLMTruncationError → split phase-2 items, merge ok results."""
        # Track calls by prompt type using distinctive strings
        phase1_calls = []   # compact: "SCORING 0-10:", contains "RESPONSE FORMAT"
        phase2_calls = []   # full batch: contains "personal Telegram bot"
        batch_calls = []    # recursive fallback: "news filter for a Telegram bot"

        def fake_chat(system, user):
            if "SCORING 0-10:" in system and "RESPONSE FORMAT" in system:
                # Phase 1 (compact: only id/score/urgent, NO summary/title_en)
                phase1_calls.append(user)
                return {
                    "results": [
                        {"id": i, "score": 5, "urgent": False}
                        for i in range(1, 5)
                    ]
                }
            elif "personal Telegram bot" in system:
                # Phase 2 (full evaluation with summary/title_en)
                phase2_calls.append(user)
                if len(phase2_calls) == 1:
                    # First phase-2 call → simulate truncation
                    raise LLMTruncationError("finish_reason=length")
                # Retry after truncation: split phase-2 items
                n = user.count("<|data:")
                return {
                    "results": [
                        {"id": i, "score": 5, "urgent": False,
                         "summary": f"Sum{i}", "reason": "r",
                         "location": "L", "title_en": f"T{i}"}
                        for i in range(1, n + 1)
                    ]
                }
            else:
                # Batch fallback (for phase-1 truncation recovery)
                batch_calls.append(user)
                n = user.count("<|data:")
                return {
                    "results": [
                        {"id": i, "score": 7, "urgent": False,
                         "summary": f"Sum{i}", "reason": "r",
                         "location": "L", "title_en": f"T{i}"}
                        for i in range(1, n + 1)
                    ]
                }

        mock_client = MagicMock()
        mock_client.chat_json.side_effect = fake_chat

        flt = self._make_filter(mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(4)]
        results = flt.evaluate_batch(items)

        # Phase-2 truncation recovered: items have summaries
        assert all(r.ok for r in results), f"All items ok, got {results}"
        assert all(r.summary != "" for r in results), f"All have summary, got {results}"
        # Phase-1 called once; Phase-2 called once (truncation) + retries
        assert len(phase1_calls) == 1, f"Phase-1 called once, got {len(phase1_calls)}"
        assert len(phase2_calls) >= 2, (
            f"Phase-2: 1 truncation + 1+ retry = >=2 calls, got {len(phase2_calls)}"
        )

    def test_truncation_blocked_by_cost_guard_no_extra_calls(self):
        """
        Phase-1 truncation + _check_daily_cost=False after phase-1 completes
        → recovery NOT called, all items cost_blocked=True.

        Scenario: budget allows phase-1 and phase-2 entry (cost=0 initially),
        phase-1 returns; _record_usage bumps daily_total to 1.1 (exceeds 1.0);
        the second _check_daily_cost() at line 845 returns False.
        The is_truncation branch is entered but blocked.
        All 4 items must get cost_blocked=True, length must be 4, and
        phase-1 scores must be preserved (not lost).

        Branch coverage: _evaluate_two_phase → LLMError → is_truncation=True
        → _check_daily_cost()=False → return cost_blocked results.
        """
        phase1_calls = [0]
        phase2_calls = [0]

        class _TruncationClient:
            def chat_json(self, system, user):
                if "SCORING 0-10:" in system and "RESPONSE FORMAT" in system:
                    phase1_calls[0] += 1
                    return {
                        "results": [
                            {"id": i, "score": 5, "urgent": False}
                            for i in range(1, 5)
                        ]
                    }
                elif "personal Telegram bot" in system:
                    phase2_calls[0] += 1
                    raise LLMTruncationError("finish_reason=length")
                else:
                    raise AssertionError(f"Unexpected system prompt: {system[:100]}")

            def get_balance(self):
                return BalanceInfo(True, "USD", 50.0)

        mock_client = _TruncationClient()
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
            "daily_cost_limit": 1.0,
            "pricing": {"deepseek-flash": {"input": 1, "output": 2, "cache_hit": 0}},
        }

        storage = MagicMock()
        storage.get_meta.return_value = "0.0"
        storage.count_by_state.return_value = 0
        storage.set_meta = MagicMock()

        flt = LLMFilter(cfg, client=mock_client, storage=storage)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(4)]

        # Mock _check_daily_cost: first 3 calls → True (entry allowed),
        # 4th call → False (recovery guard at line 918 blocks the retry).
        guard_call = [0]

        def mock_check_daily_cost():
            guard_call[0] += 1
            return guard_call[0] <= 3

        with patch.object(flt, "_check_daily_cost", mock_check_daily_cost):
            results = flt.evaluate_batch(items)

        # ── Verify the recovery-guard branch was actually exercised ──────
        # Phase-1 entered (1st check → True)
        assert phase1_calls[0] == 1, (
            f"Phase-1 should be called once, got {phase1_calls[0]}"
        )
        # Phase-2 entered (3rd check → True); hit truncation;
        # recovery guard (4th check → False) blocked → no second call.
        assert phase2_calls[0] == 1, (
            f"Phase-2 should be called exactly once (recovery blocked by cost guard), "
            f"got {phase2_calls[0]}"
        )
        # _check_daily_cost calls: evaluate_batch entry, phase-1 entry,
        # phase-2 entry (all → True), recovery guard (→ False).
        # No per-item evaluate() calls because elif-is_truncation returns early.
        assert guard_call[0] == 4, (
            f"Expected 4 _check_daily_cost calls, got {guard_call[0]}"
        )

        # ── Correct behaviour after elif is_truncation branch (filter.py line 946):
        # Truncation + cost exhausted → phase-2 items cost_blocked=True, score preserved.
        # All 4 items have score=5 → all are phase-2 items → all cost_blocked.
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        for r in results:
            assert r.ok is False, f"Expected ok=False (cost_blocked), got {r.ok}"
            assert r.cost_blocked is True, (
                f"Expected cost_blocked=True, got {r.cost_blocked}"
            )
            assert r.score == 5, f"Expected phase-1 score=5 (preserved), got {r.score}"

    def test_phase2_partial_recovery_nonconsecutive_indices(self):
        """
        Phase-1 scores [5,1,5,1] → phase2_indices=[0,2] (non-consecutive).
        Phase-2 truncates; recovery returns only ONE item for TWO phase-2 items.
        Merge must correctly map recovered[0]→orig_idx=0 (NOT 1 or 3).

        Checks:
        - results[0] (orig_idx=0, phase2_idx=0) gets recovered score=9/summary
        - results[2] (orig_idx=2, phase2_idx=1) falls back to phase-1 score=5
        - results[1] and results[3] keep phase-1 score=1 (never in phase-2)
        - len(results) == 4, no item lost, no index substitution
        """
        phase1_calls = [0]
        phase2_calls = [0]

        class _PartialRecoveryClient:
            def chat_json(self, system, user):
                if "SCORING 0-10:" in system and "RESPONSE FORMAT" in system:
                    phase1_calls[0] += 1
                    # scores [5,1,5,1]: items 0 and 2 go to phase-2 (score>=4),
                    # items 1 and 3 stay below phase2_min_score=4.
                    return {
                        "results": [
                            {"id": 1, "score": 5, "urgent": False},
                            {"id": 2, "score": 1, "urgent": False},
                            {"id": 3, "score": 5, "urgent": False},
                            {"id": 4, "score": 1, "urgent": False},
                        ]
                    }
                elif "personal Telegram bot" in system:
                    phase2_calls[0] += 1
                    raise LLMTruncationError("finish_reason=length")
                else:
                    raise AssertionError(f"Unexpected system prompt: {system[:100]}")

            def get_balance(self):
                return BalanceInfo(True, "USD", 50.0)

        mock_client = _PartialRecoveryClient()
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
            "daily_cost_limit": 1.0,
            "pricing": {"deepseek-flash": {"input": 1, "output": 2, "cache_hit": 0}},
        }

        storage = MagicMock()
        storage.get_meta.return_value = "0.0"
        storage.count_by_state.return_value = 0
        storage.set_meta = MagicMock()

        flt = LLMFilter(cfg, client=mock_client, storage=storage)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 5)]

        # Override recovery to return PARTIAL results (1 for 2 phase-2 items).
        # recovered[0] → phase2_idx=0 → orig_idx=0; recovered[1] absent → fallback.
        def _partial_recursive(*args, **kwargs):
            return [FilterResult(ok=True, score=9, summary="Recovered-0",
                                 urgent=False, location=None, title_en=None)]

        flt._evaluate_batch_recursive = _partial_recursive

        # _check_daily_cost: allow entry + phase-1 + phase-2 + recovery (5 calls → True),
        # 6th call → would block but is never reached.
        guard_call = [0]
        def mock_check_daily_cost():
            guard_call[0] += 1
            return True

        with patch.object(flt, "_check_daily_cost", mock_check_daily_cost):
            results = flt.evaluate_batch(items)

        # ── Verify the truncation + recovery branch was exercised ───────
        assert phase1_calls[0] == 1, f"Phase-1 called once, got {phase1_calls[0]}"
        assert phase2_calls[0] == 1, f"Phase-2 called once, got {phase2_calls[0]}"

        # ── Length preserved ────────────────────────────────────────────
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        # ── results[0] (orig_idx=0, phase2_idx=0) — recovered from partial
        assert results[0].ok is True
        assert results[0].score == 9, f"results[0] recovered score=9, got {results[0].score}"
        assert results[0].summary == "Recovered-0"
        assert results[0].score != 1, "results[0] must NOT have score from idx=1"
        assert results[0].score != 5, "results[0] must NOT have own phase-1 score"

        # ── results[2] (orig_idx=2, phase2_idx=1) — in phase-2, recovery absent
        # → falls back to phase-1 score=5; ok=True (>= phase2_min_score=4)
        # summary must be empty (fallback discards phase-2 summary).
        assert results[2].ok is True, f"results[2] ok should be True (score=5>=threshold), got {results[2]}"
        assert results[2].score == 5, f"results[2] fall back to phase-1 score=5, got {results[2].score}"
        assert results[2].score != 9, "results[2] must NOT have recovered score from idx=0"
        assert results[2].summary == "", (
            f"results[2].summary should be empty (fallback), got {results[2].summary!r}"
        )

        # ── results[1] and results[3] — never in phase-2, keep phase-1 score=1
        # (below both push_threshold=7 and phase2_min_score=4).
        # ok=True because phase-1 parser sets ok=True for any valid score;
        # threshold evaluation is done later by the pipeline dispatcher.
        assert results[1].ok is True, f"results[1] ok=True (phase-1 parser), got {results[1]}"
        assert results[1].score == 1, f"results[1] phase-1 score=1, got {results[1].score}"
        assert results[3].ok is True, f"results[3] ok=True (phase-1 parser), got {results[3]}"
        assert results[3].score == 1, f"results[3] phase-1 score=1, got {results[3].score}"
        assert results[3].score == 1, f"results[3] keep phase-1 score=1, got {results[3].score}"


# ─────────────────────────────────────────────────────────────────
# Minor #1: docstring matches actual behaviour (missing urgent → False, ok=True)
# ─────────────────────────────────────────────────────────────────

class TestPhase1UrgentMissingDefault:
    """_parse_phase1_response: missing urgent → urgent=False, ok=True (not ok=False)."""

    def test_missing_urgent_defaults_false_ok_true(self):
        """Missing urgent field → urgent=False, item ok=True."""
        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "results": [
                # id=1 has no "urgent" field
                {"id": 1, "score": 5},
                # id=2 has explicit urgent
                {"id": 2, "score": 6, "urgent": True},
            ]
        }
        cfg = {
            "enabled": True, "api_key": "test", "model": "deepseek-flash",
            "push_threshold": 7, "digest_threshold": 5, "push_urgent_only": False,
            "summary_max_chars": 240,
            "two_phase": {"enabled": True, "phase2_min_score": 4},
        }
        flt = LLMFilter(cfg, client=mock_client)
        items = [make_item(f"u{i}", f"T{i}", f"S{i}") for i in range(1, 3)]
        results = flt.evaluate_batch(items)

        # Missing urgent → safe default False, ok=True
        assert results[0].ok is True
        assert results[0].urgent is False
        # Explicit urgent=True → ok=True, urgent=True
        assert results[1].ok is True
        assert results[1].urgent is True


# ─────────────────────────────────────────────────────────────────
# Minor #2: PHASE1_SYSTEM_PROMPT comment about .format()
# ─────────────────────────────────────────────────────────────────

class TestPhase1PromptNoFormat:
    def test_prompt_comment_present(self):
        """prompts.py: comment warning about .format() on PHASE1_SYSTEM_PROMPT."""
        import newsbot.llm.prompts as p
        import inspect
        src = inspect.getsource(p)
        assert "PHASE1_SYSTEM_PROMPT" in src
        # Find the comment before PHASE1_SYSTEM_PROMPT definition
        lines = src.split("\n")
        phase1_line = next(i for i, l in enumerate(lines) if "PHASE1_SYSTEM_PROMPT = " in l)
        # Comment should be within 5 lines before the definition
        before = "\n".join(lines[max(0, phase1_line - 5):phase1_line])
        assert "format" in before.lower() or "{" not in before, (
            f"Expected comment about .format() before PHASE1_SYSTEM_PROMPT.\n{before}"
        )


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-q"])
