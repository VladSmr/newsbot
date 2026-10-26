"""
M11-fix: тесты cost-guard — fail-open к ошибкам SQLite и NaN-блокировка.

(a) get_meta кидает OperationalError → cost-check fail-open, evaluate проходит.
(b) set_meta кидает → накопление не падает.
(c) meta="nan" → не блокирует вызовы (isfinite отсекает).
"""
import math
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from newsbot.llm.filter import LLMFilter


class MockStorage:
    """Storage-заглушка с настраиваемым поведением get_meta/set_meta."""

    def __init__(self, get_meta_side_effect=None, set_meta_side_effect=None):
        self._get_meta_side_effect = get_meta_side_effect
        self._set_meta_side_effect = set_meta_side_effect
        self._store = {}

    def get_meta(self, key):
        if callable(self._get_meta_side_effect):
            return self._get_meta_side_effect(key)
        if isinstance(self._get_meta_side_effect, dict):
            return self._get_meta_side_effect.get(key)
        if isinstance(self._get_meta_side_effect, BaseException):
            raise self._get_meta_side_effect
        return self._store.get(key)

    def set_meta(self, key, value):
        if isinstance(self._set_meta_side_effect, BaseException):
            raise self._set_meta_side_effect
        self._store[key] = value

    def count_by_state(self, state):
        return 0


def _make_filter(storage=None, daily_cost_limit=1.0):
    cfg = {
        "enabled": True,
        "model": "deepseek-chat",
        "daily_cost_limit": daily_cost_limit,
        "push_urgent_only": False,
        "pricing": {
            "deepseek-chat": {"input": 1.0, "output": 2.0, "cache_hit": 0.01},
        },
    }
    return LLMFilter(config_llm=cfg, storage=storage, client=MagicMock())


# ─── helpers ────────────────────────────────────────────────────────────────

def _today_key(prefix):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}_{today}"


# ─── (a) get_meta OperationalError → fail-open ────────────────────────────────

class TestCostGuardFailOpenOnGetMeta:
    """(a) get_meta кидает OperationalError → cost-check fail-open, evaluate проходит."""

    def test_check_daily_cost_fail_open_on_get_meta_error(self):
        """OperationalError при get_meta cost → fail-open → returns True."""
        storage = MockStorage()
        storage._get_meta_side_effect = sqlite3.OperationalError("database is locked")
        filt = _make_filter(storage=storage, daily_cost_limit=0.01)
        # fail-open: должен вернуть True (LLM вызов разрешён)
        result = filt._check_daily_cost()
        assert result is True, "При ошибке чтения get_meta должен быть fail-open"

    def test_check_daily_cost_fail_open_on_warning_flag_get_error(self):
        """OperationalError при get_meta warning-flag → fail-open на чтение флага."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"
        flag_key = f"llm_cost_limit_warning_done_{today}"

        def broken_get(key):
            if key == flag_key:
                raise sqlite3.OperationalError("database is locked")
            return "999.0"

        storage = MockStorage()
        storage._get_meta_side_effect = broken_get

        filt = _make_filter(storage=storage, daily_cost_limit=1.0)
        # cost=999.0 >= limit=1.0 → лимит превышен, но ошибка чтения флага
        # не должна сломать цикл
        result = filt._check_daily_cost()
        assert result is False  # лимит превышен


# ─── (b) set_meta OperationalError → no-op ──────────────────────────────────

class TestCostGuardFailOpenOnSetMeta:
    """(b) set_meta кидает → накопление не падает."""

    def test_accumulate_daily_cost_fail_open_on_set_error(self):
        """OperationalError при set_meta → no-op, не падает."""
        storage = MockStorage()
        storage._set_meta_side_effect = sqlite3.OperationalError("database is locked")
        filt = _make_filter(storage=storage, daily_cost_limit=1.0)

        # Не должно выбросить исключение
        filt._accumulate_daily_cost(prompt_tokens=1000, completion_tokens=500)
        # Просто проходит без exception

    def test_check_daily_cost_fail_open_on_set_warning_flag_error(self):
        """OperationalError при set_meta warning-flag → WARNING заблокирован, но не падает."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        storage = MockStorage()
        storage._get_meta_side_effect = {cost_key: "999.0"}
        storage._set_meta_side_effect = sqlite3.OperationalError("database is locked")

        filt = _make_filter(storage=storage, daily_cost_limit=1.0)
        # Не должно выбросить исключение, лимит превышен
        result = filt._check_daily_cost()
        assert result is False  # лимит превышен


# ─── (c) meta="nan" → не блокирует вызовы ──────────────────────────────────

class TestCostGuardNaNBlockade:
    """(c) meta='nan' → не блокирует вызовы (isfinite отсекает)."""

    def test_check_daily_cost_nan_meta_not_blocking(self):
        """meta='nan' → float('nan') not finite → treated as 0.0 → not blocking."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        storage = MockStorage()
        storage._get_meta_side_effect = {cost_key: "nan"}
        filt = _make_filter(storage=storage, daily_cost_limit=1.0)
        # NaN не должно заблокировать: isfinite → 0.0 → 0.0 < 1.0 → True
        result = filt._check_daily_cost()
        assert result is True, "NaN в meta не должно блокировать вызовы"

    def test_accumulate_daily_cost_nan_current_not_blocking(self):
        """Текущий cost='nan' → current=0.0, запись выполняется корректно."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        storage = MockStorage()
        storage._get_meta_side_effect = {cost_key: "nan"}
        filt = _make_filter(storage=storage, daily_cost_limit=1.0)

        # Не должно выбросить исключение, запись 0.0 + cost
        filt._accumulate_daily_cost(prompt_tokens=1000, completion_tokens=500)

        # storage.set_meta был вызван с корректным числом
        stored = storage._store.get(cost_key)
        assert stored is not None, "set_meta должен быть вызван"
        # cost = (1000*1.0 + 500*2.0) / 1_000_000 = 0.002
        assert math.isfinite(float(stored)), (
            f"Записанное значение должно быть конечным: {stored}"
        )

    def test_check_daily_cost_float_nan_meta_not_blocking(self):
        """meta=flt('nan') в storage → тест на случай NaN без строки 'nan'."""
        # float(float.nan) в storage — изолированный случай
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        # Симулируем: storage вернул строку, которая при float даёт inf
        storage = MockStorage()
        storage._get_meta_side_effect = {cost_key: "inf"}
        filt = _make_filter(storage=storage, daily_cost_limit=1.0)
        result = filt._check_daily_cost()
        assert result is True, "inf в meta не должно блокировать вызовы"


# ─── (d) cache-hit cost split ───────────────────────────────────────────────

class TestCostGuardCacheHitSplit:
    """(d) Стоимость считается по двум ценам: hit и miss."""

    def test_cache_hit_miss_split_cost(self):
        """hit_tokens*price_hit + miss_tokens*price_input + completion*price_output."""
        storage = MockStorage()
        filt = _make_filter(
            storage=storage,
            daily_cost_limit=10.0,
        )

        # Цены из конфига: input=1.0, output=2.0, cache_hit=0.01
        # Hit:  900 tokens × 0.01 = 0.009
        # Miss: 100 tokens × 1.00 = 0.1
        # Comp: 200 tokens × 2.00 = 0.4
        # Total: 0.009 + 0.1 + 0.4 = 0.509
        filt._accumulate_daily_cost(
            prompt_tokens=1000,
            completion_tokens=200,
            cache_hit_tokens=900,
            cache_miss_tokens=100,
        )

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"
        stored = storage._store.get(cost_key)
        assert stored is not None, "set_meta должен быть вызван"
        assert math.isfinite(float(stored)), f"Записанное значение должно быть конечным: {stored}"

        # cost = (900*0.01 + 100*1.0 + 200*2.0) / 1_000_000 = 0.509 / 1_000_000 = 0.000000509
        # Но в _accumulate_daily_cost делим на 1_000_000: 0.509 / 1_000_000 = 0.000000509
        # Хм, на самом деле деление происходит в _accumulate_daily_cost:
        # cost = (cache_hit_tokens * cache_hit_price + ...) / 1_000_000
        # = (900*0.01 + 100*1.0 + 200*2.0) / 1_000_000
        # = (9.0 + 100.0 + 400.0) / 1_000_000
        # = 509.0 / 1_000_000
        # = 0.000509
        expected = (900 * 0.01 + 100 * 1.0 + 200 * 2.0) / 1_000_000
        assert abs(float(stored) - expected) < 1e-9, f"expected {expected}, got {stored}"

    def test_cache_hit_zero_uses_input_price(self):
        """Все токены — miss: fallback на цену input (старое поведение)."""
        storage = MockStorage()
        filt = _make_filter(storage=storage, daily_cost_limit=10.0)

        # Нет split'а (hit=0, miss=0) → fallback: все prompt по цене input
        filt._accumulate_daily_cost(
            prompt_tokens=1000,
            completion_tokens=200,
            cache_hit_tokens=0,
            cache_miss_tokens=0,
        )

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"
        stored = storage._store.get(cost_key)
        expected = (1000 * 1.0 + 200 * 2.0) / 1_000_000  # = 0.0024
        assert abs(float(stored) - expected) < 1e-9, f"expected {expected}, got {stored}"

    def test_cache_hit_partial_split_correct(self):
        """50% hit, 50% miss: стоимость должна быть значительно ниже old formula."""
        storage = MockStorage()
        filt = _make_filter(storage=storage, daily_cost_limit=10.0)

        # 1000 prompt tokens: 500 hit, 500 miss; 100 completion
        filt._accumulate_daily_cost(
            prompt_tokens=1000,
            completion_tokens=100,
            cache_hit_tokens=500,
            cache_miss_tokens=500,
        )

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"
        stored = float(storage._store.get(cost_key))

        # Новая формула: 500*0.01 + 500*1.0 + 100*2.0 = 5 + 500 + 200 = 705 / 1e6 = 0.000705
        # Старая формула: 1000*1.0 + 100*2.0 = 1200 / 1e6 = 0.0012
        # Новая должна быть примерно вдвое меньше
        assert stored < 0.001, f"cache-hit должен был снизить стоимость: got {stored}"
        assert stored > 0.0001, f"стоимость слишком мала: {stored}"


# ─── (e) max_retries=0 / -1 — ровно 1 попытка ───────────────────────────────

class TestMaxRetriesZeroOne:
    """(e) max_retries=0 или -1: ровно 1 попытка."""

    def test_zero_retries_one_call_with_mock(self, caplog):
        """max_retries=0 → ровно один вызов _post."""
        import logging
        import newsbot.llm.client as client_module

        call_count = [0]

        class _MockOnceHTTPClient:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                call_count[0] += 1
                # Первый вызов — Timeout, чтобы проверить что retry НЕ делается
                import httpx
                raise httpx.TimeoutException("timeout")

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        prev = client_module.httpx.Client
        client_module.httpx.Client = _MockOnceHTTPClient
        try:
            with caplog.at_level(logging.WARNING, logger="newsbot.llm.client"):
                from newsbot.llm.client import DeepSeekClient, LLMError
                client = DeepSeekClient(
                    api_key="test",
                    base_url="https://api.deepseek.com",
                    model="test",
                    max_retries=0,
                )
                try:
                    client.chat_json("sys", "user")
                except LLMError:
                    pass
            # Ровно 1 вызов (без ретрая)
            assert call_count[0] == 1, f"Expected exactly 1 call, got {call_count[0]}"
        finally:
            client_module.httpx.Client = prev

    def test_negative_retries_one_call(self, caplog):
        """max_retries=-1 → ровно один вызов."""
        import logging
        import newsbot.llm.client as client_module

        call_count = [0]

        class _MockOnceHTTPClient2:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                call_count[0] += 1
                import httpx
                raise httpx.TimeoutException("timeout")

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        prev = client_module.httpx.Client
        client_module.httpx.Client = _MockOnceHTTPClient2
        try:
            with caplog.at_level(logging.WARNING, logger="newsbot.llm.client"):
                from newsbot.llm.client import DeepSeekClient, LLMError
                client = DeepSeekClient(
                    api_key="test",
                    base_url="https://api.deepseek.com",
                    model="test",
                    max_retries=-1,
                )
                try:
                    client.chat_json("sys", "user")
                except LLMError:
                    pass
            assert call_count[0] == 1, f"Expected exactly 1 call, got {call_count[0]}"
        finally:
            client_module.httpx.Client = prev

    def test_zero_retries_warns_once(self, caplog):
        """max_retries=0 → ровно один warning при создании клиента."""
        import logging

        from newsbot.llm.client import DeepSeekClient

        with caplog.at_level(logging.WARNING, logger="newsbot.llm.client"):
            DeepSeekClient(
                api_key="test",
                base_url="https://api.deepseek.com",
                model="test",
                max_retries=0,
            )

        messages = [r.getMessage() for r in caplog.records]
        warning_msgs = [m for m in messages if "трактуется как 1" in m]
        assert len(warning_msgs) == 1, (
            f"Expected exactly 1 warning about retries, got {len(warning_msgs)}: {warning_msgs}"
        )