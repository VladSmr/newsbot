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
            "deepseek-chat": {"input": 1.0, "output": 2.0},
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