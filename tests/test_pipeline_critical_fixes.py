"""
Тесты Critical-1, Critical-2, Major-3, Major-4 для pipeline.py.

Без сети. Использует реальный API NewsPipeline: __new__ + ручная
инициализация атрибутов (паттерн из test_pipeline_llm.py).

Запуск: python -m pytest tests/test_pipeline_critical_fixes.py -q
"""
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.delivery.telegram import (
    BatchAbortedError,
    BotBlockedError,
    ChatIdNotBoundError,
)
from newsbot.llm.filter import FilterResult
from newsbot.pipeline import NewsPipeline, _normalize_cap_key
from newsbot.storage import Storage


# =============================================================================
# helpers
# =============================================================================

def _cleanup_db(path: str) -> None:
    for attempt in range(3):
        try:
            Path(path).unlink(missing_ok=True)
            Path(path + "-wal").unlink(missing_ok=True)
            Path(path + "-shm").unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.1)


def fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    Path(path).unlink()
    return path, lambda: _cleanup_db(path)


def make_item(url="https://example.com/1", title="Тест", summary="Краткое"):
    return NewsItem(
        source="Test",
        url=url,
        title=title,
        summary=summary,
        published="2024-01-01",
        published_parsed=None,
    )


class _Raise:
    """Резолвит исключение при каждом вызове send_batch."""

    def __init__(self, exc):
        self.exc = exc

    def send_batch(self, items):
        raise self.exc

    def send(self, item):
        raise self.exc

    def send_text(self, text):
        return False

    def close(self):
        pass


class _MockDelivery:
    """MagicMock-подобный delivery с close()."""

    def __getattr__(self, name):
        return MagicMock()

    def close(self):
        pass


# =============================================================================
# Critical-1: BatchAbortedError → successful_urls/permanent_failed_urls читаются
# =============================================================================

class TestCritical1BatchAborted:
    """
    Critical-1-фикс: при BatchAbortedError (ChatIdNotBoundError/BotBlockedError
    с partial success) pipeline читает successful_urls и permanent_failed_urls
    из исключения и помечает items СООТВЕТСТВУЮЩЕ.
    """

    def _run(self, exc: Exception):
        path, cleanup = fresh_db()
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = {"sources": [], "database": {"path": path}, "llm": {}}
        p.sources = []
        p._shutdown_event = None
        p.storage = Storage(path)
        p.delivery = _Raise(exc)
        p.llm_filter = None
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        items = [
            make_item(url="https://ex.com/1"),
            make_item(url="https://ex.com/2"),
            make_item(url="https://ex.com/3"),
        ]
        for item in items:
            p.storage.save(item)

        # _process_batch(items) — items как positional (budget_override=None)
        result = p._process_batch(items)

        conn = sqlite3.connect(path)
        states = dict(conn.execute("SELECT url, state FROM items").fetchall())
        conn.close()

        p.storage.close()
        cleanup()
        return result, states

    def test_batch_aborted_marks_successful_delivered(self):
        """
        BatchAbortedError с successful_urls=["u1"] → u1 получает state='delivered'.
        """
        exc = BatchAbortedError(
            successful_urls=["https://ex.com/1"],
            permanent_failed_urls=[],
        )
        result, states = self._run(exc)
        assert result == [], "batch aborted → пустой failed_urls"
        assert states["https://ex.com/1"] == "delivered", (
            f"successful_url должен быть 'delivered', got {states['https://ex.com/1']}"
        )
        assert states["https://ex.com/2"] not in ("delivered", "failed"), (
            f"Недоставленные остаются pending, got {states['https://ex.com/2']}"
        )

    def test_batch_aborted_marks_permanent_failed_failed(self):
        """
        BatchAbortedError с permanent_failed_urls=["u2"] → u2 получает state='failed'.
        """
        exc = BatchAbortedError(
            successful_urls=["https://ex.com/1"],
            permanent_failed_urls=["https://ex.com/2"],
        )
        result, states = self._run(exc)
        assert result == []
        assert states["https://ex.com/1"] == "delivered"
        assert states["https://ex.com/2"] == "failed", (
            f"permanent_failed_url должен быть 'failed', got {states['https://ex.com/2']}"
        )

    def test_batch_aborted_successful_not_in_failed_urls(self):
        """
        Успешно доставленные из BatchAbortedError НЕ попадают в failed_urls
        (не уходят в повторную отправку).
        """
        exc = BatchAbortedError(
            successful_urls=["https://ex.com/1"],
            permanent_failed_urls=["https://ex.com/2"],
        )
        result, _ = self._run(exc)
        assert "https://ex.com/1" not in result, (
            "Успешно доставленный url не должен быть в failed_urls"
        )
        assert "https://ex.com/2" not in result, (
            "Permanent-failed url не должен быть в failed_urls (terminal)"
        )


# =============================================================================
# Critical-2: ingest-guard — «все помечены» только при >1 кандидата
# =============================================================================

class TestCritical2IngestGuard:
    """
    Critical-2-фикс: guard len(dup_0based) >= len(candidates) срабатывает
    только когда кандидатов > 1. При ровно 1 кандидате «все помечены» —
    валидный cross-cycle dedup.
    """

    def _run_llm_dedup(self, candidates, mock_filter):
        """
        Запускает _run_llm_dedup с given candidates и mock filter.
        _run_llm_dedup(candidates) сам берёт recent из storage (get_recent_titles).
        """
        path, cleanup = fresh_db()
        cfg = {
            "sources": [],
            "database": {"path": path},
            "llm": {
                "llm_dedup": True,
                "enabled": True,
                "api_key": "key",
                "push_threshold": 7,
                "digest_threshold": 5,
                "push_urgent_only": False,
            },
            "dedup": {"window_days": 3, "llm_recent_limit": 20},
            "telegram": {"token": "test", "chat_id": "test"},
        }
        mock_del = _MockDelivery()
        p_del_patch = patch.object(NewsPipeline, "_build_delivery", return_value=mock_del)
        p_flt_patch = patch.object(NewsPipeline, "_build_llm_filter", return_value=mock_filter)

        p_del_patch.start()
        p_flt_patch.start()
        try:
            p = NewsPipeline(cfg)
            p.llm_filter = mock_filter
            p._llm_calls_left = 10
            p._feedback_context = None
            p._llm_dedup_skipped_urls = set()

            for item in candidates:
                p.storage.save(item)

            result = p._run_llm_dedup(candidates)

            count_new = p.storage.count_by_state("new")
            count_skipped = p.storage.count_by_state("skipped")
        finally:
            p_flt_patch.stop()
            p_del_patch.stop()

        p.close()
        cleanup()
        return result, p, count_new, count_skipped

    def test_single_candidate_all_marked_applies_skipped(self):
        """
        (а) 1 кандидат + все помечен → skipped ПРИМЕНЯЕТСЯ (валидный dedup).
        guard НЕ срабатывает (len>1) → set_state("skipped") применяется.
        """
        item = make_item(url="https://single.com/u1", title="Single")

        class FakeDedupFilter:
            _DEDUP_CHUNK_SIZE = 40
            push_threshold = 7
            digest_threshold = 5
            push_urgent_only = False

            def detect_duplicates(self, items, recent):
                return {0}  # 0-based: единственный кандидат помечен

        mock_filter = FakeDedupFilter()
        skipped, p, count_new, count_skipped = self._run_llm_dedup([item], mock_filter)

        assert count_skipped == 1, (
            f"При 1 кандидате 'skipped' должен применяться, "
            f"new={count_new}, skipped={count_skipped}"
        )

    def test_three_candidates_all_marked_ignored_nonempty_recent(self):
        """
        (б) 3 кандидата + непустой recent + все помечены → результат ИГНОРИРУЕТСЯ,
        items остаются 'new', set_state('skipped') НЕ вызывался.
        """
        items = [
            make_item(url=f"https://m.com/{i}", title=f"Item {i}")
            for i in range(1, 4)
        ]

        class FakeDedupFilter:
            _DEDUP_CHUNK_SIZE = 40
            push_threshold = 7
            digest_threshold = 5
            push_urgent_only = False

            def detect_duplicates(self, items, recent):
                return {0, 1, 2}  # все 3 помечены (0-based)

        mock_filter = FakeDedupFilter()
        skipped, p, count_new, count_skipped = self._run_llm_dedup(items, mock_filter)

        assert skipped == 0
        assert count_new == 3, (
            "Все 3 items должны остаться state='new', не 'skipped'"
        )
        assert count_skipped == 0

    def test_three_candidates_all_marked_ignored_empty_recent(self):
        """
        (в) 3 кандидата + пустой recent + все помечены → тоже ИГНОР.
        """
        items = [
            make_item(url=f"https://n.com/{i}", title=f"Item {i}")
            for i in range(1, 4)
        ]

        class FakeDedupFilter:
            _DEDUP_CHUNK_SIZE = 40
            push_threshold = 7
            digest_threshold = 5
            push_urgent_only = False

            def detect_duplicates(self, items, recent):
                return {0, 1, 2}  # все 3 помечены (0-based)

        mock_filter = FakeDedupFilter()
        skipped, p, count_new, count_skipped = self._run_llm_dedup(items, mock_filter)

        assert skipped == 0
        assert count_new == 3
        assert count_skipped == 0


# =============================================================================
# Major-3: _normalize_cap_key — защита от запятой (case-insensitive state codes)
# =============================================================================

class TestMajor3CapLocation:
    """
    Major-3-фикс: _normalize_cap_key проверяет РЕГИСТР на исходной строке
    (ДО lower()). Без фикса "Portland, OR" и "Portland, ME" схлопывались
    в "portland".
    """

    def test_portland_or_different_from_portland_me(self):
        """Portland, OR ≠ Portland, ME — разные коды штатов."""
        assert _normalize_cap_key("Portland, OR") != _normalize_cap_key("Portland, ME")

    def test_portland_or_vs_oregon(self):
        """Portland, OR vs Oregon → разные."""
        assert _normalize_cap_key("Portland, OR") != _normalize_cap_key("Oregon")

    def test_voronezh_region_russia_vs_voronezh_russia(self):
        """Voronezh region, Russia == Voronezh, Russia."""
        key1 = _normalize_cap_key("Voronezh region, Russia")
        key2 = _normalize_cap_key("Voronezh, Russia")
        assert key1 == key2, f"{key1!r} != {key2!r}"

    def test_russian_moscow(self):
        """Москва, Россия → москва."""
        assert _normalize_cap_key("Москва, Россия") == "москва"

    def test_russian_voronezh_oblast(self):
        """Воронежская область → воронежская (generic suffix stripped)."""
        key = _normalize_cap_key("Воронежская область")
        assert "воронежская" in key

    def test_voronezh_region_russia_vs_russia(self):
        """Voronezh, Russia ≠ Russia."""
        assert _normalize_cap_key("Voronezh, Russia") != _normalize_cap_key("Russia")

    def test_country_code_us(self):
        """US — код страны, не обрезается."""
        assert _normalize_cap_key("Washington, US") == "washington, us"

    def test_multiple_words_country(self):
        """Несколько заглавных слов — цельное название, не обрезается."""
        key = _normalize_cap_key("New York, United States")
        assert "new york" in key


# =============================================================================
# Major-4: fatal-ошибки LLM (401/402/403) — item остаётся pending
# =============================================================================

class TestMajor4FatalError:
    """
    Major-4-фикс: FilterResult.fatal=True (LLMFatalError 401/402/403)
    обрабатывается так же, как cost_blocked — item остаётся pending,
    НЕ уходит в fallback/deferred/push.
    """

    def _run(self, res: FilterResult) -> tuple:
        path, cleanup = fresh_db()
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = {
            "sources": [],
            "database": {"path": path},
            "llm": {
                "push_threshold": 7,
                "digest_threshold": 5,
                "push_urgent_only": False,
            },
        }
        p.sources = []
        p.storage = Storage(path)
        p.delivery = _MockDelivery()
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        item = make_item(url="https://fatal.com/1", title="Fatal", summary="S")
        item.cluster_key = "test|"
        p.storage.save(item)

        skipped: set = set()
        deferred: set = set()
        to_deliver: list = []
        records: list = []

        p._apply_result(
            item, res,
            push_th=7, digest_th=5, push_urgent_only=False,
            skipped_urls=skipped, deferred_urls=deferred,
            to_deliver=to_deliver, llm_batch_records=records,
        )

        p.storage.close()
        cleanup()
        return to_deliver, skipped, deferred

    def test_fatal_true_remains_pending_not_delivered(self):
        """fatal=True → item НЕ попадает в to_deliver."""
        res = FilterResult(ok=False, fatal=True)
        to_deliver, skipped, deferred = self._run(res)

        assert to_deliver == [], "fatal=True: item не должен идти в to_deliver"
        assert skipped == set()
        assert deferred == set()

    def test_fatal_true_remains_pending_not_deferred(self):
        """fatal=True → item НЕ попадает в deferred."""
        res = FilterResult(ok=False, fatal=True)
        _, skipped, deferred = self._run(res)

        assert deferred == set(), "fatal=True: item не должен идти в deferred"
        assert skipped == set()

    def test_fatal_without_attribute_still_works(self):
        """
        FilterResult БЕЗ поля fatal (legacy) — getattr возвращает False,
        ok=False идёт по обычному fallback-маршруту (НЕ падает).
        """
        res = FilterResult(ok=False)
        assert getattr(res, "fatal", False) is False
        to_deliver, skipped, deferred = self._run(res)
        # fallback при ok=False (без fatal) → to_deliver
        assert len(to_deliver) == 1, "legacy FilterResult (без fatal) → fallback → to_deliver"


# =============================================================================
# Minor: _coerce_int защищает от отрицательного конфига
# =============================================================================

class TestMinorCoerceInt:
    """
    Minor-фикс: max(0, _coerce_int(...)) для llm.max_per_cycle,
    и _coerce_int для max_retries / stuck_pending_limit.
    """

    def _make_pipeline(self, db_path, max_per_cycle, max_retries, stuck_limit):
        """Создаёт NewsPipeline с заданными значениями через реальный __init__."""
        config = {
            "sources": [],
            "database": {"path": db_path, "max_retries": max_retries,
                          "stuck_pending_limit": stuck_limit},
            "llm": {
                "enabled": True, "api_key": "key",
                "max_per_cycle": max_per_cycle,
                "push_threshold": 7, "digest_threshold": 5,
                "push_urgent_only": False,
            },
        }
        p = NewsPipeline(config)
        return p

    def test_negative_max_per_cycle_clamped_to_zero(self):
        """llm.max_per_cycle: -5 → _max_per_cycle = 0 (LLM отключён)."""
        path, cleanup = fresh_db()
        try:
            p = self._make_pipeline(path, max_per_cycle=-5, max_retries=3, stuck_limit=500)
            assert p._max_per_cycle == 0, (
                f"Отрицательный max_per_cycle должен быть 0, got {p._max_per_cycle}"
            )
            p.close()
        finally:
            cleanup()

    def test_string_max_per_cycle_parsed(self):
        """llm.max_per_cycle: "10" (YAML string) → int 10."""
        path, cleanup = fresh_db()
        try:
            p = self._make_pipeline(path, max_per_cycle="10", max_retries=3, stuck_limit=500)
            assert p._max_per_cycle == 10
            p.close()
        finally:
            cleanup()

    def test_coerce_int_max_retries_string(self):
        """database.max_retries через _coerce_int (не raw int())."""
        path, cleanup = fresh_db()
        try:
            p = self._make_pipeline(path, max_per_cycle=150, max_retries="5", stuck_limit="200")
            assert p._max_retries == 5
            assert p._stuck_pending_limit == 200
            p.close()
        finally:
            cleanup()

    def test_coerce_int_stuck_pending_limit_float(self):
        """database.stuck_pending_limit через _coerce_int (float string → int)."""
        path, cleanup = fresh_db()
        try:
            p = self._make_pipeline(path, max_per_cycle=150, max_retries=3, stuck_limit="300.0")
            assert p._stuck_pending_limit == 300
            p.close()
        finally:
            cleanup()