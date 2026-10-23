"""
Тесты:
  1) проброс shutdown_event из NewsPipeline в TelegramDelivery;
  2) cost-blocked items (дневной лимит LLM) остаются pending,
     а не уходят в deferred/fallback/failed.

Без сети.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.delivery import TelegramDelivery
from newsbot.llm.filter import FilterResult
from newsbot.pipeline import NewsPipeline
from newsbot.storage import Storage


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


def _base_config(db_path: str, token="", chat_id="") -> dict:
    return {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": token, "chat_id": chat_id},
        "llm": {
            "enabled": True,
            "provider": "deepseek",
            "model": "deepseek-flash",
            "api_key": "",  # нет ключа → llm_filter=None (сеть не нужна)
            "base_url": "https://api.deepseek.com",
            "push_threshold": 7,
            "digest_threshold": 5,
            "push_urgent_only": True,
        },
    }


class CostBlockedFilter:
    """Фейковый фильтр: воспроизводит сигнал cost_blocked от LLMFilter."""

    def __init__(self, push_urgent_only=True):
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = push_urgent_only

    def evaluate_batch(self, items, feedback_context=None):
        return [FilterResult(ok=False, cost_blocked=True) for _ in items]


class FailFilter:
    """Фейковый фильтр: обычный отказ LLM (не cost-block)."""

    def __init__(self, push_urgent_only=True):
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = push_urgent_only

    def evaluate_batch(self, items, feedback_context=None):
        return [FilterResult(ok=False) for _ in items]


class MockDelivery:
    def __init__(self):
        self.sent: List[str] = []

    def send_batch(self, items):
        self.sent.extend(i.url for i in items)
        return ([i.url for i in items], [])

    def send_text(self, text):
        return True

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────
# Задача 1: проброс shutdown_event
# ─────────────────────────────────────────────────────────────────


class TestShutdownEventPropagation:
    def test_pipeline_passes_event_to_telegram_delivery(self):
        path, cleanup = fresh_db()
        try:
            ev = threading.Event()
            p = NewsPipeline(
                _base_config(path, token="tok", chat_id="123"),
                shutdown_event=ev,
            )
            try:
                assert isinstance(p.delivery, TelegramDelivery)
                assert p.delivery._shutdown_event is ev
            finally:
                p.close()
        finally:
            cleanup()

    def test_pipeline_without_event_keeps_backward_compat(self):
        path, cleanup = fresh_db()
        try:
            p = NewsPipeline(_base_config(path, token="tok", chat_id="123"))
            try:
                assert isinstance(p.delivery, TelegramDelivery)
                assert p.delivery._shutdown_event is None
            finally:
                p.close()
        finally:
            cleanup()

    def test_set_event_interrupts_delivery_wait(self):
        ev = threading.Event()
        ev.set()
        td = TelegramDelivery(token="tok", chat_id="123", shutdown_event=ev)
        started = time.monotonic()
        td._wait(10)
        assert time.monotonic() - started < 1.0


# ─────────────────────────────────────────────────────────────────
# Задача 2: cost-blocked items остаются pending
# ─────────────────────────────────────────────────────────────────


class TestCostBlockedItemsStayPending:
    def _run(self, items, fake_filter):
        path, cleanup = fresh_db()
        cfg = _base_config(path)
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = cfg
        p.sources = []
        p._shutdown_event = None
        p.storage = Storage(path)
        p.delivery = MockDelivery()
        p.llm_filter = fake_filter
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        for item in items:
            p.storage.save(item)

        p._process_batch(items)

        pending = p.storage.get_all_by_state("pending")
        deferred = p.storage.get_all_by_state("deferred")
        failed = p.storage.get_all_by_state("failed")
        sent = list(p.delivery.sent)
        p.storage.close()
        cleanup()
        return pending, deferred, failed, sent

    def test_cost_blocked_items_remain_pending(self):
        items = [
            make_item(url="https://cost.com/1"),
            make_item(url="https://cost.com/2"),
        ]
        pending, deferred, failed, sent = self._run(items, CostBlockedFilter())

        assert sorted(i.url for i in pending) == [
            "https://cost.com/1", "https://cost.com/2",
        ]
        assert deferred == []
        assert failed == []
        assert sent == []

    def test_plain_llm_failure_still_goes_to_deferred(self):
        """Обычный ok=False (не cost-block) при push_urgent_only → deferred."""
        items = [make_item(url="https://fail.com/1")]
        pending, deferred, failed, sent = self._run(
            items, FailFilter(push_urgent_only=True)
        )

        assert pending == []
        assert [i.url for i in deferred] == ["https://fail.com/1"]
        assert failed == []
        assert sent == []
