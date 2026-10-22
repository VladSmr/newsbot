"""
P1-регресс: постоянные ошибки доставки (chat_id не привязан / бот заблокирован)
НЕ расходуют retry_count и НЕ возвращаются как failed_urls.

Раньше ChatIdNotBoundError/BotBlockedError (подклассы TelegramDeliveryError)
перехватывались общим `except Exception` в _process_batch → url'ы попадали в
failed_urls → increment_retry_count каждый цикл → max_retries →
mark_failed_if_retries_exhausted → 'failed' → purge. Сценарий «юзер заблокировал
бота на 2 часа» стирал все новости.

Без сети.
"""
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.delivery.telegram import (
    BotBlockedError,
    ChatIdNotBoundError,
    TelegramDeliveryError,
)
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


class RaisingDelivery:
    """Фейковая доставка, всегда бросающая заданное исключение."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def send_batch(self, items):
        raise self.exc

    def send_text(self, text):
        return False

    def close(self):
        pass


class TestPermanentDeliveryErrors:
    def _run(self, exc: Exception) -> tuple:
        path, cleanup = fresh_db()
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = {"sources": [], "database": {"path": path}, "llm": {}}
        p.sources = []
        p._shutdown_event = None
        p.storage = Storage(path)
        p.delivery = RaisingDelivery(exc)
        p.llm_filter = None
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        items = [make_item(url="https://ex.com/1"), make_item(url="https://ex.com/2")]
        for item in items:
            p.storage.save(item)

        result = p._process_batch(items)

        pending = {i.url for i in p.storage.get_all_by_state("pending")}
        conn = sqlite3.connect(path)
        retries = dict(conn.execute("SELECT url, retry_count FROM items").fetchall())
        conn.close()
        p.storage.close()
        cleanup()
        return result, pending, retries

    def test_chat_id_not_bound_does_not_consume_retries(self):
        result, pending, retries = self._run(ChatIdNotBoundError("chat_id не задан"))
        assert result == [], "P1: не должно быть failed_urls"
        assert pending == {"https://ex.com/1", "https://ex.com/2"}
        assert all(v == 0 for v in retries.values()), (
            f"P1: retry_count не должен инкрементироваться, got {retries}"
        )

    def test_bot_blocked_does_not_consume_retries(self):
        result, pending, retries = self._run(BotBlockedError("403 blocked"))
        assert result == [], "P1: бот заблокирован — не failed_urls"
        assert pending == {"https://ex.com/1", "https://ex.com/2"}
        assert all(v == 0 for v in retries.values()), (
            f"P1: retry_count не должен инкрементироваться, got {retries}"
        )

    def test_transient_error_still_goes_to_retry(self):
        """Транзиентная ошибка (таймаут/5xx/429) по-прежнему → failed_urls."""
        result, pending, retries = self._run(
            TelegramDeliveryError("HTTP-ошибка TimeoutException после 3 попыток")
        )
        assert sorted(result) == ["https://ex.com/1", "https://ex.com/2"], (
            "транзиентная ошибка должна возвращаться как failed_urls (retry)"
        )
        assert pending == {"https://ex.com/1", "https://ex.com/2"}
