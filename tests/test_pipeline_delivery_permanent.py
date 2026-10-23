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
    PermanentDeliveryError,
    TelegramDeliveryError,
    _strip_control_chars,
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
        # FIX-1: send_batch теперь возвращает Tuple[List[str], List[str]]
        raise self.exc

    def send(self, items):
        raise self.exc

    def send_text(self, text):
        return False

    def close(self):
        pass


class MockDelivery:
    """Фейковая доставка, имитирующая partial delivery с permanent-failed items."""

    def __init__(self, successful_urls, permanent_failed_urls=()):
        self.successful_urls = successful_urls
        self.permanent_failed_urls = permanent_failed_urls
        # FIX-1: возвращает Tuple[List[str], List[str]] (новый контракт send_batch)
        self._calls = []

    def send_batch(self, items):
        self._calls.append(list(items))
        return (self.successful_urls, self.permanent_failed_urls)

    def send_text(self, text):
        return True

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

    def test_permanent_failed_items_end_up_failed(self):
        """
        FIX-1: items с PermanentDeliveryError должны получить state='failed' в БД,
        НЕ 'delivered'. retry_count НЕ инкрементируется.
        """
        path, cleanup = fresh_db()
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = {"sources": [], "database": {"path": path}, "llm": {}}
        p.sources = []
        p._shutdown_event = None
        p.storage = Storage(path)
        # delivery возвращает partial success + permanent-failed
        p.delivery = MockDelivery(
            successful_urls=["https://ex.com/1"],
            permanent_failed_urls=["https://ex.com/2"],
        )
        p.llm_filter = None
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        items = [
            make_item(url="https://ex.com/1"),
            make_item(url="https://ex.com/2"),
        ]
        for item in items:
            p.storage.save(item)

        result = p._process_batch(items)

        # permanent-failed НЕ попадает в failed_urls (retry не нужен — уже terminal)
        assert "https://ex.com/2" not in result, (
            "FIX-1: permanent-failed item не должен возвращаться как failed_url"
        )
        # state в БД: success → delivered, perm-failed → failed
        conn = sqlite3.connect(path)
        states = dict(conn.execute("SELECT url, state FROM items").fetchall())
        retries = dict(conn.execute("SELECT url, retry_count FROM items").fetchall())
        conn.close()
        p.storage.close()
        cleanup()

        assert states["https://ex.com/1"] == "delivered", (
            f"Успешно доставленный item должен быть 'delivered', got {states['https://ex.com/1']}"
        )
        assert states["https://ex.com/2"] == "failed", (
            f"FIX-1: permanent-failed item должен быть 'failed', got {states['https://ex.com/2']}"
        )
        assert all(v == 0 for v in retries.values()), (
            f"FIX-1: retry_count не должен инкрементироваться при permanent-fail, got {retries}"
        )

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


# =============================================================================
# Major #4: PermanentDeliveryError — класс существует и наследует правильно
# =============================================================================

def test_permanent_delivery_error_class():
    """PermanentDeliveryError — подкласс TelegramDeliveryError."""
    assert issubclass(PermanentDeliveryError, TelegramDeliveryError)
    # 400/413/422 — не BotBlockedError и не ChatIdNotBoundError
    assert not issubclass(PermanentDeliveryError, BotBlockedError)
    assert not issubclass(PermanentDeliveryError, ChatIdNotBoundError)
    # Экземпляр создаётся и поднимается
    err = PermanentDeliveryError("HTTP 400: Bad request")
    assert isinstance(err, TelegramDeliveryError)
    assert "400" in str(err)


# =============================================================================
# Minor #5: _strip_control_chars удаляет control-chars из исходящего текста
# =============================================================================

def test_strip_control_chars():
    """Control-chars удаляются, \\n/\\t/\\r сохраняются."""
    # Basic control chars
    assert _strip_control_chars("hello\x00world") == "helloworld"
    assert _strip_control_chars("hello\x07world") == "helloworld"  # bell
    assert _strip_control_chars("hello\x1fworld") == "helloworld"  # unit separator
    assert _strip_control_chars("hello\x7fworld") == "helloworld"  # del
    assert _strip_control_chars("hello\x85world") == "helloworld"  # next line (NEL)
    # Unicode line separators
    assert _strip_control_chars("hello\u2028world") == "helloworld"
    assert _strip_control_chars("hello\u2029world") == "helloworld"
    # Preserve \n, \t, \r
    assert _strip_control_chars("hello\nworld") == "hello\nworld"
    assert _strip_control_chars("hello\tworld") == "hello\tworld"
    assert _strip_control_chars("hello\rworld") == "hello\rworld"
    # Mix
    assert _strip_control_chars("hello\x00\n\tworld\x1f") == "hello\n\tworld"
    # Normal text unchanged
    assert _strip_control_chars("Hello, world! Привет! 123") == "Hello, world! Привет! 123"
    # Empty
    assert _strip_control_chars("") == ""
    assert _strip_control_chars("   ") == "   "


# =============================================================================
# Minor #6: whitespace-only body fallback (проверяется в _split_text)
# =============================================================================

def test_whitespace_body_no_error():
    """
    Если body whitespace-only и не влезает, packed пуст,
    _split_text падает в fallback → возвращает title+footer, не бросает.
    """
    from newsbot.delivery.telegram import TelegramDelivery

    td = TelegramDelivery(token="test", chat_id="123")
    # _http не используется в _build_message_parts — это private helper

    item = NewsItem(
        source="Test",
        url="https://example.com/1",
        title="Test Title",
        summary="   \n\n\t\t   ",  # whitespace-only summary
        published="2024-01-01",
        published_parsed=None,
    )

    # Не должно бросить исключение — fallback на title+footer
    parts = td._build_message_parts(item)
    assert len(parts) >= 1, "Fallback должен вернуть хотя бы 1 part"
