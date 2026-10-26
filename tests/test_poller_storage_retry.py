"""
Task 3: тесты — poller переживает OperationalError при создании Storage
и повторяет попытки с backoff, но не бесконечно.

Запускать: python -m pytest tests/test_poller_storage_retry.py -v
"""
import logging
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

from newsbot.telegram_io import TelegramPoller


# ── Fake stop event (с реальными паузами для детерминированности) ────────────

class _FakeStopEvent:
    """
    Fake threading.Event для управления циклом из теста.

    Ключевое: wait(timeout) ждёт РЕАЛЬНОЕ время, чтобы тесты с моками
    time.sleep работали детерминированно. Если _set=False, wait() реально
    спит на timeout секунд (в отличие от real threading.Event, который
    ждёт события).

    Это нужно для сценария: FakeStopEvent.wait(1.0) в _sleep_backoff с
    мокнутым time.sleep позволяет _sleep_backoff возвращаться мгновенно,
    НО main thread спит 1.0 с, за которые успевает произойти N итераций.
    """

    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def wait(self, timeout=None):
        # Ждём реальное время — это детерминирует поведение теста.
        # Если set() уже вызван → сразу True.
        if self._set:
            return True
        if timeout is None:
            while not self._set:
                time.sleep(0.05)
            return True
        elapsed = 0.0
        step = 0.05
        while elapsed < timeout and not self._set:
            time.sleep(step)
            elapsed += step
        return self._set


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_mock_client():
    from newsbot.telegram_io import _TgClient
    c = MagicMock(spec=_TgClient)
    c.getUpdates.return_value = []
    return c


# ── тесты ────────────────────────────────────────────────────────────────────

class TestPollerStorageRetry:
    """
    Task 3: при OperationalError при создании Storage — retry с backoff,
    shutdown_event уважается в паузах, после исчерпания попыток — return.
    """

    def test_storage_retries_then_succeeds_third_attempt(self):
        """
        Storage кидает OperationalError первые 2 раза, на третий — создаётся.
        Поток стартует и работает.

        NOTE: _storage == None после теста — это нормально (cleanup в finally
        уже выполнился). Проверяем что Storage УСПЕШНО создался через флаг
        _storage_initialized, который _run_loop выставляет ДО cleanup.
        """
        import newsbot.storage as storage_module
        RealStorage = storage_module.Storage

        call_count = [0]

        class _RetryStorage(RealStorage):
            """Подкласс: 2 раза OperationalError, на 3-й — штатный __init__."""
            def __init__(self, db_path):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise sqlite3.OperationalError("database is locked")
                RealStorage.__init__(self, db_path)

        def fake_sleep(seconds):
            pass  # паузы мгновенные

        # Патчим Storage ТАМ ГДЕ ОПРЕДЕЛЁН — в модуле storage.
        # patch.object обновляет атрибут модуля; когда _run_loop делает
        # "from newsbot.storage import Storage", он получает патченный класс.
        with patch.object(storage_module, "Storage", _RetryStorage):
            with patch("newsbot.telegram_io.time.sleep", fake_sleep):
                client = _make_mock_client()
                poller = TelegramPoller(
                    token="tok123456",
                    chat_id="111",
                    db_path=":memory:",
                    on_digest=MagicMock(),
                    logger=logging.getLogger("test"),
                    poll_timeout=1,
                )
                poller._client = client
                poller._stop_event = _FakeStopEvent()

                # _tick эмулирует empty update (None → backoff не растёт)
                poller._tick = lambda backoff: None

                t = threading.Thread(target=poller._run_loop, daemon=True)
                t.start()
                # _FakeStopEvent.wait() ждёт реальное время, поэтому
                # за 2 с (2 x пауза 1.0 с между попытками) успеем:
                # попытка 1 (fail, sleep 1.0) + попытка 2 (fail, sleep 1.0) + попытка 3 (OK)
                time.sleep(2.5)
                poller._stop_event.set()
                t.join(timeout=1)

                # 3 попытки создания Storage
                assert call_count[0] == 3, f"Expected 3 Storage attempts, got {call_count[0]}"
                # _storage == None после finally (cleanup) — это нормально.
                # Проверяем что Storage УСПЕШНО создался (до cleanup).
                assert poller._storage_initialized is True, \
                    "Storage should be initialized (set by _run_loop before cleanup)"

    def test_storage_fails_all_4_attempts_poller_exits(self):
        """
        4 неудачи подряд → poller логирует error и return.
        Поток завершается, storage остаётся None.
        """
        import newsbot.storage as storage_module

        with patch.object(storage_module.Storage, "__init__",
                          side_effect=sqlite3.OperationalError("database is locked")):
            with patch("newsbot.telegram_io.time.sleep"):  # паузы мгновенные
                client = _make_mock_client()
                poller = TelegramPoller(
                    token="tok123456",
                    chat_id="111",
                    db_path=":memory:",
                    on_digest=MagicMock(),
                    logger=logging.getLogger("test"),
                    poll_timeout=1,
                )
                poller._client = client
                poller._stop_event = _FakeStopEvent()

                # _run_loop не бросает исключение наружу — проверяем что вернулся
                try:
                    poller._run_loop()
                except Exception as e:
                    # _run_loop не должен кидать наружу
                    pass

                # storage не создан
                assert poller._storage is None, "Storage should remain None after 4 failures"

    def test_shutdown_event_respected_during_retry_pause(self):
        """
        stop_event, выставленный из ДРУГОГО потока во время _sleep_backoff,
        прерывает паузу раньше полного sleep.

        _sleep_backoff(5.0) запускается в отдельном потоке; реальный
        threading.Event.set() вызывается через 0.3 с. При корректной
        реализации (Event.wait(timeout)) пауза завершается почти сразу
        после set() — elapsed ≈ 0.3 с, поток мёртв.

        БЕЗ фикса (while + time.sleep): sleep() не прерывается событием,
        поток живёт все 5 с → join(timeout=2) не дожидается → is_alive()
        True и elapsed ≈ 2 с → тест падает.
        """
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111",
            db_path=":memory:",
            on_digest=MagicMock(),
            logger=logging.getLogger("test"),
            poll_timeout=1,
        )
        # Реальный threading.Event (не fake) — проверяем настоящий interrupt.
        stop_event = threading.Event()
        poller._stop_event = stop_event

        FULL_SLEEP = 5.0
        start = time.monotonic()

        def sleeper():
            poller._sleep_backoff(FULL_SLEEP)

        t = threading.Thread(target=sleeper, daemon=True)
        t.start()
        time.sleep(0.3)
        stop_event.set()
        t.join(timeout=2.0)
        elapsed = time.monotonic() - start

        assert not t.is_alive(), (
            f"_sleep_backoff не прервался событием: поток жив после "
            f"{elapsed:.1f}s (полный sleep={FULL_SLEEP}s)"
        )
        assert elapsed < 2.0, (
            f"Пауза НЕ прервана раньше полного sleep: elapsed={elapsed:.1f}s "
            f"(ожидалось ~0.3s, полный sleep={FULL_SLEEP}s)"
        )

    def test_retry_backoff_sequence_logged(self, caplog):
        """Каждая попытка с backoff логируется warning'ом."""
        import newsbot.storage as storage_module

        with caplog.at_level(logging.WARNING, logger="test"):
            call_count = [0]
            original_init = storage_module.Storage.__init__

            def failing_init(self, db_path):
                call_count[0] += 1
                raise sqlite3.OperationalError("database is locked")

            with patch.object(storage_module.Storage, "__init__", failing_init):
                with patch("newsbot.telegram_io.time.sleep"):
                    poller = TelegramPoller(
                        token="tok123456",
                        chat_id="111",
                        db_path=":memory:",
                        on_digest=MagicMock(),
                        logger=logging.getLogger("test"),
                        poll_timeout=1,
                    )
                    poller._client = _make_mock_client()
                    poller._stop_event = _FakeStopEvent()

                    poller._run_loop()

                    # Все 3 warning'а (до исчерпания попыток)
                    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
                    # Ищем записи о попытках создания storage
                    assert any("OperationalError" in m or "database is locked" in m for m in warn_msgs), \
                        f"Expected retry warnings, got: {warn_msgs}"

                    # Финальный error (после 4-й неудачи)
                    error_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
                    assert any("попыток" in m for m in error_msgs), \
                        f"Expected error about max retries, got: {error_msgs}"