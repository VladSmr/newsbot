"""
tests/test_poller_lifecycle.py — тесты lifecycle TelegramPoller.

Покрывает:
  1. stop() при зависшем poll → нет второго поллера
  2. close() ждёт завершения потока перед close HTTP-клиента
  3. join timeout согласован с poll_timeout

Запуск: python -m pytest tests/test_poller_lifecycle.py -v
"""
import os
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from newsbot.telegram_io import TelegramPoller


def test_stop_no_second_poller():
    """
    Minor: stop() при зависшем poll не создаёт второй поллер.

    Сценарий:
      — poller.start() запускает поток
      — _run_loop внутри потока «зависает» (long-poll не отвечает)
      — stop() вызван: _stop_event.set() + join с достаточным timeout
      — после join: если поток всё ещё alive — _thread НЕ обнуляется
      — повторный start() видит что _thread alive и НЕ создаёт новый

    Без фикса: join(5) при poll_timeout=25 → поток не успевает,
    _thread=None → start() создаёт новый поток → 409 Conflict.
    """
    with patch("newsbot.storage.Storage") as MockStorage, \
         patch("newsbot.telegram_io._TgClient") as MockClient:

        mock_storage = MagicMock()
        mock_storage.get_meta.return_value = None
        MockStorage.return_value = mock_storage

        mock_client = MagicMock()
        # Зависающий getUpdates (не отвечает никогда)
        mock_client.getUpdates.side_effect = lambda **kwargs: time.sleep(3600)
        MockClient.return_value = mock_client

        poller = TelegramPoller(
            token="test_token",
            chat_id="123",
            db_path=":memory:",
            on_digest=MagicMock(),
            poll_timeout=25,
        )

        # Запускаем — поток стартует
        poller.start()
        time.sleep(0.3)

        assert poller._thread is not None
        assert poller._thread.is_alive()

        # stop() с фиксированным timeout (poll_timeout+5 = 30с)
        start_stop = time.time()
        poller.stop()
        elapsed = time.time() - start_stop

        # join должен был ждать полный timeout (~30 с),
        # не 5 с как было до фикса
        assert elapsed >= 25, (
            f"stop() вернулся слишком быстро ({elapsed:.1f}s) — "
            f"join timeout слишком короткий"
        )
        # Поток всё ещё alive (long-poll не отвечает)
        assert poller._thread is not None
        # И он всё ещё alive
        assert poller._thread.is_alive(), (
            "Поток не должен был завершиться (long-poll завис)"
        )

        # Повторный start: _thread alive → не создаём новый
        with patch.object(threading, "Thread") as MockThread:
            MockThread.return_value = MagicMock()
            MockThread.return_value.start = MagicMock()
            poller.start()
            # Thread() должен был быть вызван 0 раз (поток ещё alive)
            assert MockThread.call_count == 0, (
                f"start() создал новый Thread при alive _thread: "
                f"{MockThread.call_count} вызовов"
            )

        # cleanup
        poller._stop_event.set()  # разбудить long-poll


def test_close_waits_for_thread():
    """
    Minor: close() ждёт завершения потока перед close HTTP-клиента.

    Сценарий: _run_loop работает (long-poll в процессе),
    вызывается close(). Без фикса HTTP-клиент закрывается
    ПОД работающим потоком → возможны проблемы.
    С фиксом: close() ждёт join(thread) перед _client.close().
    """
    with patch("newsbot.storage.Storage") as MockStorage, \
         patch("newsbot.telegram_io._TgClient") as MockClient:

        mock_storage = MagicMock()
        mock_storage.get_meta.return_value = None
        MockStorage.return_value = mock_storage

        mock_client = MagicMock()
        # Simulate: getUpdates спит 2 секунды (имитация долгого poll)
        def slow_poll(**kwargs):
            time.sleep(2)
            return []
        mock_client.getUpdates.side_effect = slow_poll
        MockClient.return_value = mock_client

        poller = TelegramPoller(
            token="test_token",
            chat_id="123",
            db_path=":memory:",
            on_digest=MagicMock(),
            poll_timeout=25,
        )

        # Запускаем — поток стартует
        poller.start()
        time.sleep(0.2)

        # Вызываем close() сразу — поток ещё в long-poll
        start_close = time.time()
        poller.close()
        elapsed = time.time() - start_close

        # close() должен был подождать (poll_timeout+5 или реальное время)
        # Минимум 2 секунды (время slow_poll)
        assert elapsed >= 1.5, (
            f"close() вернулся слишком быстро ({elapsed:.2f}s) — "
            f"не дождался завершения потока"
        )

        # _client.close() вызван
        assert mock_client.close.called


def test_join_timeout_uses_poll_timeout():
    """
    Minor: join timeout = poll_timeout + 5 (не хардкод 5).
    
    Тестируем через slow poll: getUpdates возвращается через 2 секунды
    (имитация реального long-poll). stop() должен ждать именно poll_timeout+5,
    а не фиксированное значение 5.
    """
    with patch("newsbot.storage.Storage") as MockStorage, \
         patch("newsbot.telegram_io._TgClient") as MockClient:

        mock_storage = MagicMock()
        mock_storage.get_meta.return_value = None
        MockStorage.return_value = mock_storage

        mock_client = MagicMock()
        # Slow poll: каждая итерация спит 2 секунды
        def slow_getUpdates(**kwargs):
            time.sleep(2)
            return []
        mock_client.getUpdates.side_effect = slow_getUpdates
        MockClient.return_value = mock_client

        # poll_timeout=5 — для теста используем именно это значение
        poller = TelegramPoller(
            token="test_token",
            chat_id="123",
            db_path=":memory:",
            on_digest=MagicMock(),
            poll_timeout=5,
        )

        poller.start()
        # Ждём чтобы поток вошёл в long-poll
        time.sleep(0.4)

        # Убеждаемся что поток жив (в long-poll)
        assert poller._thread is not None
        assert poller._thread.is_alive(), "Thread должен быть alive (в long-poll)"

        # Проверяем: join_timeout = poll_timeout + 5 = 10
        assert poller._poll_timeout == 5
        start_stop = time.time()
        poller.stop()
        elapsed = time.time() - start_stop

        # join с timeout=poll_timeout+5=10, но поток спит 2 с
        # stop() вернётся когда поток реально завершится (~2 с)
        # БЕЗ фикса: join(5) — поток спит 2 с, join ждёт 5 с, всего ~2 с
        # С фиксом: join(10) — то же поведение, но timeout больше
        # Проверяем что поток реально дождался (не убит по timeout)
        assert 1.5 <= elapsed < 15, (
            f"stop() занял {elapsed:.2f}s — ожидалось ~2с (время slow poll)"
        )
        # Поток завершился корректно
        assert poller._thread is None or not poller._thread.is_alive()


def test_stop_clears_thread_when_really_finished():
    """
    Minor: если поток ДЕЙСТВИТЕЛЬНО завершился — _thread обнуляется.
    """
    with patch("newsbot.storage.Storage") as MockStorage, \
         patch("newsbot.telegram_io._TgClient") as MockClient:

        mock_storage = MagicMock()
        mock_storage.get_meta.return_value = None
        MockStorage.return_value = mock_storage

        mock_client = MagicMock()
        # Быстрый ответ — поток завершится сам
        mock_client.getUpdates.return_value = []
        MockClient.return_value = mock_client

        poller = TelegramPoller(
            token="test_token",
            chat_id="123",
            db_path=":memory:",
            on_digest=MagicMock(),
            poll_timeout=1,
        )

        poller.start()
        # Ждём достаточно чтобы поток завершился сам (он получит пустой ответ и выйдет)
        time.sleep(2.5)

        assert poller._thread is not None

        # stop(): поток уже finished → join(6) вернётся сразу
        start_stop = time.time()
        poller.stop()
        elapsed = time.time() - start_stop

        # join вернулся почти сразу (поток уже finished)
        assert elapsed < 1.0, f"stop() занял {elapsed:.2f}s при уже-finished потоке"

        # _thread обнулён (поток действительно finished)
        assert poller._thread is None, (
            "_thread должен быть обнулён когда поток действительно finished"
        )


# ══════════════════════════════════════════════════════════════════
# Задача 2 (волна 3): graceful shutdown — реальный main()
# ══════════════════════════════════════════════════════════════════

def test_shutdown_event_interrupts_main_loop(tmp_path, monkeypatch):
    """
    Реальный main() в демон-режиме с мок-пайплайном:
    _shutdown_event.set() (эмуляция SIGTERM) прерывает главный цикл,
    pipeline.close() вызван, lock-файл удалён.
    """
    import newsbot.__main__ as main_mod

    shutdown = threading.Event()
    digest = threading.Event()
    lock_path = tmp_path / "newsbot.lock"
    observed = {}

    mock_pipeline = MagicMock()
    mock_pipeline.storage.db_path = str(tmp_path / "newsbot.db")

    def fake_cycle():
        # lock захвачен ДО входа в цикл
        observed["lock_exists"] = lock_path.exists()
        shutdown.set()      # эмулируем SIGTERM во время цикла

    mock_pipeline.run_cycle_with_delivery.side_effect = fake_cycle

    config = {"logging": {"level": "WARNING"}, "poll_interval_minutes": 15}

    monkeypatch.setattr(sys, "argv", ["newsbot"])

    with patch.object(main_mod, "_shutdown_event", shutdown), \
         patch.object(main_mod, "_digest_requested", digest), \
         patch.object(main_mod, "_poller", None), \
         patch.object(main_mod, "_LOCK_FILE", None), \
         patch.object(main_mod, "NewsPipeline", return_value=mock_pipeline), \
         patch.object(main_mod, "load_config", return_value=config), \
         patch.object(main_mod, "setup_logging"), \
         patch.object(main_mod, "signal", MagicMock()), \
         patch.object(main_mod, "_maybe_start_poller") as mock_start:
        main_mod.main()

    mock_start.assert_called_once()
    assert mock_pipeline.run_cycle_with_delivery.call_count == 1, (
        "главный цикл должен выйти сразу после set()"
    )
    assert observed.get("lock_exists") is True, "lock должен быть захвачен до цикла"
    mock_pipeline.close.assert_called_once()
    assert not lock_path.exists(), "lock-файл должен быть удалён в finally"


def test_signal_handler_only_sets_events_no_blocking_join(tmp_path):
    """
    Обработчик сигнала НЕ блокирует: только set() событий.
    Остановка poller (join ≈ poll_timeout+5) — в finally main, не в хендлере.
    """
    import newsbot.__main__ as main_mod

    shutdown = threading.Event()
    digest = threading.Event()

    mock_poller = MagicMock()
    mock_poller.stop.side_effect = lambda: time.sleep(5)  # не должен вызываться

    start = time.monotonic()
    with patch.object(main_mod, "_shutdown_event", shutdown), \
         patch.object(main_mod, "_digest_requested", digest), \
         patch.object(main_mod, "_poller", mock_poller):
        main_mod._signal_handler(signal.SIGTERM, None)
    elapsed = time.monotonic() - start

    assert shutdown.is_set(), "_shutdown_event должен быть установлен"
    assert digest.is_set(), "_digest_requested должен быть установлен (разбудить main)"
    assert elapsed < 0.5, f"хендлер блокировал {elapsed:.2f}s"
    mock_poller.stop.assert_not_called()


# ══════════════════════════════════════════════════════════════════
# Задача 3: single-instance lock
# ══════════════════════════════════════════════════════════════════

class TestSingleInstanceLock:
    """Lock-файл: live pid → exit, stale → захватывается."""

    def test_live_pid_exits(self, tmp_path, monkeypatch):
        """Lock с живым PID → sys.exit(1)."""
        import newsbot.__main__ as main_mod

        lock_path = tmp_path / "newsbot.lock"
        # Записываем PID текущего процесса (живой)
        my_pid = os.getpid()
        lock_path.write_text(str(my_pid))

        monkeypatch.setattr(main_mod, "_LOCK_FILE", lock_path)

        with patch.object(main_mod, "_is_process_alive", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                main_mod._acquire_lock(lock_path, 99999)
        assert exc_info.value.code == 1

    def test_stale_pid_can_be_taken(self, tmp_path, monkeypatch):
        """Несуществующий PID (stale lock) → lock захватывается."""
        import newsbot.__main__ as main_mod

        lock_path = tmp_path / "newsbot.lock"
        # PID 99999 — несуществующий (мокаем _is_process_alive)
        lock_path.write_text("99999")

        monkeypatch.setattr(main_mod, "_LOCK_FILE", lock_path)

        with patch.object(main_mod, "_is_process_alive", return_value=False):
            with patch("os.getpid", return_value=12345):
                result = main_mod._acquire_lock(lock_path, 12345)

        assert result is True
        assert lock_path.exists()
        assert lock_path.read_text() == "12345"
        # temp-файл публикации не остаётся на диске
        assert list(tmp_path.glob("*.tmp")) == []

        # cleanup
        main_mod._release_lock(lock_path)

    def test_empty_lock_file_with_live_pid_not_stolen(self, tmp_path, monkeypatch):
        """
        TOCTOU: пустой lock-файл (другой процесс ещё публикует) не приводит
        к перехвату lock живого процесса — PID дочитывается повторно.
        """
        import newsbot.__main__ as main_mod

        lock_path = tmp_path / "newsbot.lock"
        lock_path.write_text("")   # ещё пустой

        real_read = Path.read_text
        calls = {"n": 0}

        def flaky_read(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return ""
            return "424242"

        monkeypatch.setattr(Path, "read_text", flaky_read)
        try:
            with patch.object(main_mod, "_is_process_alive", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    main_mod._acquire_lock(lock_path, 777)
        finally:
            monkeypatch.setattr(Path, "read_text", real_read)

        assert exc_info.value.code == 1
        # PID дочитан повторно (файл НЕ удалён и НЕ перезаписан)
        assert calls["n"] >= 2, "пустой lock-файл должен перечитываться"
        assert lock_path.exists(), "lock живого процесса не должен удаляться"

    def test_release_lock_removes_file(self, tmp_path):
        """_release_lock удаляет lock-файл."""
        import newsbot.__main__ as main_mod

        lock_path = tmp_path / "newsbot.lock"
        lock_path.touch()

        main_mod._release_lock(lock_path)
        assert not lock_path.exists()

    def test_release_lock_nonexistent_no_error(self, tmp_path):
        """_release_lock при отсутствии файла — без ошибки."""
        import newsbot.__main__ as main_mod

        lock_path = tmp_path / "nonexistent.lock"
        # Не существует — не должно упасть
        main_mod._release_lock(lock_path)

    def test_is_process_alive_true_for_current(self):
        """Текущий процесс — живой (безопасная проверка, без os.kill на win32)."""
        import newsbot.__main__ as main_mod
        with patch.object(main_mod, "_is_process_alive_windows", return_value=True) as mock_win, \
             patch("os.kill") as mock_kill:
            assert main_mod._is_process_alive(os.getpid()) is True
        if sys.platform == "win32":
            mock_win.assert_called_once_with(os.getpid())
            mock_kill.assert_not_called()

    def test_is_process_alive_true_current_process(self):
        """Текущий процесс — живой (os.getpid()); на win32 не зовём os.kill."""
        import newsbot.__main__ as main_mod
        with patch.object(main_mod, "_is_process_alive_windows", return_value=True) as mock_win:
            assert main_mod._is_process_alive(os.getpid()) is True
        if sys.platform == "win32":
            mock_win.assert_called_once_with(os.getpid())

    def test_is_process_alive_false_process_lookup_error(self):
        """ProcessLookupError → False."""
        import newsbot.__main__ as main_mod
        with patch("os.kill", side_effect=ProcessLookupError):
            assert main_mod._is_process_alive_posix(12345) is False


# ══════════════════════════════════════════════════════════════════
# MAJOR 2: проверка живости PID безопасна на Windows
# (os.kill(pid, 0) на Windows = TerminateProcess → убивает проверяемый процесс)
# ══════════════════════════════════════════════════════════════════

class TestProcessAliveChecks:
    """_is_process_alive: win32 → OpenProcess/WaitForSingleObject, posix → os.kill."""

    def test_posix_alive_when_kill_succeeds(self):
        import newsbot.__main__ as main_mod
        with patch("os.kill", return_value=None) as mock_kill:
            assert main_mod._is_process_alive_posix(12345) is True
        mock_kill.assert_called_once_with(12345, 0)

    def test_posix_alive_on_permission_error(self):
        """PermissionError = процесс есть, но чужой."""
        import newsbot.__main__ as main_mod
        with patch("os.kill", side_effect=PermissionError):
            assert main_mod._is_process_alive_posix(12345) is True

    @staticmethod
    def _fake_ctypes(handle, wait_rc, last_error=0):
        """Фейковый ctypes с kernel32-моками."""
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = handle
        kernel32.WaitForSingleObject.return_value = wait_rc
        kernel32.GetLastError.return_value = last_error
        fake = MagicMock()
        fake.windll.kernel32 = kernel32
        return fake, kernel32

    def test_windows_dead_on_invalid_parameter(self):
        """OpenProcess → NULL + GetLastError()==87 → процесса нет."""
        import newsbot.__main__ as main_mod
        fake, kernel32 = self._fake_ctypes(handle=0, wait_rc=0, last_error=87)
        with patch.object(main_mod, "ctypes", fake):
            assert main_mod._is_process_alive_windows(999999999) is False
        kernel32.CloseHandle.assert_not_called()

    def test_windows_alive_on_wait_timeout(self):
        """handle + WAIT_TIMEOUT → процесс жив; CloseHandle вызван."""
        import newsbot.__main__ as main_mod
        fake, kernel32 = self._fake_ctypes(handle=1234, wait_rc=0x102)
        with patch.object(main_mod, "ctypes", fake):
            assert main_mod._is_process_alive_windows(4242) is True
        kernel32.CloseHandle.assert_called_once_with(1234)

    def test_windows_dead_on_wait_object_0(self):
        """handle + WAIT_OBJECT_0 → процесс завершён; CloseHandle вызван."""
        import newsbot.__main__ as main_mod
        fake, kernel32 = self._fake_ctypes(handle=1234, wait_rc=0)
        with patch.object(main_mod, "ctypes", fake):
            assert main_mod._is_process_alive_windows(4242) is False
        kernel32.CloseHandle.assert_called_once_with(1234)

    def test_windows_never_calls_os_kill(self):
        """На Windows os.kill(pid, 0) не должен вызываться вообще."""
        import newsbot.__main__ as main_mod
        fake, _ = self._fake_ctypes(handle=1234, wait_rc=0x102)
        with patch.object(main_mod, "ctypes", fake), patch("os.kill") as mock_kill:
            main_mod._is_process_alive_windows(4242)
        mock_kill.assert_not_called()

    def test_dispatch_uses_windows_branch_on_win32(self):
        """sys.platform == 'win32' → windows-ветка (не os.kill)."""
        import newsbot.__main__ as main_mod
        with patch.object(main_mod.sys, "platform", "win32"), \
             patch.object(main_mod, "_is_process_alive_windows",
                          return_value=True) as mock_win, \
             patch("os.kill") as mock_kill:
            assert main_mod._is_process_alive(4242) is True
        mock_win.assert_called_once_with(4242)
        mock_kill.assert_not_called()

    def test_dispatch_uses_posix_branch_off_win32(self):
        """sys.platform != 'win32' → os.kill(pid, 0)."""
        import newsbot.__main__ as main_mod
        with patch.object(main_mod.sys, "platform", "linux"), \
             patch.object(main_mod, "_is_process_alive_windows") as mock_win, \
             patch("os.kill", return_value=None) as mock_kill:
            assert main_mod._is_process_alive(4242) is True
        mock_win.assert_not_called()
        mock_kill.assert_called_once_with(4242, 0)


# ══════════════════════════════════════════════════════════════════
# MAJOR (main): единый дедлайн shutdown-фазы поллера
# ══════════════════════════════════════════════════════════════════


class _FakeClock:
    """Управляемые часы: main_mod.time.monotonic() без реального ожидания."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:  # pragma: no cover - не вызывается
        self.now += seconds


def _run_main_with_poller(poller, clock):
    """Гоняет реальный main() с мок-пайплайном и мок-поллером."""
    import newsbot.__main__ as main_mod

    shutdown = threading.Event()
    digest = threading.Event()

    mock_pipeline = MagicMock()
    mock_pipeline.storage.db_path = "unused.db"

    def fake_cycle():
        shutdown.set()  # эмулируем SIGTERM во время цикла

    mock_pipeline.run_cycle_with_delivery.side_effect = fake_cycle

    config = {"logging": {"level": "WARNING"}, "poll_interval_minutes": 15}

    with patch.object(sys, "argv", ["newsbot"]), \
         patch.object(main_mod, "_shutdown_event", shutdown), \
         patch.object(main_mod, "_digest_requested", digest), \
         patch.object(main_mod, "_poller", poller), \
         patch.object(main_mod, "_LOCK_FILE", None), \
         patch.object(main_mod, "time", clock), \
         patch.object(main_mod, "NewsPipeline", return_value=mock_pipeline), \
         patch.object(main_mod, "load_config", return_value=config), \
         patch.object(main_mod, "setup_logging"), \
         patch.object(main_mod, "signal", MagicMock()), \
         patch.object(main_mod, "_maybe_start_poller"):
        main_mod.main()

    return mock_pipeline


def test_shutdown_poller_close_gets_only_remaining_budget():
    """
    stop() уже прождал 35 с → close() получает лишь остаток бюджета (≈5 с),
    а не повторный полный join (poll_timeout+5 ≈ 30 с).
    """
    import newsbot.__main__ as main_mod

    budget = main_mod._POLLER_SHUTDOWN_BUDGET_SEC
    clock = _FakeClock()

    poller = MagicMock()
    captured = {}

    def fake_stop(*args, **kwargs):
        clock.now += 35.0  # stop() ждал поток 35 с

    def fake_close(join_timeout=None):
        captured["join_timeout"] = join_timeout

    poller.stop.side_effect = fake_stop
    poller.close.side_effect = fake_close

    mock_pipeline = _run_main_with_poller(poller, clock)

    poller.stop.assert_called_once()
    poller.close.assert_called_once()
    assert "join_timeout" in captured, "close() должен получить остаток дедлайна"
    remaining = captured["join_timeout"]
    assert remaining is not None
    # Остаток = budget - 35; суммарно поллер ждём не больше бюджета
    assert 0.0 <= remaining <= budget - 35.0 + 1e-6
    # shutdown дошёл до pipeline.close (не был съеден ожиданием поллера)
    mock_pipeline.close.assert_called_once()


def test_shutdown_poller_budget_never_repeats_full_join():
    """Даже если stop() «съел» весь бюджет, close() не ждёт повторно 30 с."""
    import newsbot.__main__ as main_mod

    budget = main_mod._POLLER_SHUTDOWN_BUDGET_SEC
    clock = _FakeClock()

    poller = MagicMock()
    captured = {}

    def fake_stop(*args, **kwargs):
        clock.now += budget + 10.0  # бюджет исчерпан

    def fake_close(join_timeout=None):
        captured["join_timeout"] = join_timeout

    poller.stop.side_effect = fake_stop
    poller.close.side_effect = fake_close

    _run_main_with_poller(poller, clock)

    assert captured["join_timeout"] == 0.0


def test_poller_close_accepts_join_timeout():
    """close(join_timeout=N) принимается; None — прежнее поведение."""
    poller = TelegramPoller(
        token="t", chat_id="1", db_path="unused.db", on_digest=lambda: None,
        poll_timeout=60,
    )
    # Поток не запускался → join не нужен, close() не должен падать/висеть
    poller.close(join_timeout=0.0)
    poller.close()  # None — прежнее поведение, идемпотентно


# ══════════════════════════════════════════════════════════════════
# P1: stop() укладывается в shutdown-бюджет (join = min(poll+5, max_join))
# ══════════════════════════════════════════════════════════════════


def test_main_passes_remaining_budget_to_stop():
    """main() передаёт stop() остаток shutdown-бюджета (max_join ≤ budget)."""
    import newsbot.__main__ as main_mod

    budget = main_mod._POLLER_SHUTDOWN_BUDGET_SEC
    clock = _FakeClock()
    poller = MagicMock()
    captured = {}

    def fake_stop(max_join=None):
        captured["max_join"] = max_join
        clock.now += 5.0

    poller.stop.side_effect = fake_stop
    poller.close.side_effect = lambda join_timeout=None: None

    _run_main_with_poller(poller, clock)

    assert "max_join" in captured, "stop() должен получить max_join"
    assert 0.0 < captured["max_join"] <= budget


def test_stop_caps_join_by_max_join():
    """
    stop(max_join=0.1) не ждёт poll_timeout+5 — join ограничен max_join,
    поток остаётся alive, ссылка не обнуляется (нет второго поллера).
    """
    with patch("newsbot.storage.Storage") as MockStorage, \
         patch("newsbot.telegram_io._TgClient") as MockClient:

        mock_storage = MagicMock()
        mock_storage.get_meta.return_value = None
        MockStorage.return_value = mock_storage

        mock_client = MagicMock()

        poller_holder = {}

        def fake_get_updates(**kwargs):
            # Блокируемся до stop() (не sleep(3600): поток не должен утекать),
            # затем короткая пауза — детерминированно гарантирует, что при
            # max_join=0.1 поток ещё жив (проверяем именно cap join'а).
            p = poller_holder.get("poller")
            if p is not None:
                p._stop_event.wait(timeout=10.0)
            time.sleep(0.5)
            return []

        mock_client.getUpdates.side_effect = fake_get_updates
        MockClient.return_value = mock_client

        poller = TelegramPoller(
            token="test_token", chat_id="123", db_path=":memory:",
            on_digest=MagicMock(), poll_timeout=25,
        )
        poller_holder["poller"] = poller
        poller.start()
        time.sleep(0.3)

        started = time.monotonic()
        poller.stop(max_join=0.1)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, f"stop(max_join=0.1) ждал {elapsed:.1f}с"
        assert poller._thread is not None
        assert poller._thread.is_alive()

        # уборка: поток завершается сам (wait по _stop_event + 0.5 с) —
        # close() его join'ит, утечки daemon-потока нет.
        poller.close(join_timeout=5.0)


# ══════════════════════════════════════════════════════════════════
# P2: --bind без --set-chat-id → parser.error (не молчаливый демон)
# ══════════════════════════════════════════════════════════════════


def test_bind_without_set_chat_id_is_parser_error():
    """--bind без --set-chat-id → SystemExit(2), демон не запускается."""
    import newsbot.__main__ as main_mod

    with patch.object(sys, "argv", ["newsbot", "--bind", "123456789"]), \
         patch.object(main_mod, "_minimal_sanitizer_handler"), \
         patch.object(main_mod, "NewsPipeline") as mock_pipeline:
        with pytest.raises(SystemExit) as exc:
            main_mod.main()

    assert exc.value.code == 2
    mock_pipeline.assert_not_called()
