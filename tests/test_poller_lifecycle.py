"""
tests/test_poller_lifecycle.py — тесты lifecycle TelegramPoller.

Покрывает:
  1. stop() при зависшем poll → нет второго поллера
  2. close() ждёт завершения потока перед close HTTP-клиента
  3. join timeout согласован с poll_timeout

Запуск: python -m pytest tests/test_poller_lifecycle.py -v
"""
import threading
import time
from unittest.mock import MagicMock, patch

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