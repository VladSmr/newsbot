"""
tests/test_pause.py — тесты паузы (/stop, /continue).

Проверяемые сценарии:
  1) /stop пишет meta paused=1, /continue → 0; подтверждения отправляются.
  2) pipeline при паузе: LLM не вызывается, доставки нет, дайджест не идёт,
     баланс-алерт не идёт; items остаются 'new'.
  3) Пауза приоритетнее off-peak.
  4) /digest при паузе — заглушка, событие не выставлено.
  5) Mid-batch пауза: обработка останавливается, остаток → pending.
  6) help/start содержат новые команды.
  7) Поллер стартует без feedback.enabled.
"""
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from newsbot.telegram_io import TelegramPoller, _Commands


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class _FakeStopEvent:
    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def wait(self, timeout=None):
        return self._set


def _make_mock_client():
    c = MagicMock()
    c.getUpdates.return_value = []
    c.answerCallbackQuery.return_value = None
    c.sendMessage.return_value = None
    c.editMessageReplyMarkup.return_value = None
    return c


def _make_msg(text: str, chat_id: str = "111222333"):
    return {"message_id": 1, "chat": {"id": chat_id}, "text": text}


def _build_poller(storage_mock=None, client_mock=None,
                  on_digest=None, chat_id="111222333",
                  feedback_enabled=True):
    on_digest = on_digest or MagicMock()
    poller = TelegramPoller(
        token="tok123456789",
        chat_id=chat_id,
        db_path=":memory:",
        on_digest=on_digest,
        logger=logging.getLogger("test"),
        poll_timeout=1,
        feedback_enabled=feedback_enabled,
    )
    poller._storage = storage_mock or MagicMock()
    poller._client = client_mock or _make_mock_client()
    poller._stop_event = _FakeStopEvent()
    return poller


def _tmp_db():
    """Возвращает путь к временному файлу БД (удаляется вручную)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _cleanup_db(path):
    """Безопасное удаление tmp-файла БД + WAL-мусора (Windows)."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except (FileNotFoundError, PermissionError):
            time.sleep(0.1)
            try:
                os.unlink(path + suffix)
            except (FileNotFoundError, PermissionError):
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. /stop, /continue — запись meta + подтверждения
# ══════════════════════════════════════════════════════════════════════════════

class TestStopCommand(unittest.TestCase):

    def test_stop_writes_meta_paused_1(self):
        storage = MagicMock()
        poller = _build_poller(storage_mock=storage)

        upd = {"update_id": 50, "message": _make_msg("/stop")}
        poller._process_update(upd)

        storage.set_meta.assert_called_with("paused", "1")
        poller._client.sendMessage.assert_called()
        args = poller._client.sendMessage.call_args[0]
        self.assertEqual(args[0], "111222333")
        self.assertIn("⏸", args[1])

    def test_stop_exception_logged_does_not_crash(self):
        """При exception в set_meta — /stop логирует, НЕ падает."""
        storage = MagicMock()
        storage.set_meta.side_effect = RuntimeError("DB error")
        poller = _build_poller(storage_mock=storage)

        upd = {"update_id": 51, "message": _make_msg("/stop")}
        # не падает (C2)
        poller._process_update(upd)
        # sendMessage НЕ вызван (упал до него)
        poller._client.sendMessage.assert_not_called()


class TestContinueCommand(unittest.TestCase):

    def test_continue_writes_meta_paused_0(self):
        storage = MagicMock()
        poller = _build_poller(storage_mock=storage)

        upd = {"update_id": 52, "message": _make_msg("/continue")}
        poller._process_update(upd)

        storage.set_meta.assert_called_with("paused", "0")
        args = poller._client.sendMessage.call_args[0]
        self.assertEqual(args[0], "111222333")
        self.assertIn("▶️", args[1])

    def test_continue_exception_logged_does_not_crash(self):
        storage = MagicMock()
        storage.set_meta.side_effect = RuntimeError("DB error")
        poller = _build_poller(storage_mock=storage)

        upd = {"update_id": 53, "message": _make_msg("/continue")}
        # не падает
        poller._process_update(upd)


# ══════════════════════════════════════════════════════════════════════════════
# 2. /digest при паузе — заглушка, _on_digest НЕ вызван
# ══════════════════════════════════════════════════════════════════════════════

class TestDigestWhilePaused(unittest.TestCase):

    def test_digest_paused_returns_placeholder_and_does_not_signal(self):
        """При паузе /digest не дёргает _on_digest."""
        storage = MagicMock()
        storage.get_meta.return_value = "1"  # paused
        on_digest = MagicMock()
        poller = _build_poller(storage_mock=storage, on_digest=on_digest)

        upd = {"update_id": 54, "message": _make_msg("/digest")}
        poller._process_update(upd)

        # _on_digest НЕ вызван
        on_digest.assert_not_called()
        # но ответ-заглушка отправлен
        poller._client.sendMessage.assert_called()
        args = poller._client.sendMessage.call_args[0]
        self.assertIn("паузе", args[1])

    def test_digest_not_paused_signals(self):
        """Без паузы /digest вызывает _on_digest."""
        storage = MagicMock()
        storage.get_meta.return_value = None  # не на паузе
        on_digest = MagicMock()
        poller = _build_poller(storage_mock=storage, on_digest=on_digest)

        upd = {"update_id": 55, "message": _make_msg("/digest")}
        poller._process_update(upd)

        on_digest.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 3. /help и /start содержат /stop и /continue
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpContainsPauseCommands(unittest.TestCase):

    def test_help_contains_stop(self):
        poller = _build_poller()
        upd = {"update_id": 56, "message": _make_msg("/help")}
        poller._process_update(upd)
        args = poller._client.sendMessage.call_args[0]
        self.assertIn("/stop", args[1])

    def test_help_contains_continue(self):
        poller = _build_poller()
        upd = {"update_id": 57, "message": _make_msg("/help")}
        poller._process_update(upd)
        args = poller._client.sendMessage.call_args[0]
        self.assertIn("/continue", args[1])

    def test_start_contains_stop_and_continue(self):
        poller = _build_poller()
        upd = {"update_id": 58, "message": _make_msg("/start")}
        poller._process_update(upd)
        args = poller._client.sendMessage.call_args[0]
        self.assertIn("/stop", args[1])
        self.assertIn("/continue", args[1])


# ══════════════════════════════════════════════════════════════════════════════
# 4. _is_paused в pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineIsPaused(unittest.TestCase):
    """Тесты _is_paused через реальный Storage с tempfile."""

    def test_is_paused_true_when_meta_1(self):
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            st.set_meta("paused", "1")
            cfg = self._minimal_config()
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        self.assertTrue(pipeline._is_paused())
                        pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def test_is_paused_false_when_meta_0(self):
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            st.set_meta("paused", "0")
            cfg = self._minimal_config()
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        self.assertFalse(pipeline._is_paused())
                        pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def test_is_paused_false_when_meta_absent(self):
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            cfg = self._minimal_config()
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        self.assertFalse(pipeline._is_paused())
                        pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def test_is_paused_exception_safe(self):
        from newsbot.pipeline import NewsPipeline
        cfg = self._minimal_config()
        with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
            with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                with patch.object(NewsPipeline, "_resolve_db_path", return_value=_tmp_db()):
                    pipeline = NewsPipeline(cfg)
                    pipeline.storage = MagicMock()
                    pipeline.storage.get_meta.side_effect = RuntimeError("DB error")
                    self.assertFalse(pipeline._is_paused())

    def _minimal_config(self):
        return {
            "sources": [],
            "llm": {"enabled": False},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": True},
            "database": {"path": "data/newsbot.db"},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline run_cycle при паузе — доставка/баланс не вызываются
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelinePausedGate(unittest.TestCase):
    """Тесты через реальный Storage с tempfile (путь корректно резолвится)."""

    def test_paused_delivery_not_called(self):
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            st.set_meta("paused", "1")
            cfg = self._make_config()
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                delivery = MagicMock()
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.run_cycle_with_delivery()
                        delivery.send_batch.assert_not_called()
                        pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def test_paused_items_remain_new(self):
        """При паузе set_state не вызывается (items остаются в state='new')."""
        from newsbot.pipeline import NewsPipeline
        cfg = self._make_config()
        storage = MagicMock()
        storage.get_meta.return_value = "1"
        storage.get_all_by_state.return_value = []
        storage.get_stuck_pending.return_value = []

        with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
            with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                with patch.object(NewsPipeline, "_resolve_db_path", return_value=_tmp_db()):
                    pipeline = NewsPipeline(cfg)
                    pipeline.storage = storage
                    pipeline.run_cycle_with_delivery()
                    storage.set_state.assert_not_called()

    def test_paused_no_balance_alert(self):
        """При паузе баланс-алерт (send_text) не отправляется."""
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            st.set_meta("paused", "1")
            cfg = self._make_config()
            cfg["llm"]["balance"] = {"enabled": True, "threshold": 2.0,
                                    "check_interval_hours": 24}
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                delivery = MagicMock()
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()
                        pipeline.run_cycle_with_delivery()
                        delivery.send_text.assert_not_called()
                        pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def test_paused_cleanup_still_runs(self):
        """При паузе purge_seen_titles и purge_items выполняются."""
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            real_st = Storage(path)
            real_st.set_meta("paused", "1")
            # Minor #3: spy — подменяем только методы purge, остальное идёт в реальный storage
            st = MagicMock()
            for attr in dir(real_st):
                if not attr.startswith("_"):
                    try:
                        val = getattr(real_st, attr)
                        if callable(val) and attr not in ("purge_seen_titles", "purge_items"):
                            setattr(st, attr, val)
                    except Exception:
                        pass
            st.purge_seen_titles = MagicMock(return_value=0)
            st.purge_items = MagicMock(return_value=0)
            cfg = self._make_config()
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.run_cycle_with_delivery()
                        # Minor #3: проверяем что purge реально вызван (spy)
                        self.assertTrue(st.purge_seen_titles.called, "purge_seen_titles должен быть вызван")
                        self.assertTrue(st.purge_items.called, "purge_items должен быть вызван")
                        pipeline.storage = MagicMock()
        finally:
            real_st.close()
            _cleanup_db(path)

    def test_paused_logs_waiting_count(self):
        """При паузе логируется сообщение с PAUSED gate."""
        from newsbot.pipeline import NewsPipeline
        import logging as _logging
        cfg = self._make_config()
        storage = MagicMock()
        storage.get_meta.return_value = "1"
        storage.get_all_by_state.return_value = []
        storage.get_stuck_pending.return_value = []

        with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
            with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                with patch.object(NewsPipeline, "_resolve_db_path", return_value=_tmp_db()):
                    pipeline = NewsPipeline(cfg)
                    pipeline.storage = storage

                    # мокаем logger напрямую
                    with patch.object(_logging.getLogger("newsbot.pipeline"), "info") as log_info:
                        pipeline.run_cycle_with_delivery()
                        pause_logs = [
                            c for c in log_info.call_args_list
                            if "PAUSED" in str(c)
                        ]
                        self.assertTrue(len(pause_logs) > 0)

    def test_paused_mid_cycle_skips_digest_and_balance_runs_purge(self):
        """
        Major #1: пауза, наступившая В СЕРЕДИНЕ цикла после батчей
        (mid-batch → _is_paused вернула True внутри _process_batch,
        но управление вернулось в run_cycle_with_delivery),
        должна пропустить _maybe_run_digest и _maybe_check_balance,
        но выполнить purge_seen_titles и purge_items.

        Проверка: спай на _maybe_run_digest / _maybe_check_balance с
        assert_not_called (а не косвенно через send_text — порог баланса
        мог и не сработать).
        """
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        from newsbot.collector import NewsItem
        from newsbot.llm.filter import FilterResult
        from datetime import datetime, timezone

        path = _tmp_db()
        try:
            real_st = Storage(path)
            # пауза НЕ активна на верхнем уровне (чтобы цикл прошёл дальше батчей)
            st = MagicMock()
            for attr in dir(real_st):
                if not attr.startswith("_"):
                    try:
                        val = getattr(real_st, attr)
                        if callable(val):
                            setattr(st, attr, val)
                    except Exception:
                        pass
            st.purge_seen_titles = MagicMock(return_value=0)
            st.purge_items = MagicMock(return_value=0)

            cfg = self._make_config()
            cfg["digest"]["enabled"] = True
            cfg["llm"]["balance"] = {"enabled": True, "threshold": 1.0,
                                     "check_interval_hours": 24}

            # пауза активируется после 2-го вызова _is_paused (mid-batch)
            call_count = [0]

            def meta_side(key):
                if key == "paused":
                    call_count[0] += 1
                    return "1" if call_count[0] > 2 else None
                return real_st.get_meta(key)

            st.get_meta = meta_side
            st.get_all_by_state = real_st.get_all_by_state
            st.get_stuck_pending = real_st.get_stuck_pending

            delivery = MagicMock()
            delivery.send_batch.return_value = ([], [])
            delivery.send_text.return_value = True

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)

            # fake LLM-filter: после 2-го evaluate пауза активируется
            fake_filter = MagicMock()
            fake_filter.evaluate.return_value = FilterResult(
                ok=True, score=7, summary="s", reason="r"
            )
            fake_filter.push_threshold = 7
            fake_filter.digest_threshold = 5
            fake_filter.push_urgent_only = False  # legacy-маршрутизация
            fake_balance = MagicMock()
            fake_balance.is_available = True
            fake_balance.total_balance = 10.0
            fake_balance.currency = "USD"
            fake_filter.get_balance.return_value = fake_balance
            fake_filter._DEDUP_CHUNK_SIZE = 40  # для budget-чека в _run_llm_dedup
            pipeline.llm_filter = fake_filter
            pipeline.storage = st

            items = [
                NewsItem(source="a", url="http://z/1", title="t1",
                         summary="s1", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://z/2", title="t2",
                         summary="s2", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://z/3", title="t3",
                         summary="s3", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
            ]
            real_st.save_many(items)

            # спай на _maybe_run_digest / _maybe_check_balance
            with patch.object(pipeline, "_maybe_run_digest") as mock_digest:
                with patch.object(pipeline, "_maybe_check_balance") as mock_balance:
                    empty_result = MagicMock()
                    empty_result.ok = True
                    empty_result.items = []
                    with patch("newsbot.pipeline.FeedCollector") as fc_cls:
                        fc_cls.return_value.collect.return_value = iter([("test", empty_result)])
                        fc_cls.return_value.close = MagicMock()
                        pipeline.run_cycle_with_delivery()

                    # Major #1: _maybe_run_digest НЕ вызван (mid-batch пауза)
                    mock_digest.assert_not_called()
                    # Major #1: _maybe_check_balance НЕ вызван (mid-batch пауза)
                    mock_balance.assert_not_called()

            # delivery заблокирована mid-batch паузой
            delivery.send_batch.assert_not_called()
            # purge вызван (независимо от паузы)
            self.assertTrue(st.purge_seen_titles.called, "purge_seen_titles должен быть вызван")
            self.assertTrue(st.purge_items.called, "purge_items должен быть вызван")

            pipeline.storage = MagicMock()
        finally:
            real_st.close()
            _cleanup_db(path)

    def _make_config(self):
        # NIT: sources=[] чтобы не дёргать реальную сеть через FeedCollector.collect()
        return {
            "sources": [],
            "llm": {"enabled": False},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": True},
            "database": {"path": "data/newsbot.db"},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Mid-batch пауза
# ══════════════════════════════════════════════════════════════════════════════

class TestMidBatchPause(unittest.TestCase):

    def test_paused_with_no_llm_skips_delivery_items_remain_pending(self):
        """
        Major #1: при llm_filter=None пауза должна блокировать доставку.
        Сценарий: /stop пришёл во время сбора RSS — батч формируется,
        но send_batch не вызывается; items остаются pending (at-least-once).
        """
        from newsbot.pipeline import NewsPipeline
        from newsbot.collector import NewsItem
        from datetime import datetime, timezone

        path = _tmp_db()
        try:
            cfg = self._make_config_no_llm()
            delivery = MagicMock()
            delivery.send_batch.return_value = ([], [])

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)

            from newsbot.storage import Storage
            real_storage = Storage(path)
            # ставим паузу ДО вызова _process_batch
            real_storage.set_meta("paused", "1")
            pipeline.storage = real_storage
            # llm_filter=None (создан как None через патч)

            items = [
                NewsItem(source="a", url="http://x/1", title="t1",
                         summary="s1", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://x/2", title="t2",
                         summary="s2", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://x/3", title="t3",
                         summary="s3", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
            ]

            # сохраняем items в БД (имитируем предварительный save_many из run_cycle)
            real_storage.save_many(items)

            result = pipeline._process_batch(items, budget_override=100)

            # delivery НЕ должна быть вызвана (пауза заблокировала доставку)
            delivery.send_batch.assert_not_called()
            # возвращаем пустой remainder → retry_count не трогается
            self.assertEqual(result, [])

            # Minor #3: все items остались pending (не доставлены, нет двойной доставки).
            # Проверяем напрямую в БД: set_state вызывается ДО гейта паузы,
            # поэтому items переходят в pending.
            import sqlite3
            conn = sqlite3.connect(path)
            cur = conn.execute(
                "SELECT url, state FROM items WHERE url LIKE 'http://x/%' ORDER BY url"
            )
            rows = list(cur)
            conn.close()
            self.assertEqual(len(rows), 3)
            for url, state in rows:
                self.assertEqual(state, "pending",
                                 f"{url} должен остаться pending при паузе, state={state}")

            pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def _make_config_no_llm(self):
        return {
            "sources": [],
            "llm": {"enabled": False},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": False},
            "database": {"path": "data/newsbot.db"},
        }

    def test_mid_batch_pause_stops_llm_loop(self):
        """
        Minor #1: mid-batch пауза прерывает LLM-цикл (между батчами).
        При batch_size=1: 2 батча обработаны (items 1-2), 3-й не оценивается
        из-за паузы. Delivery заблокирована.

        Семантика батча: _is_paused() проверяется перед каждым батчем.
        При паузе следующий батч не стартует → pending на retry.
        """
        from newsbot.pipeline import NewsPipeline
        from newsbot.collector import NewsItem
        from newsbot.llm.filter import FilterResult
        from datetime import datetime, timezone

        path = _tmp_db()
        try:
            cfg = self._make_config()
            delivery = MagicMock()
            delivery.send_batch.return_value = ([], [])

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)

            from newsbot.storage import Storage
            real_storage = Storage(path)
            real_storage.set_meta("paused", "1")
            pipeline.storage = real_storage

            # batch_size=1: пауза срабатывает перед 3-м батчем
            pipeline.config.setdefault("llm", {})["batch_size"] = 1

            # Счётчик: _is_paused возвращает True (пауза) на 3-м вызове
            call_count = [0]

            def meta_side(key):
                if key == "paused":
                    call_count[0] += 1
                    return "1" if call_count[0] > 2 else None
                return None

            pipeline.storage.get_meta = meta_side

            fake_filter = MagicMock()

            def batch_side(items, feedback_context=None):
                return [FilterResult(ok=True, score=7, summary="s", reason="r")
                        for _ in items]
            fake_filter.evaluate_batch.side_effect = batch_side
            fake_filter.evaluate.return_value = FilterResult(
                ok=True, score=7, summary="s", reason="r"
            )
            fake_filter.push_threshold = 7
            fake_filter.digest_threshold = 5
            fake_filter.push_urgent_only = False
            pipeline.llm_filter = fake_filter

            items = [
                NewsItem(source="a", url="http://x/1", title="t1",
                         summary="s1", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://x/2", title="t2",
                         summary="s2", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
                NewsItem(source="a", url="http://x/3", title="t3",
                         summary="s3", published="—",
                         published_parsed=None, fetched_at=datetime.now(timezone.utc)),
            ]

            real_storage.save_many(items)

            result = pipeline._process_batch(items, budget_override=100)

            # При batch_size=1: evaluate_batch вызывается ровно 2 раза
            # (2 батча обработаны, 3-й остановлен паузой перед стартом)
            self.assertEqual(
                fake_filter.evaluate_batch.call_count, 2,
                f"evaluate_batch должен быть вызван ровно 2 раза, "
                f"got {fake_filter.evaluate_batch.call_count}"
            )

            # delivery заблокирована паузой
            delivery.send_batch.assert_not_called()
            self.assertEqual(result, [])

            # все items остались pending (не доставлены, нет двойной доставки)
            import sqlite3
            conn = sqlite3.connect(path)
            cur = conn.execute(
                "SELECT url, state FROM items WHERE url LIKE 'http://x/%' ORDER BY url"
            )
            rows = list(cur)
            conn.close()
            self.assertEqual(len(rows), 3)
            for url, state in rows:
                self.assertEqual(
                    state, "pending",
                    f"{url} должен остаться pending при паузе, state={state}"
                )

            pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def _make_config(self):
        return {
            "sources": [],
            "llm": {"enabled": True, "push_threshold": 7, "digest_threshold": 5},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": False},
            "database": {"path": "data/newsbot.db"},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Пауза приоритетнее off-peak
# ══════════════════════════════════════════════════════════════════════════════

class TestPausePriorityOverOffpeak(unittest.TestCase):

    def test_paused_takes_precedence_over_offpeak(self):
        """Пауза проверяется ДО off-peak гейта."""
        from newsbot.storage import Storage
        from newsbot.pipeline import NewsPipeline
        path = _tmp_db()
        try:
            st = Storage(path)
            st.set_meta("paused", "1")
            cfg = self._make_config()
            cfg["llm"]["off_peak_only"] = True
            delivery = MagicMock()

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=delivery):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.run_cycle_with_delivery()

            # доставка не вызвана (до неё не дошли)
            delivery.send_batch.assert_not_called()
            delivery.send_text.assert_not_called()
            pipeline.storage = MagicMock()
        finally:
            _cleanup_db(path)

    def _make_config(self):
        return {
            "sources": [],
            "llm": {
                "enabled": False,
                "off_peak_only": False,
                "balance": {"enabled": False},
            },
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": True},
            "database": {"path": "data/newsbot.db"},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8. Поллер без feedback.enabled
# ══════════════════════════════════════════════════════════════════════════════

class TestPollerWithoutFeedback(unittest.TestCase):

    def test_callback_ignored_when_feedback_disabled(self):
        """feedback_enabled=False: callback_query не обрабатывается."""
        storage = MagicMock()
        poller = _build_poller(
            storage_mock=storage,
            feedback_enabled=False,
            chat_id="owner",
        )

        upd = {
            "update_id": 60,
            "callback_query": {
                "id": "cb_test",
                "data": "fb:42:u",
                "message": {
                    "chat": {"id": "owner"},
                    "message_id": 100,
                },
            },
        }
        poller._process_update(upd)

        # record_feedback НЕ вызван (feedback выключен)
        storage.record_feedback.assert_not_called()
        # answerCallbackQuery отправлен (пользователь видит реакцию)
        poller._client.answerCallbackQuery.assert_called_once()

    def test_commands_work_without_feedback(self):
        """Без feedback.enabled команды /stop|/continue работают."""
        storage = MagicMock()
        poller = _build_poller(
            storage_mock=storage,
            feedback_enabled=False,
            chat_id="owner",
        )

        upd = {"update_id": 61, "message": _make_msg("/stop", "owner")}
        poller._process_update(upd)
        storage.set_meta.assert_called_with("paused", "1")


# ══════════════════════════════════════════════════════════════════════════════
# 9. _Commands.KNOWN содержит новые команды
# ══════════════════════════════════════════════════════════════════════════════

class TestCommandsKnown(unittest.TestCase):

    def test_stop_in_known_commands(self):
        self.assertIn("/stop", _Commands.KNOWN)

    def test_continue_in_known_commands(self):
        self.assertIn("/continue", _Commands.KNOWN)

    def test_stop_from_owner_only(self):
        msg = _make_msg("/stop", "111222333")
        self.assertTrue(_Commands.is_owner_command(msg, "111222333"))

    def test_stop_from_other_chat_ignored(self):
        msg = _make_msg("/stop", "999888777")
        self.assertFalse(_Commands.is_owner_command(msg, "111222333"))

    def test_continue_from_owner_only(self):
        msg = _make_msg("/continue", "111222333")
        self.assertTrue(_Commands.is_owner_command(msg, "111222333"))

    def test_continue_from_other_chat_ignored(self):
        msg = _make_msg("/continue", "999888777")
        self.assertFalse(_Commands.is_owner_command(msg, "111222333"))


if __name__ == "__main__":
    unittest.main()