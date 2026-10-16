"""
tests/test_digest_schedule.py — тесты логики выбора окна в _maybe_run_digest.

Проверяемые сценарии:
  (a) свежая БД (last_digest=None), время ДО первого сегодняшнего scheduled
      → дайджест НЕ отправляется.
  (b) свежая БД, после сегодняшнего scheduled → дайджест отправляется.
  (c) непустой last_digest, пропущено вчерашнее окно → catch-up отправляется.
  (d) непустой last_digest, сегодняшнее окно уже обработано → НЕ отправляется.
  (e) YAML enabled: 1 (int) корректно парсится как True.
"""
import os
import tempfile
import unittest
from datetime import datetime, timezone as tz, timedelta
from unittest.mock import MagicMock, patch

from newsbot.pipeline import NewsPipeline


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _patch_dt_now(fake_utc: datetime):
    """
    Патчит ссылку newsbot.pipeline.datetime — мокает только class-level,
    не builtin. Все методы экземпляра (.replace, .astimezone, .timestamp)
    работают с реальными объектами.
    """
    real_dt = datetime
    fake_cls = MagicMock()
    fake_cls.now.return_value = fake_utc
    fake_cls.side_effect = lambda *a, **kw: real_dt(*a, **kw)
    fake_cls.timedelta = timedelta
    fake_cls.replace = real_dt.replace
    fake_cls.fromisoformat = real_dt.fromisoformat
    return patch('newsbot.pipeline.datetime', fake_cls)


class TestDigestScheduleFreshDb(unittest.TestCase):
    """(a) + (b): свежая БД — только сегодняшние уже прошедшие окна."""

    def test_fresh_db_before_first_scheduled_no_digest(self):
        """
        Свежая БД (last_digest=None), now=07:00 MSK, times=[08:00, 20:00].
        → ни одно сегодняшнее окно ещё не наступило → дайджест НЕ отправляется.
        """
        path = _tmp_db()
        try:
            from newsbot.storage import Storage
            st = Storage(path)
            cfg = self._make_config()
            fake_now_utc = datetime(2026, 10, 16, 4, 0, tzinfo=tz.utc)  # 07:00 MSK

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()

                        with _patch_dt_now(fake_now_utc):
                            with patch.object(pipeline, "run_digest") as mock_digest:
                                pipeline._maybe_run_digest()
                                # 07:00 < 08:00 → дайджест НЕ вызван
                                mock_digest.assert_not_called()

                        pipeline.storage = MagicMock()
        finally:
            st.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_fresh_db_after_first_scheduled_triggers_digest(self):
        """
        Свежая БД, now=09:00 MSK, times=[08:00, 20:00]
        → сегодняшнее 08:00 прошло → дайджест ОТПРАВЛЯЕТСЯ (однократно).
        """
        path = _tmp_db()
        try:
            from newsbot.storage import Storage
            st = Storage(path)
            cfg = self._make_config()
            fake_now_utc = datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc)  # 09:00 MSK

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()

                        with _patch_dt_now(fake_now_utc):
                            with patch.object(pipeline, "run_digest") as mock_digest:
                                pipeline._maybe_run_digest()
                                # 09:00 > 08:00 → дайджест вызван ровно 1 раз
                                mock_digest.assert_called_once_with(force=False)

                        pipeline.storage = MagicMock()
        finally:
            st.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def _make_config(self):
        return {
            "sources": [],
            "llm": {"enabled": False},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": True, "times": ["08:00", "20:00"]},
            "database": {"path": "data/newsbot.db"},
        }


class TestDigestScheduleCatchUp(unittest.TestCase):
    """(c) + (d): непустой last_digest — catch-up пропущенных окон."""

    def test_last_digest_catches_up_missed_yesterday_window(self):
        """
        last_digest вчера (2026-10-15 21:05 MSK), now=2026-10-16 09:00 MSK,
        times=[08:00, 20:00].
        Вчерашнее 20:00 пропущено → catch-up уходит.
        """
        path = _tmp_db()
        try:
            from newsbot.storage import Storage
            st = Storage(path)
            st.set_meta("last_digest_at", "2026-10-15T21:05:00+03:00")
            cfg = self._make_config()
            fake_now_utc = datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc)  # 09:00 MSK

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()

                        with _patch_dt_now(fake_now_utc):
                            with patch.object(pipeline, "run_digest") as mock_digest:
                                pipeline._maybe_run_digest()
                                mock_digest.assert_called_once_with(force=False)

                        pipeline.storage = MagicMock()
        finally:
            st.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_last_digest_skips_already_processed_today_window(self):
        """
        last_digest сегодня 08:05 MSK, times=[08:00, 20:00], now=09:00 MSK.
        Сегодняшнее 08:00 уже обработано → дайджест НЕ отправляется.
        """
        path = _tmp_db()
        try:
            from newsbot.storage import Storage
            st = Storage(path)
            st.set_meta("last_digest_at", "2026-10-16T08:05:00+03:00")
            cfg = self._make_config()
            fake_now_utc = datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc)  # 09:00 MSK

            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()

                        with _patch_dt_now(fake_now_utc):
                            with patch.object(pipeline, "run_digest") as mock_digest:
                                pipeline._maybe_run_digest()
                                # сегодняшнее 08:00 уже обработано → НЕ шлём
                                mock_digest.assert_not_called()

                        pipeline.storage = MagicMock()
        finally:
            st.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def _make_config(self):
        return {
            "sources": [],
            "llm": {"enabled": False},
            "telegram": {"token": "", "chat_id": ""},
            "feedback": {"enabled": False},
            "digest": {"enabled": True, "times": ["08:00", "20:00"]},
            "database": {"path": "data/newsbot.db"},
        }


class TestParseBoolInt(unittest.TestCase):
    """(e): _parse_bool корректно обрабатывает int 1/0."""

    def test_parse_bool_int_1_true(self):
        from newsbot.pipeline import _parse_bool
        self.assertTrue(_parse_bool(1))

    def test_parse_bool_int_0_false(self):
        from newsbot.pipeline import _parse_bool
        self.assertFalse(_parse_bool(0))

    def test_parse_bool_bool_true_still_works(self):
        from newsbot.pipeline import _parse_bool
        self.assertTrue(_parse_bool(True))
        self.assertFalse(_parse_bool(False))


if __name__ == "__main__":
    unittest.main()