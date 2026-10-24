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
from contextlib import contextmanager
from datetime import datetime, timezone as tz, timedelta
from unittest.mock import MagicMock, patch

from newsbot.pipeline import NewsPipeline


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@contextmanager
def _patch_dt_now(fake_utc: datetime):
    """
    Патчит newsbot.pipeline.datetime и _local_now() так, чтобы pipeline
    видел детерминированное local time в UTC+3, согласованное с fake_utc.

    fake_utc — aware UTC datetime. _local_now() возвращает aware UTC+3.

    Проблема: _local_now() (pipeline.py:118) делает
    datetime.now(tz.utc).astimezone() → astimezone на aware-объекте
    конвертирует в TZ ХОСТА. Без патча _local_now тесты с fake_now_utc=06:00
    (задумано как 09:00 MSK) упадут на хосте с TZ≠MSK.
    """
    utc3 = tz(timedelta(hours=3))
    fake_local = fake_utc.astimezone(utc3)  # aware UTC+3

    def fake_local_now():
        return fake_local

    real_dt = datetime
    fake_cls = MagicMock()
    fake_cls.now.return_value = fake_utc
    fake_cls.side_effect = lambda *a, **kw: real_dt(*a, **kw)
    fake_cls.timedelta = timedelta
    fake_cls.replace = real_dt.replace
    fake_cls.fromisoformat = real_dt.fromisoformat
    with patch('newsbot.pipeline.datetime', fake_cls), \
         patch('newsbot.pipeline._local_now', fake_local_now):
        yield


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

                        with _patch_dt_now(fake_now_utc), \
                             patch.object(pipeline, "run_digest") as mock_digest:
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


class TestDigestTimesValidation(unittest.TestCase):
    """
    digest.times не валидировался → слоты молча терялись или дайджест
    вообще не запускался (TypeError вне try вокруг _run_digest).
    """

    def _assert_digest_called(self, times, fake_now_utc, called,
                              last_digest_iso=None):
        import logging

        records: list = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=logging.WARNING)
        pipe_logger = logging.getLogger("newsbot.pipeline")
        pipe_logger.addHandler(handler)
        path = _tmp_db()
        try:
            from newsbot.storage import Storage
            st = Storage(path)
            if last_digest_iso:
                st.set_meta("last_digest_at", last_digest_iso)
            cfg = {
                "sources": [],
                "llm": {"enabled": False},
                "telegram": {"token": "", "chat_id": ""},
                "feedback": {"enabled": False},
                "digest": {"enabled": True, "times": times},
                "database": {"path": "data/newsbot.db"},
            }
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=None):
                with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
                    with patch.object(NewsPipeline, "_resolve_db_path", return_value=path):
                        pipeline = NewsPipeline(cfg)
                        pipeline.storage = st
                        pipeline.llm_filter = MagicMock()

                        with _patch_dt_now(fake_now_utc):
                            with patch.object(pipeline, "run_digest") as mock_digest:
                                pipeline._maybe_run_digest()
                        pipeline.storage = MagicMock()
                        if called:
                            mock_digest.assert_called_once_with(force=False)
                        else:
                            mock_digest.assert_not_called()
                        return "\n".join(records)
        finally:
            pipe_logger.removeHandler(handler)
            st.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_hour_without_leading_zero_keeps_both_slots(self):
        """
        (а) ["9:00", "21:00"]: раньше int("9:") → ValueError → слот молча
        выключен (1 дайджест в сутки). Теперь «9:00» нормализуется в
        «09:00» с warning, слот работает.
        """
        # now=09:30 MSK (06:30 UTC) — слот 09:00 сегодня уже прошёл
        out = self._assert_digest_called(
            ["9:00", "21:00"], datetime(2026, 10, 16, 6, 30, tzinfo=tz.utc),
            called=True,
        )
        self.assertIn("нормализовано", out)

        # второй слот тоже живой: last=09:05, now=22:00 MSK (19:00 UTC) →
        # окно 21:00 не обработано → дайджест уходит
        self._assert_digest_called(
            ["9:00", "21:00"], datetime(2026, 10, 16, 19, 0, tzinfo=tz.utc),
            called=True, last_digest_iso="2026-10-16T09:05:00+03:00",
        )

    def test_yaml_sexagesimal_int_times_falls_back_to_default(self):
        """
        (б) times: [9:00] без кавычек → YAML sexagesimal → int 540.
        Раньше: `540[:2]` → TypeError ВНЕ перехвата → _maybe_run_digest
        падал каждый цикл, дайджест не запускался вообще.
        Теперь: warning + дефолт 08:00/20:00 → дайджест работает.
        """
        # now=09:00 MSK — дефолтный слот 08:00 прошёл → дайджест уходит
        out = self._assert_digest_called(
            [540], datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc),
            called=True,
        )
        self.assertIn("540", out)
        self.assertIn("дефолт", out)

    def test_times_as_string_falls_back_to_default(self):
        """
        (в) times: "08:00" строкой — раньше итерация по символам
        ('0','8',':','0','0') давала мусорные слоты/TypeError.
        Теперь: warning + дефолт.
        """
        out = self._assert_digest_called(
            "08:00", datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc),
            called=True,
        )
        self.assertIn("ожидался список", out)

    def test_times_as_dict_falls_back_to_default(self):
        """Не-список (dict) → warning + дефолт, без исключения."""
        out = self._assert_digest_called(
            {"first": "08:00"}, datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc),
            called=True,
        )
        self.assertIn("ожидался список", out)

    def test_invalid_string_slots_skipped_and_digest_still_runs(self):
        """
        Один валидный + один битый слот: битый отфильтрован с warning,
        валидный работает (дайджест не выключен целиком).
        """
        out = self._assert_digest_called(
            ["08:00", "25:00"],
            datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc),
            called=True,
        )
        self.assertIn("невалидное время", out)

    def test_empty_times_list_disables_schedule(self):
        """Явный пустой список = расписание выключено (без падения)."""
        self._assert_digest_called(
            [], datetime(2026, 10, 16, 6, 0, tzinfo=tz.utc), called=False,
        )


class TestNormalizeDigestTimes(unittest.TestCase):
    """Юнит-тесты нормализатора digest.times."""

    def test_default_slots_match_config(self):
        from newsbot.config import DEFAULT_CONFIG
        from newsbot.pipeline import _normalize_digest_times

        expected = [
            (int(t[:2]), int(t[3:])) for t in DEFAULT_CONFIG["digest"]["times"]
        ]
        self.assertEqual(_normalize_digest_times(None), expected)
        self.assertEqual(_normalize_digest_times([540]), expected)

    def test_valid_and_normalized_slots(self):
        from newsbot.pipeline import _normalize_digest_times

        self.assertEqual(
            _normalize_digest_times(["08:00", "21:00"]), [(8, 0), (21, 0)]
        )
        # «9:00» / пробелы — осмысленная нормализация (с warning)
        self.assertEqual(
            _normalize_digest_times(["9:00", " 21:05 "]), [(9, 0), (21, 5)]
        )

    def test_invalid_elements_skipped(self):
        from newsbot.pipeline import _normalize_digest_times

        slots = _normalize_digest_times(["08:00", 540, None, "nope", "25:00"])
        self.assertEqual(slots, [(8, 0)])

    def test_never_raises(self):
        from newsbot.pipeline import _normalize_digest_times

        for raw in ("08:00", 540, object(), {"a": 1}, [None], [b"08:00"]):
            _normalize_digest_times(raw)  # не должно бросить


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