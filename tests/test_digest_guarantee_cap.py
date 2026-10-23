"""
P2-регресс: гарантированные категории не отсекаются location-cap'ом.

Сценарий: max_per_location=2, две топ-новости 'россия|…' в deferred,
cinema-новость с той же (локация, день) в pool. До фикса _category_guarantee
дописывала cinema в КОНЕЦ списка, а _cap_by_location шёл по порядку →
cinema отбрасывалась, min=1 не выполнялся.

Фикс: cap учитывает guarantee-items первыми (protected_urls).
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.pipeline import NewsPipeline, _cap_by_location
from newsbot.storage import Storage


@contextmanager
def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _insert_deferred(db_path, source, url, title, cluster_key, score):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at,
         llm_score, cluster_key)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?, ?)
        """,
        (source, url, title, "Test summary", "2026-10-20 12:00",
         "2026-10-20T12:00:00+00:00", "2026-10-20T12:00:00+00:00",
         score, cluster_key),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, digest_cfg):
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": 15,
        "digest": digest_cfg,
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    # Закрываем первое соединение перед подменой
    p.storage.close()
    p.storage = Storage(db_path)
    return p


class TestGuaranteeSurvivesLocationCap(unittest.TestCase):
    def test_guaranteed_cinema_not_dropped_by_cap(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # Топ-3 (deferred, 'россия', один день): other-источники (tass).
            _insert_deferred(db_path, "tass", "https://ex.com/o8", "Other8",
                             "россия|a|2026-10-20", 8)
            _insert_deferred(db_path, "tass", "https://ex.com/o7", "Other7",
                             "россия|b|2026-10-20", 7)
            _insert_deferred(db_path, "tass", "https://ex.com/o5", "Other5",
                             "россия|c|2026-10-20", 5)
            # 4-й по score → попадает только в pool гарантии: cinema, та же локация/день.
            _insert_deferred(db_path, "deadline", "https://ex.com/cinema", "Cinema",
                             "россия|d|2026-10-20", 6)

            p = _mk_pipeline(db_path, {
                "enabled": True,
                "max_items": 3,
                "max_items_per_message": 20,
                "max_per_location": 2,
                "category_min": {"cinema": 1},
            })
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/cinema", digested,
                "P2: гарантированная cinema-новость не должна отсекаться cap'ом",
            )
            sent_msg = delivery.send_text.call_args[0][0]
            self.assertIn("Cinema", sent_msg)

            p.storage.close()
            p.close()


class TestCapByLocationProtected(unittest.TestCase):
    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_protected_items_win_over_cap_first_come(self):
        """Protected обрабатываются первыми и не отсекаются."""
        items = [
            self.FakeItem("u1", "россия|a|2026-10-20"),
            self.FakeItem("u2", "россия|b|2026-10-20"),
            self.FakeItem("u3", "россия|c|2026-10-20"),  # protected, в конце
        ]
        result = _cap_by_location(
            items, max_per_location=2, protected_urls={"u3"},
        )
        self.assertEqual([it.url for it in result], ["u3", "u1"])

    def test_without_protection_behaves_as_before(self):
        items = [
            self.FakeItem("u1", "россия|a|2026-10-20"),
            self.FakeItem("u2", "россия|b|2026-10-20"),
            self.FakeItem("u3", "россия|c|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual([it.url for it in result], ["u1", "u2"])

    def test_protected_beyond_cap_truncated_by_cap(self):
        """Minor 2 (wave 2): защита не может превышать cap — лишние отсечены.

        Иначе при sum(category_min) ≈ max_items protected-ы покрывали почти
        весь дайджест и max_per_location молча переставал работать.
        """
        items = [
            self.FakeItem("u1", "россия|a|2026-10-20"),
            self.FakeItem("u2", "россия|b|2026-10-20"),
            self.FakeItem("u3", "россия|c|2026-10-20"),
        ]
        with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
            result = _cap_by_location(
                items, max_per_location=1, protected_urls={"u2", "u3"},
            )
        self.assertEqual([it.url for it in result], ["u2"])
        self.assertTrue(
            any("protected items" in m for m in cm.output),
            f"ожидался warning о превышении защиты: {cm.output}",
        )

    def test_protection_quota_is_per_group(self):
        """Квота защиты считается на (локация, день), а не глобально."""
        items = [
            self.FakeItem("u1", "россия|a|2026-10-20"),
            self.FakeItem("u2", "россия|b|2026-10-20"),
            self.FakeItem("u3", "сша|c|2026-10-20"),
            self.FakeItem("u4", "сша|d|2026-10-20"),
        ]
        result = _cap_by_location(
            items, max_per_location=2, protected_urls={"u1", "u2", "u3", "u4"},
        )
        # ни одна группа не превышает cap=2, отсекать нечего
        self.assertEqual(
            sorted(it.url for it in result), ["u1", "u2", "u3", "u4"],
        )


class TestStringCategoryMinDoesNotCrash(unittest.TestCase):
    """Minor 1 (wave 2): category_min из YAML строкой не роняет дайджест."""

    def test_string_category_min_digest_succeeds(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred(db_path, "tass", "https://ex.com/o8", "Other8",
                             "россия|a|2026-10-20", 8)
            _insert_deferred(db_path, "deadline", "https://ex.com/cinema",
                             "Cinema", "россия|d|2026-10-20", 6)

            p = _mk_pipeline(db_path, {
                "enabled": True,
                "max_items": 3,
                "max_items_per_message": 20,
                "max_per_location": 2,
                # строки вместо int — раньше: TypeError вне try run_digest
                "category_min": {"cinema": "1"},
            })
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)
            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn("https://ex.com/cinema", digested)
            p.storage.close()
            p.close()


class TestProtectionQuotaInDigest(unittest.TestCase):
    """Minor 2 (wave 2): гарантия трёх категорий в одной (локация, день)
    не позволяет защите обойти max_per_location."""

    def test_protection_cannot_exceed_location_cap(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # три «гарантированных» категории в одной локации/дне
            _insert_deferred(db_path, "deadline", "https://ex.com/c1",
                             "Cinema1", "россия|a|2026-10-20", 9)
            _insert_deferred(db_path, "sciencedaily", "https://ex.com/s1",
                             "Science1", "россия|b|2026-10-20", 8)
            _insert_deferred(db_path, "ign", "https://ex.com/g1",
                             "Games1", "россия|c|2026-10-20", 7)

            p = _mk_pipeline(db_path, {
                "enabled": True,
                "max_items": 3,
                "max_items_per_message": 20,
                "max_per_location": 2,
                "category_min": {"cinema": 1, "science": 1, "games": 1},
            })
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
                count = p.run_digest(force=True)

            self.assertTrue(
                any("превышает location-cap" in m for m in cm.output),
                f"ожидался warning о квоте защиты: {cm.output}",
            )
            self.assertEqual(count, 2, "cap=2 на (локация, день) должен удержаться")
            sent_msg = delivery.send_text.call_args[0][0]
            # cinema (приоритетнее) и science остались, games — нет
            self.assertIn("Cinema1", sent_msg)
            self.assertIn("Science1", sent_msg)
            self.assertNotIn("Games1", sent_msg)
            p.storage.close()
            p.close()


class TestMaxPerLocationInvalidValues(unittest.TestCase):
    """Major/Minor (wave 4): NaN/inf/пустое значение max_per_location.

    Было: `int(float('nan'))` → ValueError, `int(float('inf'))` →
    OverflowError ВНЕ try вокруг _run_digest (`.nan`/`.inf` — валидный
    YAML) → дайджест падал каждый цикл. Плюс `if raw_max else 0` молча
    ОТКЛЮЧАЛ cap на null/"".
    Теперь: NaN/inf/null/"" → documented default 2 + warning; явный 0 —
    по-прежнему осознанное «без cap».
    """

    @staticmethod
    def _seed_same_group(db_path, n=3):
        for i in range(n):
            _insert_deferred(db_path, "tass", f"https://ex.com/g{i}",
                             f"Same{i}", f"россия|a{i}|2026-10-20", 9 - i)

    def _run(self, digest_cfg):
        import logging

        records = []

        class _H(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _H()
        pipe_logger = logging.getLogger("newsbot.pipeline")
        pipe_logger.addHandler(handler)
        try:
            with _tmp_db() as db_path:
                s = Storage(db_path)
                s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
                s.close()
                self._seed_same_group(db_path)
                cfg = {
                    "enabled": True,
                    "max_items": 10,
                    "max_items_per_message": 20,
                }
                cfg.update(digest_cfg)
                p = _mk_pipeline(db_path, cfg)
                delivery = MagicMock()
                delivery.send_text.return_value = True
                p.delivery = delivery
                count = p.run_digest(force=True)
                self.assertIn(p.storage.get_meta("digest_failure_count"), (None, "0"))
                p.storage.close()
                p.close()
        finally:
            pipe_logger.removeHandler(handler)
        return count, records

    def test_nan_falls_back_to_default_cap(self):
        count, logs = self._run({"max_per_location": float("nan")})
        self.assertEqual(count, 2, "NaN → default 2 (cap работает), не падение")
        self.assertTrue(
            any("не конечное число" in m for m in logs),
            f"ожидался warning о не конечном числе: {logs}",
        )

    def test_positive_inf_falls_back_to_default_cap(self):
        count, logs = self._run({"max_per_location": float("inf")})
        self.assertEqual(count, 2, "inf → default 2, а не OverflowError")
        self.assertTrue(any("не конечное число" in m for m in logs), logs)

    def test_none_does_not_silently_disable_cap(self):
        count, logs = self._run({"max_per_location": None})
        self.assertEqual(count, 2, "null → default 2, cap НЕ выключается молча")
        self.assertTrue(any("пустое значение" in m for m in logs), logs)

    def test_empty_string_does_not_silently_disable_cap(self):
        count, logs = self._run({"max_per_location": ""})
        self.assertEqual(count, 2, "'' → default 2, cap НЕ выключается молча")
        self.assertTrue(any("пустое значение" in m for m in logs), logs)

    def test_explicit_zero_still_disables_cap(self):
        count, _logs = self._run({"max_per_location": 0})
        self.assertEqual(count, 3, "явный 0 — осознанное «без cap»")

    def test_float_string_is_accepted(self):
        """Nit (wave 4): "2.0" раньше отвергалась (int("2.0") → ValueError)."""
        count, _logs = self._run({"max_per_location": "2.0"})
        self.assertEqual(count, 2, '"2.0" должен читаться как 2')


class TestLocDayKeyMatchesCap(unittest.TestCase):
    """Nit (wave 4): _loc_day_key группирует ровно как _cap_by_location.

    Раньше cluster_key из двух частей давал в _loc_day_key None («cap не
    применяется»), а cap резал его по ('loc','unknown') — разные мнения
    об одном item. Теперь обе стороны дают ('loc','unknown').
    """

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_truncated_cluster_key_uses_unknown_day(self):
        from newsbot.pipeline import _loc_day_key
        self.assertEqual(
            _loc_day_key(self.FakeItem("u1", "россия|a")),
            ("россия", "unknown"),
        )
        # и cap режет ровно эту группу
        items = [self.FakeItem(f"u{i}", "россия|a") for i in range(3)]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual([it.url for it in result], ["u0", "u1"])

    def test_no_key_and_other_bucket_still_none(self):
        from newsbot.pipeline import _loc_day_key
        self.assertIsNone(_loc_day_key(self.FakeItem("u1", None)))
        self.assertIsNone(_loc_day_key(self.FakeItem("u2", "")))
        self.assertIsNone(_loc_day_key(self.FakeItem("u3", "other|a|2026-10-20")))

    def test_protection_quota_counts_truncated_key_group(self):
        """Защита по truncated-key item'ам считается в квоту группы, а не
        как «бесплатная» — иначе cap молча переставал работать (backstop
        отсекал protected уже после fact)."""
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # оба ключа без дня → одна группа ('россия','unknown')
            _insert_deferred(db_path, "deadline", "https://ex.com/c1",
                             "Cinema1", "россия|a", 9)
            _insert_deferred(db_path, "sciencedaily", "https://ex.com/s1",
                             "Science1", "россия|b", 8)

            p = _mk_pipeline(db_path, {
                "enabled": True,
                "max_items": 5,
                "max_items_per_message": 20,
                "max_per_location": 1,
                "category_min": {"cinema": 1, "science": 1},
            })
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
                count = p.run_digest(force=True)

            self.assertEqual(count, 1, "cap=1 на ('россия','unknown') держится")
            self.assertTrue(
                any("превышает location-cap=1 для локации россия/unknown" in m
                    for m in cm.output),
                f"ожидался warning о квоте защиты по truncated-ключу: {cm.output}",
            )
            self.assertFalse(
                any("отсечено" in m for m in cm.output),
                f"cap не должен отсекать защиту задним числом: {cm.output}",
            )
            sent_msg = delivery.send_text.call_args[0][0]
            self.assertIn("Cinema1", sent_msg)
            p.storage.close()
            p.close()


if __name__ == "__main__":
    unittest.main()
