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


# =============================================================================
# F3-тесты: нормализация локации для cap'а
# =============================================================================

class TestNormalizeCapKey(unittest.TestCase):
    """F3-фикс: _normalize_cap_key схлопывает вариации локаций в одно ядро."""

    def _key(self, loc):
        from newsbot.pipeline import _normalize_cap_key
        return _normalize_cap_key(loc)

    def test_english_variations_collapse(self):
        """«Voronezh region» и «Voronezh» → одинаковый ключ."""
        self.assertEqual(self._key("Voronezh region"), self._key("Voronezh"))
        self.assertEqual(self._key("Voronezh region, Russia"), self._key("Voronezh"))
        self.assertEqual(self._key("Voronezh"), self._key("voronezh"))

    def test_comma_cut_keeps_core(self):
        """Обрезка по запятой: «City, Country» → «city» (generic-суффикс убирается)."""
        self.assertEqual(self._key("Voronezh, Russia"), "voronezh")
        self.assertEqual(self._key("Moscow, Russia"), "moscow")
        # "Krasnodar Krai" → comma-cut → "Krasnodar Krai" → strip "krai" → "krasnodar"
        self.assertEqual(self._key("Krasnodar Krai, Russia"), "krasnodar")

    def test_generic_suffixes_stripped(self):
        """English generic-суффиксы убираются."""
        self.assertEqual(self._key("Voronezh region"), "voronezh")
        self.assertEqual(self._key("Leningrad oblast"), "leningrad")
        self.assertEqual(self._key("Stavropol krai"), "stavropol")
        self.assertEqual(self._key("Primorsky district"), "primorsky")
        self.assertEqual(self._key("Catalonia province"), "catalonia")
        self.assertEqual(self._key("Bavaria area"), "bavaria area")  # area — НЕ generic-суффикс, "Bay Area" не схлопывается
        # Bay Area — независимый регион, не схлопывается
        self.assertNotEqual(self._key("Bay Area"), "bay")
        self.assertEqual(self._key("Bavaria region"), "bavaria")

    def test_russian_generic_suffixes_stripped(self):
        """Russian generic-суффиксы убираются."""
        self.assertEqual(self._key("Воронежская область"), "воронежская")
        self.assertEqual(self._key("Краснодарский край"), "краснодарский")
        self.assertEqual(self._key("Ростовская обл"), "ростовская")
        self.assertEqual(self._key("Татарстан регион"), "татарстан")
        self.assertEqual(self._key("Саратовский район"), "саратовский")

    def test_moscow_vs_moscow_oblast_not_collapsed(self):
        """Критический регрессионный: «Москва» ≠ «Московская область».

        «московская область» не схлопывается, т.к. «область» здесь —
        основное слово (без уточнителя), а не generic-суффикс.
        """
        # «Москва» → «москва»
        self.assertEqual(self._key("Москва"), "москва")
        # «Московская область» → comma-cut не срабатывает (нет запятой) →
        #   generic-суффикс «область» убирается → «московская»
        self.assertNotEqual(self._key("Московская область"), self._key("Москва"))

    def test_empty_and_none(self):
        """Пустая/None → ''."""
        self.assertEqual(self._key(""), "")
        self.assertEqual(self._key(None), "")
        self.assertEqual(self._key("   "), "")

    def test_comma_only_before_generic(self):
        """«Voronezh region, Russia» → comma-cut даёт «voronezh region»,
        затем убирается « region»."""
        self.assertEqual(self._key("Voronezh region, Russia"), "voronezh")


class TestCapByLocationNormalization(unittest.TestCase):
    """F3-фикс: cap схлопывает «Voronezh region» и «Voronezh» в одну группу."""

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_varied_location_variants_capped_together(self):
        """«Voronezh region» и «Voronezh» попадают в одну группу → cap=2 пропускает 2."""
        items = [
            self.FakeItem("u1", "voronezh region|a|2026-10-20"),
            self.FakeItem("u2", "Voronezh|b|2026-10-20"),
            self.FakeItem("u3", "voronezh, russia|c|2026-10-20"),
            self.FakeItem("u4", "Voronezh Region|d|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # Все 4 → одно нормализованное ядро «voronezh» → cap=2 → первые 2 проходят
        self.assertEqual(len(result), 2)
        self.assertEqual(
            sorted(it.url for it in result),
            ["u1", "u2"],
            "Первые 2 (по порядку) должны пройти, остальные — cap отсек",
        )

    def test_different_locations_separate(self):
        """Разные локации не схлопываются."""
        items = [
            self.FakeItem("u1", "voronezh region|a|2026-10-20"),
            self.FakeItem("u2", "moscow region|b|2026-10-20"),
            self.FakeItem("u3", "voronezh|c|2026-10-20"),
            self.FakeItem("u4", "krasnodar|d|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # voronezh (2 variant) + moscow + krasnodar = 4, cap=2 на группу
        # voronezh: 2 items (both pass cap=2)
        # moscow: 1 item (passes)
        # krasnodar: 1 item (passes)
        self.assertEqual(len(result), 4, "cap не схлопывает voronezh и moscow")

    def test_russian_location_variants_capped(self):
        """F3: русские вариации схлопываются через generic-stripping.

        «Воронежская область» → «воронежская» (убираем « область»)
        «Воронежская» → «воронежская» (без изменений)
        Оба имеют ключ «воронежская» → cap=2 пропускает оба.
        «Воронеж» → «воронеж» (отдельный ключ, cap=2 пропускает).
        Итого все 3 проходят: cap НЕ ограничивает «воронежскую»+«воронежскую»+
        «воронеж» в одно ведро, т.к. «воронежская»≠«воронеж».
        """
        items = [
            self.FakeItem("u1", "воронежская область|a|2026-10-20"),
            self.FakeItem("u2", "воронеж|b|2026-10-20"),
            self.FakeItem("u3", "воронежская|c|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # «воронежская» (u1,u3) в одном бакете — cap=2, оба проходят.
        # «воронеж» (u2) — отдельный бакет — cap=2, проходит.
        # Все 3 проходят.
        self.assertEqual(len(result), 3,
            "воронежская+воронежская (область) НЕ схлопываются с воронеж")

    def test_loc_day_key_uses_normalization(self):
        """_loc_day_key возвращает нормализованную локацию."""
        from newsbot.pipeline import _loc_day_key
        item1 = self.FakeItem("u1", "Voronezh region|a|2026-10-20")
        item2 = self.FakeItem("u2", "Voronezh|b|2026-10-20")
        key1 = _loc_day_key(item1)
        key2 = _loc_day_key(item2)
        self.assertEqual(key1, key2,
            "_loc_day_key должен схлопывать вариации")
        self.assertEqual(key1[0], "voronezh")


class TestCapNormalizationWarning(unittest.TestCase):
    """F3: warning при коллизиях нормализации."""

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_warning_on_collision(self):
        """При схлопывании >1 исходной строки в одно ядро — WARNING.

        Warning вызывается когда cap реально удаляет items (len(result)<len(items)).
        Используем 3 варианта локации → 2 нормализованных ядра:
        - 'voronezh region' и 'Voronezh' → нормализуются в 'voronezh'
        - 'krasnodar krai' → нормализуется в 'krasnodar'
        При cap=1: voronezh бакет получает 2 item'а → cap отсекает 1 →
        warning срабатывает (len(result)=2 < len(items)=3).
        """
        items = [
            self.FakeItem("u1", "Voronezh region|a|2026-10-20"),
            self.FakeItem("u2", "Voronezh|b|2026-10-20"),
            self.FakeItem("u3", "Krasnodar Krai|c|2026-10-20"),
        ]
        with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
            _cap_by_location(items, max_per_location=1)
        self.assertTrue(
            len(cm.output) >= 1,
            f"Expected at least 1 WARNING, got: {cm.output}",
        )

    def test_same_location_no_warning(self):
        """Одинаковые локации — схлопывания нет → warning НЕ вызывается."""
        import logging

        records = []

        class _H(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _H()
        pipe_logger = logging.getLogger("newsbot.pipeline")
        pipe_logger.addHandler(handler)
        try:
            items = [
                self.FakeItem("u1", "voronezh|a|2026-10-20"),
                self.FakeItem("u2", "voronezh|b|2026-10-20"),
                self.FakeItem("u3", "voronezh|c|2026-10-20"),
            ]
            # cap=1: убирает 2 из 3, но все из одной локации → нет вариантов → нет warning
            _cap_by_location(items, max_per_location=1)
            # Нет warning-ов о нормализации
            uniq_warns = [r for r in records if "вариантов" in r]
            self.assertEqual(uniq_warns, [], f"Не должно быть warning о вариантах: {uniq_warns}")
        finally:
            pipe_logger.removeHandler(handler)

    def test_variant_collapse_logs_warning_with_uniq_text(self):
        """«Voronezh region» + «Voronezh» → warning ЕСТЬ, 2 уникальных варианта."""
        import logging

        records = []

        class _H(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _H()
        pipe_logger = logging.getLogger("newsbot.pipeline")
        pipe_logger.addHandler(handler)
        try:
            items = [
                self.FakeItem("u1", "Voronezh region|a|2026-10-20"),
                self.FakeItem("u2", "Voronezh|b|2026-10-20"),
                self.FakeItem("u3", "krasnodar|c|2026-10-20"),
            ]
            _cap_by_location(items, max_per_location=1)
            uniq_warns = [r for r in records if "вариантов" in r]
            self.assertTrue(
                len(uniq_warns) >= 1,
                f"Expected warning о вариантах, got: {records}",
            )
            # текст содержит 2 варианта (не len(orig_locations)=2, а uniq count)
            self.assertIn(
                "2 вариантов", uniq_warns[0],
                f"Warning должен показать 2 варианта: {uniq_warns}",
            )
        finally:
            pipe_logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
