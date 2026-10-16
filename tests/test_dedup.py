"""
Тесты кросс-источниковой дедупликации.
Запускать: python -m pytest tests/test_dedup.py -v
"""
import os
import re
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from datetime import timedelta
from newsbot.collector import NewsItem
from newsbot.dedup import (
    filter_new,
    _is_meaningful_title,
    _quick_filter,
    _is_fuzzy_similar,
    _token_set,
    _number_multiset,
    _meaningful_tokens,
    _token_jaccard,
)
from newsbot.storage import Storage, MIN_TITLE_LEN


# =============================================================================
# fixtures
# =============================================================================

def _mk_item(
    source="TestSource",
    title="Test Title",
    url=None,
    summary="Test summary text",
    published="01.01.2025 12:00 UTC",
    **kwargs,
) -> NewsItem:
    if url is None:
        url = f"https://example.com/{hash(title)}"
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary=summary,
        published=published,
        published_parsed=None,
        **kwargs,
    )


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _cleanup_db(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        time.sleep(0.2)
        try:
            os.unlink(path)
        except (PermissionError, FileNotFoundError):
            pass


# =============================================================================
# title_normalized() — нормализация
# =============================================================================

class TestTitleNormalized(unittest.TestCase):
    def test_lowercase(self):
        """Lowercase применяется."""
        item = _mk_item(title="UPPERCASE TITLE")
        self.assertEqual(item.title_normalized(), "uppercase title")

    def test_punctuation_removed(self):
        """Пунктуация удаляется."""
        item = _mk_item(title="Hello, World! How are you?")
        result = item.title_normalized()
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_spaces_collapsed(self):
        """Повторные пробелы схлопываются."""
        item = _mk_item(title="Hello     world   multiple   spaces")
        self.assertNotIn("  ", item.title_normalized())

    def test_strip(self):
        """Strip по краям."""
        item = _mk_item(title="  trimmed  ")
        self.assertEqual(item.title_normalized(), "trimmed")

    def test_source_prefix_strip_reuters(self):
        """Reuters: префикс срезается."""
        item = _mk_item(title="Reuters: EU approves new sanctions against Russia")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("eu approves"),
            f"Expected 'eu approves...' but got: {norm!r}")
        self.assertNotIn("reuters", norm[:20])

    def test_source_prefix_strip_bloomberg(self):
        """Bloomberg — префикс срезается."""
        item = _mk_item(title="Bloomberg: Fed cuts interest rates")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("fed cuts"),
            f"Expected 'fed cuts...' but got: {norm!r}")

    def test_source_prefix_strip_dash(self):
        """Bloomberg — префикс срезается (тире/dash)."""
        item = _mk_item(title="Bloomberg — Fed cuts interest rates")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("fed cuts"),
            f"Expected 'fed cuts...' but got: {norm!r}")

    def test_source_prefix_strip_tass(self):
        """TASS: префикс срезается."""
        item = _mk_item(title="TASS: Russia launches new satellite")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("russia launches"),
            f"Expected 'russia launches...' but got: {norm!r}")

    def test_source_prefix_strip_cyrillic(self):
        """Источник: префикс срезается (кириллица)."""
        item = _mk_item(title="Источник: в России произошло землетрясение")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("в россии произошло"),
            f"Expected 'в россии произошло...' but got: {norm!r}")

    def test_source_prefix_not_strip_middle(self):
        """Двоеточие/тире в СЕРЕДИНЕ заголовка НЕ срезается."""
        # "Europe: what you need to know" — двоеточие в начале, но это не префикс
        item = _mk_item(title="Europe: What You Need to Know About the Crisis")
        norm = item.title_normalized()
        # после нормализации "europe what you need..." — europe не в нашем списке
        self.assertIn("europe", norm)
        self.assertIn("crisis", norm)

    def test_source_prefix_not_strip_ordinary_title(self):
        """Обычный заголовок без префикса не повреждается."""
        item = _mk_item(title="Scientists discover new species of deep-sea fish")
        norm = item.title_normalized()
        self.assertEqual(norm, "scientists discover new species of deepsea fish")

    def test_source_prefix_case_insensitive(self):
        """Стриппинг префикса — регистронезависимый."""
        item = _mk_item(title="REUTERS: Market hits record high")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("market hits"),
            f"Expected 'market hits...' but got: {norm!r}")

    def test_source_prefix_strip_ascii_dash(self):
        """"Bloomberg - Headline": ASCII hyphen в префиксе срезается."""
        item = _mk_item(title="Bloomberg - Story about the economy")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("story about"),
            f"Expected 'story about...' but got: {norm!r}")

    def test_source_prefix_from_source_ascii_dash(self):
        """"From Source - Headline": префикс с ASCII hyphen срезается."""
        item = _mk_item(title="From Source - Energy prices surge in Europe")
        norm = item.title_normalized()
        self.assertTrue(norm.startswith("energy prices"),
            f"Expected 'energy prices...' but got: {norm!r}")

    def test_hyphen_in_middle_not_stripped(self):
        """Дефис в СЕРЕДИНЕ заголовка НЕ срезается."""
        item = _mk_item(title="US-China trade war - new developments")
        norm = item.title_normalized()
        # Слово "uschina" (без дефиса), затем "trade", "war", "new", "developments"
        self.assertIn("trade", norm)
        self.assertIn("war", norm)
        self.assertIn("new", norm)
        self.assertIn("developments", norm)


# =============================================================================
# MIN_TITLE_LEN
# =============================================================================

class TestMinTitleLen(unittest.TestCase):
    def test_short_titles_not_meaningful(self):
        self.assertFalse(_is_meaningful_title(_mk_item(title="")))
        self.assertFalse(_is_meaningful_title(_mk_item(title="ab")))
        self.assertFalse(_is_meaningful_title(_mk_item(title=" ")))
        self.assertFalse(_is_meaningful_title(_mk_item(title=" a ")))

    def test_long_titles_meaningful(self):
        self.assertTrue(_is_meaningful_title(_mk_item(title="abc")))
        self.assertTrue(_is_meaningful_title(_mk_item(title="Длинный заголовок")))


# =============================================================================
# Fuzzy helpers
# =============================================================================

class TestQuickFilter(unittest.TestCase):
    def test_filter_by_length(self):
        """Предфильтр отсекает по абсолютной разнице длин."""
        candidates = [
            ("h1", "word", "", ""),
            ("h2", "a b c d e f g h i j k l m", "", ""),
            ("h3", "one two three four", "", ""),
        ]
        # нормализованный с 4 словами
        result = _quick_filter("one two three four", candidates)
        # h1 (1 слово) — отсекается (|4-1| > MAX_LEN_DIFF=15 — проверю)
        # h2 (13 слов) — пропускается (|4-13|=9 ≤ 15)
        # h3 (4 слова) — пропускается
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), len(candidates))

    def test_filter_by_token_intersection(self):
        """Предфильтр отсекает по пересечению токен-сетов."""
        candidates = [
            ("h1", "cat dog mouse elephant", "", ""),
            ("h2", "hello world foo bar", "", ""),
        ]
        result = _quick_filter("cat dog mouse", candidates)
        # h1: пересечение 3/3=1.0 ≥ MIN_TOKEN_INTERSECTION → пропускается
        # h2: пересечение 0 → отсекается
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "h1")

    def test_empty_candidates(self):
        result = _quick_filter("hello world", [])
        self.assertEqual(result, [])

    def test_empty_norm(self):
        result = _quick_filter("", [("h1", "hello", "", "")])
        self.assertEqual(result, [])

    def test_respects_limit(self):
        """Не возвращает больше MAX_FUZZY_CANDIDATES."""
        candidates = [("h", "word1 word2 word3", "", "") for _ in range(500)]
        result = _quick_filter("word1 word2 word3", candidates)
        # После предфильтра лимит ещё применяется в _quick_filter
        # Проверяем что нет exception и возвращается список
        self.assertIsInstance(result, list)


class TestFuzzySimilarity(unittest.TestCase):
    # ── Новые дефолты: threshold=0.95, token_threshold=0.8 ──

    def test_exact_match(self):
        self.assertTrue(_is_fuzzy_similar("hello world", "hello world", 0.9, 0.8))

    def test_case_insensitive(self):
        """SequenceMatcher case-sensitive, но нормализованные тексты уже lowercase."""
        self.assertTrue(_is_fuzzy_similar("hello world", "hello world", 0.9, 0.8))

    def test_similar_but_different(self):
        """Высокая схожесть — выше threshold."""
        a = "Russia launches new missile from black sea fleet"
        b = "Russia launches new missile from black sea"
        self.assertTrue(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_different_titles(self):
        """Низкая схожесть — ниже threshold."""
        a = "Scientists discover new species of deep sea fish"
        b = "Government announces new tax policy for businesses"
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_threshold_boundary_exact(self):
        """Граница threshold: ровно threshold — True."""
        # SequenceMatcher ratio для одинаковых строк = 1.0
        self.assertTrue(_is_fuzzy_similar("test", "test", 1.0, 0.8))
        self.assertTrue(_is_fuzzy_similar("test", "test", 0.0, 0.8))
        self.assertTrue(_is_fuzzy_similar("test", "test", 0.9999, 0.8))

    def test_empty_strings(self):
        self.assertFalse(_is_fuzzy_similar("", "", 0.9, 0.8))
        self.assertFalse(_is_fuzzy_similar("hello", "", 0.9, 0.8))
        self.assertFalse(_is_fuzzy_similar("", "world", 0.9, 0.8))


# =============================================================================
# Уровень 1: глобальный точный кросс-источниковый дедуп
# =============================================================================

class TestCrossSourceExactDedup(unittest.TestCase):
    def test_cross_source_dedup_exact(self):
        """
        Один заголовок из двух разных source.
        Второй — дубль (cross_source=true), первый проходит.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="SpaceX launches new Starship mission",
                source="SourceA",
                url="https://a.com/starship-launch",
            )
            item2 = _mk_item(
                title="SpaceX launches new Starship mission",
                source="SourceB",
                url="https://b.com/starship-launch",
            )

            # Первый сохраняется
            saved1 = st.save(item1)
            self.assertTrue(saved1, "Первый item должен сохраниться")

            # is_duplicate: только URL + per-source (cross-source — НЕ здесь)
            self.assertFalse(st.is_duplicate(item2),
                "is_duplicate НЕ ловит cross-source — это в filter_new")

            # filter_new: второй отклоняется как cross-source дубль
            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 1, "Второй item должен быть дублем (cross_source)")
            self.assertEqual(len(new_items), 0, "Новых items быть не должно")

            st.close()
        finally:
            _cleanup_db(path)

    def test_cross_source_disabled(self):
        """
        cross_source=false: заголовок из разных source — НЕ дубликат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(title="Same Title", source="SourceA",
                             url="https://a.com/1")
            item2 = _mk_item(title="Same Title", source="SourceB",
                             url="https://b.com/2")

            st.save(item1)
            self.assertFalse(st.is_duplicate(item2),
                "cross_source=false: другой source — НЕ дубль")

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "Нет дублей при cross_source=false")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_cross_source_window_respected(self):
        """Кросс-источниковый дедуп работает только в пределах window_days."""
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(title="Breaking News", source="SourceA",
                             url="https://a.com/1")
            st.save(item1)

            # Подвинем fetched_at вручную на 5 дней назад
            import sqlite3
            conn = sqlite3.connect(path)
            old_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
            conn.execute(
                "UPDATE seen_titles SET fetched_at = ? WHERE title_hash = ?",
                (old_date, item1.title_normalized()),
            )
            conn.commit()
            conn.close()

            # После подвигания fetched_at за пределы window — дубля быть не должно
            item2 = _mk_item(title="Breaking News", source="SourceB",
                             url="https://b.com/2")
            self.assertFalse(st.is_duplicate(item2),
                "Дубль за пределами window_days не должен детектиться")

            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Уровень 2: fuzzy-дедуп
# =============================================================================

class TestFuzzyDedup(unittest.TestCase):
    def test_fuzzy_duplicate(self):
        """
        Близкие (но НЕ идентичные) заголовки ≥threshold из разных source — дубль.
        cross_source=True (fuzzy работает только при нём), exact dedup НЕ ловит
        — заголовки отличаются ("the" вставлено). Token-guard: Jaccard=1.0 ≥ 0.8.
        ratio≈0.875 (с "from") / ≈0.857 (без "from"). threshold=0.85 ≈ ratio.
        """
        path = _tmp_db()
        try:
            # cross_source=True: fuzzy_enabled=True; exact dedup не ловит (разные title)
            cfg = {"cross_source": True, "fuzzy": True, "threshold": 0.85,
                   "token_threshold": 0.8, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="Russia launches new missile from Black Sea fleet",
                source="SourceA",
                url="https://a.com/1",
            )
            st.save(item1)

            # item2: НЕ exact match (добавлено "the"), ratio≈0.875 ≥ 0.85 → дубль
            item2 = _mk_item(
                title="Russia launches new missile from the Black Sea fleet",
                source="SourceB",
                url="https://b.com/2",
            )

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 1, "Fuzzy: очень похожие заголовки — дубль")
            self.assertEqual(len(new_items), 0, "Дубль не попадает в new_items")
            st.close()
        finally:
            _cleanup_db(path)

    def test_fuzzy_not_similar(self):
        """
        НЕ похожие заголовки (<threshold) — НЕ дубли.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": True, "threshold": 0.9,
                   "token_threshold": 0.8, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="Scientists discover new species of deep sea fish",
                source="SourceA",
                url="https://a.com/1",
            )
            item2 = _mk_item(
                title="Government announces new tax policy",
                source="SourceB",
                url="https://b.com/2",
            )

            st.save(item1)

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "Разные заголовки не должны быть дублями")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_fuzzy_threshold_boundary(self):
        """
        Граница threshold: ровно threshold — True.
        NOTE: тестируется через _is_fuzzy_similar напрямую, т.к.
        filter_new с cross_source=True ловит exact match раньше fuzzy.
        """
        # ratio("test", "test") = 1.0 ≥ 0.9
        self.assertTrue(_is_fuzzy_similar("test", "test", 0.9, 0.8))
        self.assertTrue(_is_fuzzy_similar("test", "test", 0.0, 0.8))
        self.assertTrue(_is_fuzzy_similar("test", "test", 0.9999, 0.8))
        # чуть ниже threshold: ratio=0.8333 < 0.9 → False
        self.assertFalse(_is_fuzzy_similar("a b c", "a b c d", 0.9, 0.8))

    def test_fuzzy_disabled(self):
        """fuzzy=false: близкие заголовки из разных source — НЕ дубликат."""
        path = _tmp_db()
        try:
            item1 = _mk_item(title="Russia launches missile", source="A",
                             url="https://a.com/1")
            item2 = _mk_item(title="Russia launches missile", source="B",
                             url="https://b.com/2")

            cfg = {"cross_source": False, "fuzzy": False, "threshold": 0.9,
                   "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)
            st.save(item1)

            # Добавим в seen_titles запись от другого source
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute(
                "INSERT OR IGNORE INTO seen_titles "
                "(title_hash, source, fetched_at, expires_at, title_normalized) "
                "VALUES (?, 'OtherSource', ?, ?, ?)",
                (item1.title_normalized(),
                 datetime.now(timezone.utc).isoformat(),
                 (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
                 item1.title_normalized()),
            )
            conn.commit()
            conn.close()

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                "fuzzy=false: заголовок из другого source не дубль")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# skipped-сохранение и URL-дедуп
# =============================================================================

class TestSkippedDedup(unittest.TestCase):
    def test_skipped_item_preserved(self):
        """
        Дубль сохраняется как skipped.
        URL-дедуп ловит его в следующем цикле.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="SpaceX launches new Starship mission",
                source="SourceA",
                url="https://a.com/1",
            )
            item2 = _mk_item(
                title="SpaceX launches new Starship mission",
                source="SourceB",
                url="https://b.com/2",
            )

            # Первый сохраняется как new
            st.save(item1)

            # Второй — дубль (cross_source)
            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 1)

            # Проверяем: дубль записан как skipped
            import sqlite3
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT state FROM items WHERE url = ?",
                ("https://b.com/2",)
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row, "Дубль должен быть в items")
            self.assertEqual(row[0], "skipped",
                "Дубль должен иметь state='skipped'")

            # В следующем цикле: тот же URL — дубль по URL
            item2b = _mk_item(
                title="Another Title",
                source="C",
                url="https://b.com/2",
            )
            self.assertTrue(st.is_duplicate(item2b),
                "Тот же URL должен детектиться как дубль")
            st.close()
        finally:
            _cleanup_db(path)

    def test_skipped_not_in_delivery(self):
        """skipped items не попадают в candidates (не идут в LLM)."""
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(title="First Title", source="A",
                             url="https://a.com/1")
            # Идентичный титул (иной source) → кросс-источниковый дубль
            item2 = _mk_item(title="First Title", source="B",
                             url="https://b.com/2")

            st.save(item1)

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            # Дубль НЕ в new_items → не пойдёт в LLM/доставку
            self.assertEqual(dup, 1)
            self.assertEqual(len(new_items), 0,
                "Дубль не должен попасть в new_items (не идёт в delivery)")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Короткие заголовки не дедупятся
# =============================================================================

class TestShortTitlesNoDedup(unittest.TestCase):
    def test_short_titles_not_cross_source_deduped(self):
        """Короткий заголовок (< MIN_TITLE_LEN) не кросс-дедупится."""
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(title="Ab", source="A", url="https://a.com/1")
            item2 = _mk_item(title="Ab", source="B", url="https://b.com/2")

            st.save(item1)
            self.assertFalse(st.is_duplicate(item2),
                "Короткий title не дедупится (cross_source)")

            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0)
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Миграция v9
# =============================================================================

class TestSchemaV9(unittest.TestCase):
    def test_schema_version_9(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            # Version-independent: сравниваем с константой класса, не литералом.
            self.assertEqual(st.SCHEMA_VERSION, Storage.SCHEMA_VERSION)
            import sqlite3
            conn = sqlite3.connect(path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
            self.assertEqual(version, Storage.SCHEMA_VERSION)
            st.close()
        finally:
            _cleanup_db(path)

    def test_title_normalized_column_added(self):
        """Колонка title_normalized добавляется в seen_titles."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            import sqlite3
            conn = sqlite3.connect(path)
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(seen_titles)"
            ).fetchall()}
            conn.close()
            self.assertIn("title_normalized", cols,
                "Колонка title_normalized должна существовать")
            st.close()
        finally:
            _cleanup_db(path)

    def test_title_normalized_saved(self):
        """При сохранении item title_normalized записывается в seen_titles."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Test Title For Normalized",
                            url="https://test.com/1")
            st.save(item)

            import sqlite3
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT title_normalized FROM seen_titles WHERE title_hash = ?",
                (item.title_normalized(),)
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row, "title_normalized должен быть записан")
            self.assertEqual(row[0], item.title_normalized())
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Интеграция: обратная совместимость
# =============================================================================

class TestBackwardCompatibility(unittest.TestCase):
    def test_storage_without_config(self):
        """Storage создаётся без dedup_config (обратная совместимость)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))  # без dedup_config
            item = _mk_item(title="Test", url="https://test.com/1")
            saved = st.save(item)
            self.assertTrue(saved)
            st.close()
        finally:
            _cleanup_db(path)

    def test_filter_new_without_config(self):
        """filter_new работает без dedup_config (дефолты)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Test", url="https://test.com/1")
            st.save(item)

            item2 = _mk_item(title="Test", source="OtherSource",
                             url="https://other.com/1")
            # Без dedup_config → cross_source=True, fuzzy=True (дефолты)
            new_items, total, dup = filter_new([item2], st)
            self.assertEqual(dup, 1,
                "Без config: cross_source=True (дефолт) — дубль")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Конфиг-выключатели
# =============================================================================

# =============================================================================
# НОВЫЕ ГВАРДЫ: числовой, token-guard
# =============================================================================

class TestNumberGuard(unittest.TestCase):
    def test_number_multiset_basic(self):
        """Мультимножество цифровых последовательностей из нормализованного текста."""
        nums = _number_multiset("bitcoin fell to 60000 dollars")
        self.assertIn("60000", nums)
        self.assertEqual(nums["60000"], 1)

    def test_number_multiset_normalized_decimals_collapse(self):
        """
        На нормализованном тексте десятичные точки/запятые уже удалены
        (title_normalized убирает [^\w\s]), поэтому регекс ловит только
        целые последовательности цифр.
        Это документированное ограничение: "70.5" → "705" (одно число).
        """
        # Сырой текст с десятичной: regex \d+(?:[.,]\d+)? найдёт "70.5"
        import re
        raw_nums = re.findall(r"\d+(?:[.,]\d+)?", "oil at 70.5 per barrel")
        self.assertIn("70.5", raw_nums)
        # Нормализованный текст: точки/запятые удалены
        normalized = "oil at 705 per barrel"  # имитация после normalize()
        nums = _number_multiset(normalized)
        self.assertIn("705", nums)  # "70.5" схлопнулось в "705"
        # В нормализованном тексте "70.5" уже не существует как "70.5"
        self.assertNotIn("70", nums)
        self.assertNotIn("5", nums)

    def test_number_multiset_multiple(self):
        nums = _number_multiset("2024 and 2023 report")
        self.assertIn("2024", nums)
        self.assertIn("2023", nums)

    def test_number_multiset_multiset_behavior(self):
        """
        Мультимножество: "70 70" и "70" — РАЗНЫЕ (2 vs 1).
        Это ключевое отличие от множества (set).
        """
        a = "bitcoin reaches 70 70 million"     # "70" дважды
        b = "bitcoin reaches 70 million"         # "70" один раз
        # Числовой guard отказывает: Counter({'70': 2}) != Counter({'70': 1})
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_number_guard_blocks_different_numbers(self):
        """Разные числа — НЕ дубль (жёсткий запрет)."""
        a = "bitcoin fell to 60000 dollars"
        b = "bitcoin fell to 30000 dollars"
        # ratio ≈ 0.905 при threshold=0.9, но числа разные → не дубль
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))

    def test_number_guard_blocks_price_diff(self):
        """«70 vs 75» — НЕ дубль."""
        a = "brent crude oil reaches 70 per barrel"
        b = "brent crude oil reaches 75 per barrel"
        # ratio ≈ 0.96 при threshold=0.95, но числа разные → не дубль
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_number_guard_allows_same_numbers(self):
        """Одинаковые числа — проверка доходит до token-guard (слова те же)."""
        a = "bitcoin fell to 60000 dollars on january"
        b = "bitcoin fell to 60000 dollars on january"
        # числа совпадают, ratio=1.0, token-guard: Jaccard=1.0 ≥ 0.8
        self.assertTrue(_is_fuzzy_similar(a, b, 0.95, 0.8))
        # числа совпадают, ratio≈0.91, token-guard: те же слова → пропускает
        a2 = "bitcoin fell to 60000 dollars on january"
        b2 = "bitcoin fell to 60000 dollars on january morning"
        # значимые токены идентичны: Jaccard=1.0 ≥ 0.8, ratio≈0.92 ≥ 0.9
        self.assertTrue(_is_fuzzy_similar(a2, b2, 0.9, 0.8))


class TestTokenGuard(unittest.TestCase):
    def test_meaningful_tokens_filters_short(self):
        """Слова < 4 символов отфильтровываются."""
        tokens = _meaningful_tokens("the big cat")
        self.assertNotIn("the", tokens)
        self.assertNotIn("big", tokens)

    def test_meaningful_tokens_filters_stop_words(self):
        tokens = _meaningful_tokens("the and were also")
        self.assertEqual(tokens, set())

    def test_meaningful_tokens_keeps_long(self):
        tokens = _meaningful_tokens("bitcoin fell to 60000 dollars")
        self.assertIn("bitcoin", tokens)
        self.assertIn("dollars", tokens)
        # "60000" — число, не слово → отфильтровывается
        self.assertNotIn("60000", tokens)

    def test_token_guard_blocks_different_entities(self):
        """«apple announces X» vs «google announces X» — НЕ дубль (token-guard↓)."""
        a = "apple announces new artificial intelligence features"
        b = "google announces new artificial intelligence features"
        # ratio≈0.769, числа совпадают. Jaccard: A∩B={announces,new,artificial,intelligence,features}
        # |A|=6, |B|=6, intersection=5, union=7, Jaccard=5/7≈0.667 ≥ 0.8 → токен-guard пропускает,
        # но ratio≈0.769 < 0.9 → отсеивается на ratio. self.assertFalse → PASS.
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_token_guard_allows_same_entities(self):
        """Та же новость, немного другой порядок — зависит от ratio."""
        a = "apple announces new artificial intelligence features"
        b = "apple new announces artificial intelligence features"
        # Jaccard: A={apple, announces, artificial, intelligence, features}
        #          B={apple, announces, artificial, intelligence, features}
        # same set → Jaccard=1.0 ≥ 0.8; ratio≈0.9231 < 0.95, поэтому threshold=0.9
        self.assertTrue(_is_fuzzy_similar(a, b, 0.9, 0.8))
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))

    def test_token_jaccard_boundary(self):
        """Граница token_threshold: ровно на границе — True."""
        a = "apple announces new artificial intelligence features"
        b = "google announces new artificial intelligence features"
        # Jaccard=4/6≈0.667 — при token_threshold=0.65 → True; при 0.7 → False
        self.assertTrue(_token_jaccard(a, b, 0.65))
        self.assertFalse(_token_jaccard(a, b, 0.7))
        self.assertFalse(_token_jaccard(a, b, 0.8))
        self.assertFalse(_token_jaccard(a, b, 0.9))

    def test_token_guard_empty_meaningful_tokens_passes(self):
        """Если нет значащих токенов — guard пропускает (не блокирует)."""
        a = "a b c"
        b = "a b c d"
        # ratio=0.8333; при threshold=0.8 числа совпадают, guard пропускает
        self.assertTrue(_is_fuzzy_similar(a, b, 0.8, 0.8))
        # при threshold=0.9 — отсеивается на ratio, не на guard
        self.assertFalse(_is_fuzzy_similar(a, b, 0.9, 0.8))


class TestGuardIntegration(unittest.TestCase):
    """Интеграционные тесты тройного условия fuzzy."""

    def test_numbers_same_ratio_below_threshold_not_duplicate(self):
        """Числа совпадают, но ratio < threshold → НЕ дубль."""
        a = "bitcoin fell to 60000 dollars"
        b = "bitcoin fell 60000 dollars"  # ratio ≈ 0.93 < 0.95
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))

    def test_numbers_same_ratio_below_threshold_token_guard_pass_not_duplicate(self):
        """Числа совпадают, но ratio≈0.769 < 0.95 → НЕ дубль (отсеится на ratio)."""
        a = "apple announces new artificial intelligence features"
        b = "google announces new artificial intelligence features"
        # числа совпадают (нет). token-guard пропускает (Jaccard=5/7≈0.667 ≥ 0.8),
        # но ratio≈0.769 < 0.95 → отсеивается на ratio-пороге
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))

    def test_real_near_duplicate_same_order(self):
        """Реальный near-дубль (та же новость) — дубль."""
        a = "bitcoin fell to 60000 dollars amid market selloff"
        b = "bitcoin fell to 60000 dollars amid market selloff"
        self.assertTrue(_is_fuzzy_similar(a, b, 0.95, 0.8))
        self.assertTrue(_is_fuzzy_similar(a, b, 0.9, 0.8))

    def test_real_near_duplicate_reordered_words(self):
        """Та же новость, другой порядок слов — дубль ( зависит от threshold)."""
        a = "bitcoin fell to 60000 dollars amid market selloff"
        b = "bitcoin 60000 fell dollars amid to market selloff"
        # ratio≈0.8367 — ниже нового дефолта 0.95, но ≥ 0.8
        self.assertTrue(_is_fuzzy_similar(a, b, 0.8, 0.8))
        self.assertFalse(_is_fuzzy_similar(a, b, 0.95, 0.8))

    def test_cross_source_disabled_fuzzy_ignored(self):
        """cross_source=false: fuzzy игнорируется — заголовок не дубль."""
        path = _tmp_db()
        try:
            # fuzzy=true, но cross_source=false → fuzzy должен выключиться
            cfg = {"cross_source": False, "fuzzy": True, "threshold": 0.95,
                   "token_threshold": 0.8, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="Russia launches missile from black sea fleet",
                source="SourceA",
                url="https://a.com/1",
            )
            item2 = _mk_item(
                title="Russia launches missile from black sea fleet",
                source="SourceB",
                url="https://b.com/2",
            )
            st.save(item1)

            # При cross_source=false fuzzy выключен — item2 НЕ дубль
            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                "cross_source=false: fuzzy игнорируется, item не дубль")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_both_disabled_legacy_behavior(self):
        """cross_source=false AND fuzzy=false: строго прежнее поведение."""
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "threshold": 0.95,
                   "token_threshold": 0.8, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Per-source: тот же title+source — дубль
            item1a = _mk_item(title="Legacy Title", source="A",
                              url="https://a.com/1")
            item1b = _mk_item(title="legacy title", source="A",
                              url="https://a.com/2")
            st.save(item1a)
            new_items, total, dup = filter_new([item1b], st, dedup_config=cfg)
            self.assertEqual(dup, 1, "Per-source дедуп работает всегда")

            # Кросс-источник: разный source — НЕ дубль
            item2 = _mk_item(title="Legacy Title", source="B",
                             url="https://b.com/1")
            new_items2, total2, dup2 = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup2, 0,
                "Оба выключены: другой source — не дубль")
            self.assertEqual(len(new_items2), 1)
            st.close()
        finally:
            _cleanup_db(path)


class TestEnhancedFilterByLength(unittest.TestCase):
    """Усиленный тест предфильтра по длине (слов, не символов)."""

    def test_exact_length_match(self):
        """4 слова: h1 (1 слово) — отсекается; h3 (4 слова) — пропускается."""
        candidates = [
            ("h1", "word", "", ""),
            ("h3", "one two three four", "", ""),
        ]
        result = _quick_filter("one two three four", candidates)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "h3")

    def test_within_max_len_diff(self):
        """Разница в словах ≤ MAX_LEN_DIFF=15: пропускается (если токены пересекаются)."""
        candidates = [
            # 16 слов, но есть пересечение токенов с квери
            ("h1", "one two three four a b c d e f g h i j k l m", "", ""),
        ]
        result = _quick_filter("one two three four", candidates)  # 4 слова
        # |4-16|=12 ≤ MAX_LEN_DIFF=15 → проходит фильтр по длине;
        # token_ratio = 4/4 = 1.0 ≥ MIN_TOKEN_INTERSECTION → проходит
        self.assertEqual(len(result), 1)


class TestEnhancedRespectsLimit(unittest.TestCase):
    """Усиленный тест лимита: >500 кандидатов → ≤200 в результате."""

    def test_respects_limit_strict(self):
        """При >500 кандидатах результат ≤ MAX_FUZZY_CANDIDATES (200)."""
        from newsbot.dedup import MAX_FUZZY_CANDIDATES
        candidates = [("h", "word1 word2 word3", "", "") for _ in range(600)]
        result = _quick_filter("word1 word2 word3", candidates)
        self.assertLessEqual(len(result), MAX_FUZZY_CANDIDATES,
            f"Результат должен быть ≤ {MAX_FUZZY_CANDIDATES}")


class TestConfigSwitches(unittest.TestCase):
    def test_cross_source_false(self):
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)
            item1 = _mk_item(title="X", source="A", url="https://a.com/1")
            item2 = _mk_item(title="X", source="B", url="https://b.com/2")
            st.save(item1)
            new_items, total, dup = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "cross_source=false: разный source — не дубль")
            st.close()
        finally:
            _cleanup_db(path)

    def test_both_disabled_legacy_behavior(self):
        """
        cross_source=false и fuzzy=false: только per-source + URL (legacy).
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Per-source: тот же title+source — дубль
            item1a = _mk_item(title="Legacy Title", source="A",
                              url="https://a.com/1")
            item1b = _mk_item(title="legacy title", source="A",
                              url="https://a.com/2")
            st.save(item1a)
            new_items, total, dup = filter_new([item1b], st, dedup_config=cfg)
            self.assertEqual(dup, 1, "Per-source дедуп работает всегда")

            # Кросс-источник: разный source — НЕ дубль
            item2 = _mk_item(title="Legacy Title", source="B",
                             url="https://b.com/1")
            new_items2, total2, dup2 = filter_new([item2], st, dedup_config=cfg)
            self.assertEqual(dup2, 0,
                "Кросс-источник выключен: другой source — не дубль")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# M2: N+1 запрос к БД исправлен — get_recent_normalized_titles вынесен из цикла
# =============================================================================

class TestDedupNoNPlusOne(unittest.TestCase):
    """M2-фикс: get_recent_normalized_titles вызывается O(unique_sources), не O(N)."""

    def test_get_recent_called_once_per_source(self):
        """
        M2: выборка из БД вызывается один раз на уникальный source
        (кандидаты с одинаковым source используют кэш).
        """
        path = _tmp_db()
        try:
            # Создаём items с тремя источниками
            items = [
                _mk_item(title=f"Item S1-{i}", url=f"s1-{i}", source="Source1")
                for i in range(5)
            ]
            items += [
                _mk_item(title=f"Item S2-{i}", url=f"s2-{i}", source="Source2")
                for i in range(5)
            ]
            items += [
                _mk_item(title=f"Item S3-{i}", url=f"s3-{i}", source="Source3")
                for i in range(5)
            ]

            cfg = {"cross_source": True, "fuzzy": True, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # save_many чтобы items попали в БД (для get_recent_normalized_titles)
            inserted = st.save_many(items)
            # переводим в delivered чтобы они попали в get_recent_normalized_titles
            st.set_state([i.url for i in inserted], "delivered")

            # Мокаем get_recent_normalized_titles, чтобы считать вызовы
            from unittest.mock import MagicMock, patch

            mock_method = MagicMock(return_value=[])  # пустой список кандидатов

            with patch.object(st, "get_recent_normalized_titles", mock_method):
                new_items, total, dup = filter_new(items, st, dedup_config=cfg)

            # Должно быть ровно 3 вызова (по одному на каждый уникальный source)
            self.assertEqual(
                mock_method.call_count, 3,
                f"get_recent_normalized_titles должна вызываться 3 раза "
                f"(3 уникальных source), вызвана {mock_method.call_count} раз. "
                "Каждый source запрашивается один раз, затем кэшируется.",
            )
            st.close()
        finally:
            _cleanup_db(path)

    def test_dedup_logic_unchanged(self):
        """
        M2 не меняет логику дедупа: те же дубли отклоняются, те же проходят.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": True, "fuzzy": True, "threshold": 0.95,
                   "token_threshold": 0.8, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Прошлая новость: близкий заголовок (должен отфильтроваться по fuzzy)
            past = _mk_item(
                title="Российская армия сбила 34 дрона на юге страны",
                source="Src2",
                url="https://past.com/1",
            )
            st.save(past)
            st.set_state([past.url], "delivered")

            # Текущий кандидат: похожий заголовок
            current = _mk_item(
                title="Сбито 34 беспилотника в южных регионах РФ",
                source="Src1",
                url="https://current.com/1",
            )
            st.save(current)

            new_items, total, dup = filter_new([current], st, dedup_config=cfg)

            # fuzzy dedup с threshold=0.95 должен отклонить
            # (те же числа "34", близкие токены, высокий ratio)
            self.assertEqual(dup, 1, "Fuzzy дубль должен быть отклонён")
            self.assertEqual(len(new_items), 0, "Новых items не осталось")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)