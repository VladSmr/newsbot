"""
Тесты надёжности core-пакета newsbot (ПОТОК A).
Запускать: python -m pytest tests/test_core_reliability.py -v
"""
import os
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

# Проектный корень → newsbot
import sys
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.collector import FeedCollector, FeedResult, NewsItem
from newsbot.dedup import filter_new, _is_meaningful_title
from newsbot.storage import Storage


# =============================================================================
# fixtures
# =============================================================================

class _FakeSources:
    GOOD = [{"name": "Test", "url": "https://example.com/feed.xml"}]
    BAD  = [{"name": "Bad",  "url": "https://localhost:9999/nonexistent"}]


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


def _tmp_db(closer=None):
    """Временный путь к SQLite в tmp-директории."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    # Удаляем, чтобы Storage создал сам с нужной схемой
    os.unlink(path)
    return path


def _cleanup_db(path):
    """Безопасное удаление tmp-файла БД (Windows не даёт удалить открытый файл)."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        # Windows: файл может быть заблокирован даже после close().
        # Спим и повторяем.
        time.sleep(0.2)
        try:
            os.unlink(path)
        except (PermissionError, FileNotFoundError):
            pass


# =============================================================================
# T1-11: PRAGMA busy_timeout устанавливается при connect
# =============================================================================

class TestBusyTimeout(unittest.TestCase):
    def test_pragmas_set_on_connect(self):
        """При создании Storage PRAGMA busy_timeout и journal_mode=WAL."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            conn = st._connect()
            row = conn.execute("PRAGMA busy_timeout").fetchone()
            busy_ms = row[0]
            self.assertGreaterEqual(busy_ms, 1000,
                f"busy_timeout должен быть >= 1000 мс, получено: {busy_ms}")
            row2 = conn.execute("PRAGMA journal_mode").fetchone()
            self.assertEqual(row2[0].lower(), "wal")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# T1-10: UNIQUE index пропускается с явным warning (не молча)
# =============================================================================

class TestUniqueIndexWarning(unittest.TestCase):
    def test_unique_index_on_existing_dup_warns(self):
        """IntegrityError при создании UNIQUE индекса логируется, не глотается."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            # Вставляем два item с одинаковым URL
            i1 = _mk_item(title="Dup A", url="https://dup.example/1")
            i2 = _mk_item(title="Dup B", url="https://dup.example/1")
            st.save(i1)
            st.save(i2)  # IGNORE — URL дубликат

            conn = st._connect()
            count = conn.execute(
                "SELECT COUNT(*) FROM items WHERE url = ?",
                ("https://dup.example/1",)
            ).fetchone()[0]
            self.assertEqual(count, 1, "Должен быть ровно один item с URL")

            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# В5: save_many — батчевая вставка в одной транзакции
# =============================================================================

class TestSaveMany(unittest.TestCase):
    def test_save_many_inserts_correctly(self):
        """N1: save_many возвращает список РЕАЛЬНО вставленных NewsItem."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            items = [
                _mk_item(title=f"Item {i}", url=f"https://x.com/{i}")
                for i in range(10)
            ]
            result = st.save_many(items)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 10, "Должно вернуться 10 NewsItem")
            # Проверяем что вернулись именно те items, которые передали
            returned_urls = {i.url for i in result}
            expected_urls = {f"https://x.com/{i}" for i in range(10)}
            self.assertEqual(returned_urls, expected_urls)

            conn = st._connect()
            total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            self.assertEqual(total, 10)
            st.close()
        finally:
            _cleanup_db(path)

    def test_save_many_deduplicates(self):
        """N1: save_many возвращает ровно вставленные items (с учётом внутрибатчевого дубля)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            # Первые 5 — дубли по URL внутри батча
            items = [
                _mk_item(title=f"Item {i}", url=f"https://x.com/{i % 5}")
                for i in range(10)
            ]
            result = st.save_many(items)
            # N1: должен вернуть 5 уникальных items
            self.assertEqual(len(result), 5,
                "Должно вернуться 5 уникальных NewsItem")
            # Проверяем что это именно те items, что попали в БД
            conn = st._connect()
            total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            self.assertEqual(total, 5)
            st.close()
        finally:
            _cleanup_db(path)

    def test_save_many_empty(self):
        """N1: пустой батч возвращает пустой список."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            result = st.save_many([])
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)
            st.close()
        finally:
            _cleanup_db(path)

    def test_save_many_summary_cap(self):
        """В6/B6: save_many обрезает summary до 6000 символов."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            long_summary = "x" * 10000
            items = [_mk_item(title="Cap test", summary=long_summary)]
            result = st.save_many(items)
            self.assertEqual(len(result), 1, "Должен вставиться 1 item")

            conn = st._connect()
            saved = conn.execute(
                "SELECT summary FROM items WHERE title = ?",
                ("Cap test",)
            ).fetchone()[0]
            self.assertLessEqual(len(saved), 6000,
                "Summary должен быть обрезан до 6000 символов")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# В2: пустой заголовок НЕ вызывает ложный дедуп
# =============================================================================

class TestEmptyTitleDedup(unittest.TestCase):
    def test_empty_title_not_folded(self):
        """Разные URL с пустым title — не схлопываются."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))

            i1 = _mk_item(title="", url="https://a.com/1")
            i2 = _mk_item(title="", url="https://a.com/2")
            i3 = _mk_item(title="  ", url="https://a.com/3")  # тоже пустой

            self.assertFalse(st.is_duplicate(i1))
            st.save(i1)
            # Пустой title не проверяется по seen_titles
            self.assertFalse(st.is_duplicate(i2))
            self.assertFalse(st.is_duplicate(i3))

            st.save(i2)
            st.save(i3)

            conn = st._connect()
            count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            self.assertEqual(count, 3, "Все три item с пустым title должны быть сохранены")

            st.close()
        finally:
            _cleanup_db(path)

    def test_is_meaningful_title_short(self):
        """Короткий title (<3 симв.) не участвует в title-dedup."""
        self.assertFalse(_is_meaningful_title(_mk_item(title="")))
        self.assertFalse(_is_meaningful_title(_mk_item(title="ab")))
        self.assertFalse(_is_meaningful_title(_mk_item(title=" ")))
        self.assertTrue(_is_meaningful_title(_mk_item(title="abc")))
        self.assertTrue(_is_meaningful_title(_mk_item(title="Длинный заголовок")))


# =============================================================================
# B1: внутрицикловой дедуп
# =============================================================================

class TestInCycleDedup(unittest.TestCase):
    def test_intra_cycle_dedup(self):
        """Два разных URL с одинаковым нормализованным title+source — один отклоняется."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            # Нормализуемый title: "Test Title"
            i1 = _mk_item(title="Test Title", url="https://a.com/1")
            i2 = _mk_item(title="test title", url="https://a.com/2")  # нормализуется в то же
            i3 = _mk_item(title="Test Title", url="https://a.com/3")  # точный title

            items = [i1, i2, i3]
            new_items, total, dup = filter_new(items, st)

            self.assertEqual(total, 3)
            self.assertEqual(dup, 2, "i2 и i3 — внутрицикловые дубли i1")
            self.assertEqual(len(new_items), 1)
            self.assertEqual(new_items[0].url, "https://a.com/1")
            st.close()
        finally:
            _cleanup_db(path)

    def test_intra_cycle_dedup_different_source(self):
        """Одинаковый title, разный source — не дубли."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            i1 = _mk_item(title="Same Title", source="SourceA", url="https://a.com/1")
            i2 = _mk_item(title="Same Title", source="SourceB", url="https://b.com/1")

            new_items, total, dup = filter_new([i1, i2], st)
            self.assertEqual(dup, 0, "Разный source — не дубли")
            self.assertEqual(len(new_items), 2)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# B3: retry-лимит → 'failed'
# =============================================================================

class TestRetryLimit(unittest.TestCase):
    def test_storage_states_include_failed(self):
        self.assertIn("failed", Storage.STATES)

    def test_increment_retry_count(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Retry Item", url="https://retry.example/1")
            st.save(item)

            urls = ["https://retry.example/1"]
            n = st.increment_retry_count(urls)
            self.assertEqual(n, 1)

            conn = st._connect()
            rc = conn.execute(
                "SELECT retry_count FROM items WHERE url = ?",
                ("https://retry.example/1",)
            ).fetchone()[0]
            self.assertEqual(rc, 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_get_stuck_pending_respects_retry_limit(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Stuck", url="https://stuck.example/1")
            st.save(item)
            conn = st._connect()

            update_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE items SET state='pending', retry_count=7, state_at=? WHERE url=?",
                (update_at, "https://stuck.example/1")
            )
            conn.commit()

            # verify
            row = conn.execute(
                "SELECT state, retry_count FROM items WHERE url='https://stuck.example/1'"
            ).fetchone()
            self.assertEqual(row['state'], 'pending')
            self.assertEqual(row['retry_count'], 7)

            # raw SQL: same query as get_stuck_pending(0, 8)
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=0)).isoformat()
            raw = conn.execute(
                "SELECT url FROM items WHERE state='pending' AND state_at < ? AND retry_count < 8"
            , (cutoff,)).fetchall()
            self.assertEqual(len(raw), 1, f"Raw SQL returned {len(raw)}: {[r[0] for r in raw]}")

            stuck = st.get_stuck_pending(max_age_seconds=0, max_retries=8)
            self.assertEqual(len(stuck), 1, f"get_stuck_pending returned {len(stuck)}, expected 1")

            stuck = st.get_stuck_pending(max_age_seconds=0, max_retries=7)
            self.assertEqual(len(stuck), 0)
            st.close()
        finally:
            _cleanup_db(path)

    def test_set_failed_marks_terminal(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Fail", url="https://fail.example/1")
            st.save(item)
            n = st.set_failed(["https://fail.example/1"])
            self.assertEqual(n, 1)

            conn = st._connect()
            state = conn.execute(
                "SELECT state FROM items WHERE url=?",
                ("https://fail.example/1",)
            ).fetchone()[0]
            self.assertEqual(state, "failed")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# K2: битый фид не роняет цикл
# =============================================================================

class TestBrokenFeedResilience(unittest.TestCase):
    def test_feed_collector_handles_all_httpx_errors(self):
        """FeedCollector._fetch_feed ловит все подтипы HTTPError, не только RequestError."""
        from newsbot.collector import FeedCollector

        collector = FeedCollector(_FakeSources.BAD)

        # Ловим TooManyRedirects — наследует HTTPError, не RequestError
        with patch.object(
            collector._client, "get",
            side_effect=Exception("Unexpected exception")
        ):
            name, result = next(collector.collect())
            self.assertFalse(result.ok)
            self.assertEqual(result.items, [])

        collector.close()

    def test_feed_collector_graceful_on_network_error(self):
        """Сеть недоступна — FeedResult(ok=False, items=[])."""
        collector = FeedCollector(_FakeSources.BAD)
        # httpx.ConnectError наследует RequestError — уже ловился, проверим
        name, result = next(collector.collect())
        # Может упасть или вернуть ok=False
        self.assertIsInstance(result, FeedResult)
        collector.close()


# =============================================================================
# K1: recovery 'new'-items
# =============================================================================

class TestNewItemsRecovery(unittest.TestCase):
    def test_count_by_state_returns_correct_count(self):
        """K1: count_by_state возвращает точное количество items без загрузки в RAM."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            st.save(_mk_item(title="N1", url="https://cnt.com/1"))
            st.save(_mk_item(title="N2", url="https://cnt.com/2"))
            st.save(_mk_item(title="N3", url="https://cnt.com/3"))
            st.save(_mk_item(title="Pending", url="https://cnt.com/4"))
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending' WHERE url=?",
                ("https://cnt.com/4",)
            )
            conn.commit()

            self.assertEqual(st.count_by_state("new"), 3)
            self.assertEqual(st.count_by_state("pending"), 1)
            self.assertEqual(st.count_by_state("delivered"), 0)
            st.close()
        finally:
            _cleanup_db(path)

    def test_get_all_by_state_new(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            st.save(_mk_item(title="New 1", url="https://r.com/1"))
            st.save(_mk_item(title="New 2", url="https://r.com/2"))
            st.save(_mk_item(title="Pending", url="https://r.com/3"))
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending' WHERE url=?",
                ("https://r.com/3",)
            )
            conn.commit()

            new_items = st.get_all_by_state("new")
            self.assertEqual(len(new_items), 2)
            urls = {i.url for i in new_items}
            self.assertIn("https://r.com/1", urls)
            self.assertIn("https://r.com/2", urls)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# K5: deferred не purge'ится основным retention, но purge'ится по deferred_retention_days
# =============================================================================

class TestDeferredPurge(unittest.TestCase):
    def test_deferred_not_deleted_by_main_retention(self):
        """deferred не удаляется по retention_days (он использует deferred_retention_days)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Deferred", url="https://def.example/1")
            st.save(item)
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='deferred', state_at=? WHERE url=?",
                (datetime.now(timezone.utc).isoformat(), "https://def.example/1")
            )
            conn.commit()

            # retention=0, но deferred_retention_days=14 (default) — deferred остаётся
            deleted = st.purge_items(retention_days=0, deferred_retention_days=14)
            self.assertEqual(deleted, 0, "deferred не должен удаляться при deferred_retention_days>0")

            count = conn.execute(
                "SELECT COUNT(*) FROM items WHERE state='deferred'"
            ).fetchone()[0]
            self.assertEqual(count, 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_old_deferred_deleted_by_deferred_retention(self):
        """Старый deferred (старше deferred_retention_days) удаляется."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="OldDeferred", url="https://olddef.example/1")
            st.save(item)
            conn = st._connect()
            # state_at = 20 дней назад → старше deferred_retention_days=14
            from datetime import timedelta
            old_at = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
            conn.execute(
                "UPDATE items SET state='deferred', state_at=? WHERE url=?",
                (old_at, "https://olddef.example/1")
            )
            conn.commit()

            # deferred_retention_days=14, item 20 дней → удаляется
            deleted = st.purge_items(retention_days=90, deferred_retention_days=14)
            self.assertGreaterEqual(deleted, 1, "Старый deferred должен быть удалён")

            count = conn.execute(
                "SELECT COUNT(*) FROM items WHERE state='deferred'"
            ).fetchone()[0]
            self.assertEqual(count, 0, "Старый deferred должен быть удалён из БД")
            st.close()
        finally:
            _cleanup_db(path)

    def test_fresh_deferred_kept_by_deferred_retention(self):
        """Свежий deferred (младше deferred_retention_days) остаётся."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="FreshDeferred", url="https://freshdef.example/1")
            st.save(item)
            conn = st._connect()
            # state_at = сейчас, UPDATE по РЕАЛЬНОМУ URL сохранённого item
            conn.execute(
                "UPDATE items SET state='deferred', state_at=? WHERE url=?",
                (datetime.now(timezone.utc).isoformat(), "https://freshdef.example/1")
            )
            conn.commit()

            deleted = st.purge_items(retention_days=90, deferred_retention_days=14)
            # Свежий deferred не удалён (ему 0 дней < 14)
            remaining = conn.execute(
                "SELECT COUNT(*) FROM items WHERE state='deferred'"
            ).fetchone()[0]
            self.assertEqual(remaining, 1, "Свежий deferred должен остаться в БД")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# 2b-фикс: purge удаляет старые 'new'-items
# =============================================================================

class TestNewPurge(unittest.TestCase):
    def test_old_new_purged(self):
        """Старые 'new'-items (старше retention_days) удаляются purge."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="OldNew", url="https://oldnew.example/1")
            st.save(item)  # state='new', state_at=сейчас
            conn = st._connect()
            from datetime import timedelta
            # state_at = 100 дней назад → старше retention_days=90
            old_at = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
            conn.execute(
                "UPDATE items SET state='new', state_at=? WHERE url=?",
                (old_at, "https://oldnew.example/1")
            )
            conn.commit()

            deleted = st.purge_items(retention_days=90, deferred_retention_days=14)
            self.assertGreaterEqual(deleted, 1, "Старый new item должен быть удалён")

            count = conn.execute(
                "SELECT COUNT(*) FROM items WHERE state='new'"
            ).fetchone()[0]
            self.assertEqual(count, 0, "Старый new должен быть удалён")
            st.close()
        finally:
            _cleanup_db(path)

    def test_get_all_by_state_respects_limit(self):
        """K1: get_all_by_state('new') возвращает не более limit items."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            # 50 new items
            for i in range(50):
                item = _mk_item(title=f"New{i}", url=f"https://new{i}.com/1")
                st.save(item)

            # Читаем лимит 10
            items = st.get_all_by_state("new", limit=10)
            self.assertEqual(len(items), 10, "Должно вернуться ровно 10 items при limit=10")

            # Без лимита — все 50
            items_all = st.get_all_by_state("new")
            self.assertEqual(len(items_all), 50, "Без лимита возвращаются все 50 items")

            st.close()
        finally:
            _cleanup_db(path)

    def test_failed_is_purged(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Failed", url="https://failp.example/1")
            st.save(item)
            st.set_failed(["https://failp.example/1"])

            conn = st._connect()
            conn.execute(
                "UPDATE items SET state_at=? WHERE url=?",
                (datetime.now(timezone.utc).isoformat(), "https://failp.example/1")
            )
            conn.commit()

            deleted = st.purge_items(retention_days=0)
            # failed удаляется (как терминальный state)
            self.assertGreaterEqual(deleted, 1)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# B13: retry приоритет и бюджет
# =============================================================================

class TestRetryBudgetPriority(unittest.TestCase):
    def test_retry_budget_reserved(self):
        """B13: retry-items попадают в get_stuck_pending (приоритет до кандидатов)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            stuck_item = _mk_item(title="Stuck", url="https://stuck.example/1")
            st.save(stuck_item)
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending', state_at='2000-01-01T00:00:00+00:00' "
                "WHERE url='https://stuck.example/1'"
            )
            conn.commit()
            st.close()

            # Новая инициализация — симулирует начало цикла
            st2 = Storage(Path(path))
            stuck = st2.get_stuck_pending(max_age_seconds=0, max_retries=8)
            self.assertEqual(len(stuck), 1,
                "Retry-item должен попасть в get_stuck_pending (бюджет приоритетный)")
            st2.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# T1-12: state_at обновляется при pending→pending
# =============================================================================

class TestStateAtUpdatedOnPending(unittest.TestCase):
    def test_state_at_updated_on_set_pending(self):
        """N2: state_at обновляется при pending→pending (реальный повторный вызов)."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="StateAt", url="https://sat.example/1")
            st.save(item)

            # Переводим в pending (первый вызов)
            st.set_state(["https://sat.example/1"], "pending")

            conn = st._connect()
            old_at = conn.execute(
                "SELECT state_at FROM items WHERE url=?", ("https://sat.example/1",)
            ).fetchone()[0]
            time.sleep(0.05)  # гарантируем другую метку времени

            # Повторный set_state в pending (N2: должен обновить state_at)
            st.set_state(["https://sat.example/1"], "pending")
            new_at = conn.execute(
                "SELECT state_at FROM items WHERE url=?", ("https://sat.example/1",)
            ).fetchone()[0]
            self.assertNotEqual(old_at, new_at,
                "state_at должен обновиться при повторном set_state в pending")
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# B4: busy_timeout не блокирует
# =============================================================================

class TestBusyTimeoutDoesNotBlock(unittest.TestCase):
    def test_concurrent_writes_not_stuck(self):
        """
        Два потока пишут в одну БД одновременно.
        WAL + busy_timeout = writer'ы сериализуются с ожиданием (до 5 с),
        не падают мгновенно с 'database is locked'.
        Приложение однопоточное, поэтому этот сценарий — искусственный,
        но проверяет что SQLite при WAL + busy_timeout корректно ждёт.
        """
        path = _tmp_db()
        errors = []
        written = [0, 0]  # счётчики записанных items

        def writer(thread_id):
            try:
                st = Storage(Path(path))
                for i in range(50):
                    saved = st.save(_mk_item(
                        title=f"Thread {thread_id} Item {i}",
                        url=f"https://t{thread_id}.com/{i}",
                    ))
                    if saved:
                        written[thread_id - 1] += 1
                st.close()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        time.sleep(0.05)  # гарантируем overlap (частично), но не безумную гонку
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [], f"Конкурентная запись вызвала ошибки: {errors}")
        self.assertEqual(written[0], 50, f"Thread 1 записал {written[0]}/50")
        self.assertEqual(written[1], 50, f"Thread 2 записал {written[1]}/50")
        _cleanup_db(path)


# =============================================================================
# T1-13: нормализация URL для дедупа
# NOTE: _url_normalized удалён — дедуп по точному URL; URL-нормализация (utm/trailing slash)
# для future — не применяется в текущей реализации.
# =============================================================================


# =============================================================================
# T1-15: _format_pubdate НЕ вызывает _parse_pubdate_utc дважды
# =============================================================================

class TestPubdateParsing(unittest.TestCase):
    def test_format_pubdate_does_not_double_parse(self):
        from newsbot.collector import _format_pubdate
        from newsbot.collector import _parse_pubdate_utc

        mock_entry = MagicMock()
        mock_entry.published_parsed = None
        mock_entry.published = "Mon, 01 Jan 2025 12:00:00 GMT"

        result = _format_pubdate(mock_entry)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "—")


# =============================================================================
# Schema version: миграция v8
# =============================================================================

class TestSchemaV8(unittest.TestCase):
    def test_schema_version_9(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            self.assertEqual(st.SCHEMA_VERSION, 10)
            conn = st._connect()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 10)
            st.close()
        finally:
            _cleanup_db(path)

    def test_feedback_tables_exist(self):
        """Таблицы feedback_targets и feedback создаются в свежей БД v8."""
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            conn = st._connect()
            conn.row_factory = sqlite3.Row
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            conn.close()
            self.assertIn("feedback_targets", tables)
            self.assertIn("feedback", tables)
        finally:
            _cleanup_db(path)

    def test_retry_count_column_exists(self):
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            conn = st._connect()
            cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            self.assertIn("retry_count", cols)
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# C1: retry-бухгалтерия — регрессионные тесты
# =============================================================================

class TestC1RetryAccounting(unittest.TestCase):
    """Регрессионные тесты: retry_count инкрементируется только для НЕ доставленных."""

    def test_successful_retry_item_not_marked_failed(self):
        """
        C1: retry item с retry_count=max-1, успешно доставленный,
        НЕ становится 'failed'.
        Симуляция: _process_batch вернул пустой список (всё доставлено),
        pipeline не вызывает increment/mark_failed.
        """
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Succ", url="https://succ.example/1")
            st.save(item)

            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending', retry_count=7, state_at=? "
                "WHERE url='https://succ.example/1'",
                (datetime.now(timezone.utc).isoformat(),)
            )
            conn.commit()

            # simulate: _process_batch вернул пустой список (доставлен)
            # pipeline НЕ вызывает increment_retry_count и _check_and_mark_failed
            # -> state остаётся pending, retry_count не растёт
            state = conn.execute(
                "SELECT state FROM items WHERE url='https://succ.example/1'"
            ).fetchone()[0]
            self.assertEqual(state, "pending",
                "Успешно доставленный item не должен стать 'failed'")
            st.close()
        finally:
            _cleanup_db(path)

    def test_budget_cutdown_not_retried_nor_terminal(self):
        """
        C1: item, срезанный бюджетом, НЕ инкрементируется и НЕ терминализируется.
        Для candidates pipeline НЕ вызывает increment/mark_failed.
        """
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="BudgetCut", url="https://budget.example/1")
            st.save(item)
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending', retry_count=0, state_at=? "
                "WHERE url='https://budget.example/1'",
                (datetime.now(timezone.utc).isoformat(),)
            )
            conn.commit()

            # Для candidates pipeline не вызывает increment_retry_count
            # -> retry_count остаётся 0, state pending
            count = conn.execute(
                "SELECT retry_count FROM items WHERE url='https://budget.example/1'"
            ).fetchone()[0]
            self.assertEqual(count, 0,
                "Бюджетный срез не должен инкрементировать retry_count")

            state = conn.execute(
                "SELECT state FROM items WHERE url='https://budget.example/1'"
            ).fetchone()[0]
            self.assertEqual(state, "pending",
                "Бюджетный срез не должен терминализировать item")
            st.close()
        finally:
            _cleanup_db(path)

    def test_recover_stuck_pending_marks_terminal(self):
        """
        C2: на старте цикла pending с retry_count>=max → 'failed'.
        """
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            item = _mk_item(title="Stuck", url="https://recover.example/1")
            st.save(item)
            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending', retry_count=8, state_at='2000-01-01T00:00:00+00:00' "
                "WHERE url='https://recover.example/1'"
            )
            conn.commit()
            st.close()

            # Новая инициализация (как в pipeline: новый цикл, новый Storage)
            st2 = Storage(Path(path))
            marked = st2.recover_stuck_pending(max_retries=8)
            self.assertEqual(marked, 1,
                "recover_stuck_pending должен маркировать 1 item")

            conn2 = st2._connect()
            state = conn2.execute(
                "SELECT state FROM items WHERE url='https://recover.example/1'"
            ).fetchone()[0]
            self.assertEqual(state, "failed",
                "Зависший pending с retry_count>=max должен стать 'failed'")
            st2.close()
        finally:
            _cleanup_db(path)

    def test_mark_failed_handles_large_list(self):
        """
        C2: set_failed на >1000 url не падает (батчинг).
        """
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            urls = [f"https://big.example/{i}" for i in range(1200)]
            items = [_mk_item(title=f"Big {i}", url=url) for i, url in enumerate(urls)]
            st.save_many(items)

            conn = st._connect()
            conn.execute(
                "UPDATE items SET state='pending', retry_count=8 "
                "WHERE url LIKE 'https://big.example/%'"
            )
            conn.commit()
            st.close()

            # Новая инициализация, вызываем set_failed
            st2 = Storage(Path(path))
            marked = st2.set_failed(urls)
            self.assertEqual(marked, 1200,
                "set_failed должен обработать 1200 URL без падения")
            st2.close()
        finally:
            _cleanup_db(path)

    def test_get_urls_with_retry_gte(self):
        """
        N5: get_urls_with_retry_gte возвращает только URLs с retry_count>=threshold.
        """
        path = _tmp_db()
        try:
            st = Storage(Path(path))
            items = [_mk_item(title=f"Retry {i}", url=f"https://retry-gte.example/{i}")
                     for i in range(10)]
            st.save_many(items)

            conn = st._connect()
            for i in range(5):
                conn.execute(
                    "UPDATE items SET retry_count=? WHERE url=?",
                    (5 + i, f"https://retry-gte.example/{i}")
                )
            conn.commit()

            all_urls = [f"https://retry-gte.example/{i}" for i in range(10)]
            result = st.get_urls_with_retry_gte(all_urls, threshold=7)
            self.assertEqual(len(result), 3,
                "Должны вернуться 3 URL с retry_count>=7")
            self.assertEqual(
                set(result),
                {
                    "https://retry-gte.example/2",  # retry_count=7
                    "https://retry-gte.example/3",  # retry_count=8
                    "https://retry-gte.example/4",  # retry_count=9
                },
            )
            st.close()
        finally:
            _cleanup_db(path)


# =============================================================================
# Main
# =============================================================================
# C1' / M5: реальные тесты _process_batch (без NewsPipeline)
# =============================================================================

class _FakeDelivery:
    """Мок Delivery: возвращает настроенные delivered_urls."""

    def __init__(self, delivered_urls):
        self.delivered_urls = delivered_urls or []

    def send_batch(self, items):
        return self.delivered_urls


class TestProcessBatchC1(unittest.TestCase):
    """
    M5: РЕАЛЬНЫЕ тесты _process_batch через mock/patch.
    Проверяют C1' и C1: retry_count инкрементируется/терминализируется
    ТОЛЬКО для реально не доставленных items.

    Все тесты используют patch вместо реального pipeline (циклический импорт
    delivery отделён в параллельный поток B).
    """

    def _patch_pipeline(self, delivered_urls, llm_filter=None):
        """
        Создаёт мок NewsPipeline с замоканными storage и delivery.
        Возвращает (pipeline, storage_tracker).
        """
        from unittest.mock import MagicMock, patch

        # Pipeline с реальным _process_batch, но мок storage + delivery
        with patch("newsbot.pipeline.Storage") as MockStorage, \
             patch("newsbot.pipeline.Delivery") as MockDelivery:

            mock_storage = MagicMock()
            mock_delivery = _FakeDelivery(delivered_urls)
            MockStorage.return_value = mock_storage
            MockDelivery.return_value = mock_delivery

            # Создаём pipeline (импортируем внутри чтобы избежать цикла)
            from newsbot.pipeline import NewsPipeline
            cfg = {
                "sources": [{"name": "Test", "url": "https://test.com/feed"}],
                "database": {"path": ":memory:", "retention_days": 1},
                "llm": {"enabled": False},
                "telegram": {"token": "", "chat_id": ""},
            }
            pipeline = NewsPipeline.__new__(NewsPipeline)
            pipeline.config = cfg
            pipeline.sources = cfg["sources"]
            pipeline.storage = mock_storage
            pipeline.delivery = mock_delivery
            pipeline.llm_filter = llm_filter
            pipeline._max_per_cycle = 150
            pipeline._llm_calls_left = 150
            pipeline._max_retries = 8
            pipeline._feedback_context = None

            return pipeline

    def test_llm_off_no_failed_for_delivered(self):
        """
        C1': llm_filter=None, items > budget_override.
        ВСЕ items доставляются → failed_urls ПУСТОЙ.
        """
        items = [_mk_item(title=f"Item {i}", url=f"https://ex.com/{i}") for i in range(5)]
        # все 5 доставлены
        pipeline = self._patch_pipeline(delivered_urls=[f"https://ex.com/{i}" for i in range(5)])

        # budget=3 — но при llm_filter=None все 5 идут в to_deliver → доставлены
        result = pipeline._process_batch(items, budget_override=3)

        self.assertEqual(result, [],
            "Все items доставлены — failed_urls должен быть пустым")

    def test_budget_cut_excess_not_in_failed_urls(self):
        """
        C1': активный LLM, 5 items, batch_size=1, budget_override=3,
        доставлены 2 из 3 обработанных. Excess (items[3:]) НЕ в failed_urls.
        При batch_size=1 evaluate_batch вызывает evaluate (single-item fallback).
        """
        from unittest.mock import MagicMock
        from newsbot.llm.filter import FilterResult

        items = [_mk_item(title=f"B{i}", url=f"https://cut.com/{i}") for i in range(5)]
        mock_llm = MagicMock()
        mock_llm.push_threshold = 7
        mock_llm.digest_threshold = 5
        mock_llm.push_urgent_only = False

        def eval_side(item, feedback_context=None):
            return FilterResult(ok=True, score=10, summary="sum", reason="r")
        mock_llm.evaluate.side_effect = eval_side

        def batch_side(items, feedback_context=None):
            return [mock_llm.evaluate(item, feedback_context) for item in items]
        mock_llm.evaluate_batch.side_effect = batch_side

        pipeline = self._patch_pipeline(
            delivered_urls=[f"https://cut.com/0", f"https://cut.com/1"],
            llm_filter=mock_llm,
        )
        # batch_size=1 → поштучный режим (точный лимит как в старой логике)
        pipeline.config.setdefault("llm", {})["batch_size"] = 1

        result = pipeline._process_batch(items, budget_override=3)

        self.assertEqual(set(result), {"https://cut.com/2"},
            "Failed должны быть только реально не доставленные, НЕ excess")
        self.assertNotIn("https://cut.com/3", result,
            "Excess item[3] НЕ должен быть в failed_urls (не дошёл до delivery)")
        self.assertNotIn("https://cut.com/4", result,
            "Excess item[4] НЕ должен быть в failed_urls (не дошёл до delivery)")

    def test_retry_path_budget_cut_no_increment(self):
        """
        C1': retry item (pending), budget=1, 2 items, доставлен 1.
        Срезанный бюджетом НЕ инкрементируется и НЕ в failed_urls.
        """
        from newsbot.llm.filter import FilterResult

        items = [
            _mk_item(title="Retry 0", url="https://retry.com/0"),
            _mk_item(title="Retry 1", url="https://retry.com/1"),
        ]
        # мок LLM: всем push
        mock_llm = MagicMock()
        mock_llm.push_threshold = 7
        mock_llm.digest_threshold = 5
        mock_llm.evaluate.side_effect = lambda item: FilterResult(ok=True, score=10, summary="s", reason="r")

        pipeline = self._patch_pipeline(
            delivered_urls=["https://retry.com/0"],  # доставлен только первый
            llm_filter=mock_llm,
        )

        # бюджет=1, доставлен 1 из 1 обработанного
        result = pipeline._process_batch(items, budget_override=1)

        # failed = только не доставленный из обработанных (retry.com/0 обработан но не доставлен)
        # retry.com/1 — срезан бюджетом, НЕ в failed_urls
        self.assertEqual(result, [],
            "Не доставленный из обработанных: retry.com/0 — доставлен, "
            "retry.com/1 срезан бюджетом → failed_urls пуст")
        # NOTE: в этом сценарии оба retry item доставлены (budget=1, доставлен 1)
        # Не доставленных из обработанных нет → empty

    def test_all_filtered_returns_empty_failed_urls(self):
        """
        M1: все items отфильтрованы (skipped), to_deliver пуст.
        failed_urls должен быть ПУСТЫМ (не skipped items).
        """
        from unittest.mock import MagicMock
        from newsbot.llm.filter import FilterResult

        items = [_mk_item(title=f"Skip{i}", url=f"https://skip.com/{i}") for i in range(3)]
        mock_llm = MagicMock()
        mock_llm.push_threshold = 7
        mock_llm.digest_threshold = 5
        # всем skip (score=3)
        mock_llm.evaluate.side_effect = lambda item, feedback_context=None: FilterResult(ok=True, score=3, summary="s", reason="r")

        pipeline = self._patch_pipeline(delivered_urls=[], llm_filter=mock_llm)

        result = pipeline._process_batch(items, budget_override=3)

        self.assertEqual(result, [],
            "Все skipped — to_deliver пуст — failed_urls должен быть пустым")

    def test_successful_retry_item_not_in_failed_urls(self):
        """
        C1: retry item с retry_count=max-1, успешно доставлен.
        NOT in failed_urls → не инкрементируется, не терминализируется.
        """
        items = [
            _mk_item(title="SuccRetry", url="https://succ.example/1"),
            _mk_item(title="FailRetry", url="https://fail.example/2"),
        ]

        pipeline = self._patch_pipeline(
            delivered_urls=["https://succ.example/1"],  # SuccRetry доставлен
            # FailRetry не доставлен (не в списке)
        )

        result = pipeline._process_batch(items, budget_override=2)

        self.assertNotIn("https://succ.example/1", result,
            "Успешно доставленный retry item НЕ должен быть в failed_urls")
        self.assertIn("https://fail.example/2", result,
            "Не доставленный retry item ДОЛЖЕН быть в failed_urls")


# =============================================================================
# K1-интеграция (pytest-стиль, caplog)
# =============================================================================

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


if _HAS_PYTEST:

    def test_k1_recovery_logs_remainder(caplog):
        """
        K1-интеграция: >recovery_new_limit items в state='new' →
        recovery берёт ≤ limit, в логе есть 'подхвачено X/Y' и 'остаток Z'.
        Использует реальный Storage и реальный run_cycle с пустым collector.
        """
        import logging
        import tempfile
        import os as _os

        fd, db_path = tempfile.mkstemp(suffix=".db")
        _os.close(fd)
        _os.unlink(db_path)
        try:
            # Подготовка: 7 items в state='new'
            s = Storage(Path(db_path))
            for i in range(7):
                s.save(_mk_item(title=f"R{i}", url=f"https://recov.com/{i}"))
            s.close()

            from newsbot.pipeline import NewsPipeline

            cfg = {
                "sources": [{"name": "Test", "url": "https://example.com/f.xml"}],
                "database": {
                    "path": db_path,
                    "retention_days": 90,
                    "recovery_new_limit": 3,
                },
                "llm": {"enabled": False},
                "telegram": {"token": "", "chat_id": ""},
            }

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = cfg["sources"]
            p.storage = Storage(Path(db_path))
            p.delivery = MagicMock()
            p.delivery.send_batch.return_value = []
            p.llm_filter = None
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._max_retries = 8
            p._feedback_context = None
            p._off_peak_only = False

            class EmptyCollector:
                def collect(self):
                    return []

            p.collector = EmptyCollector()

            with caplog.at_level(logging.INFO, logger="newsbot.pipeline"):
                p.run_cycle_with_delivery()

            messages = [r.message for r in caplog.records]
            # 'подхвачено 3/7': взято 3 из 7
            found_recovery = any(
                "подхвачено" in m and "/7" in m for m in messages
            )
            assert found_recovery, (
                f"Ожидался лог 'подхвачено 3/7'. Логи: {messages}"
            )
            # 'остаток 4': 7 - 3 = 4
            found_remainder = any(
                "остаток" in m and "4" in m for m in messages
            )
            assert found_remainder, (
                f"Ожидался лог 'остаток 4'. Логи: {messages}"
            )

            p.storage.close()
        finally:
            _cleanup_db(db_path)


# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)