"""
Тесты storage.py — покрывают tags-поля и URL-normalization дедуп (v14).
"""
import json
import tempfile
from pathlib import Path

import pytest

from newsbot.storage import Storage
from newsbot.collector import NewsItem
from newsbot.urlnorm import normalize_url


@pytest.fixture
def tmp_db(tmp_path):
    """Временная БД для каждого теста."""
    db_path = tmp_path / "test.db"
    storage = Storage(db_path=db_path)
    yield storage
    storage.close()


class TestTagsField:
    """Задача 1.3: тесты на поле tags."""

    def test_save_and_read_tags(self, tmp_db):
        """Сохранить item с tags, прочитать — tags совпадают."""
        item = NewsItem(
            source="TestSource",
            url="https://example.com/tags-test",
            title="Тест тегов",
            summary="Краткое описание",
            published="2025-01-01",
            tags=json.dumps(["News", "Broadway", "Reviews"]),
        )
        saved = tmp_db.save(item)
        assert saved, "Item должен быть сохранён"

        # Чтение через get_all_by_state
        items = tmp_db.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].tags == json.dumps(["News", "Broadway", "Reviews"])

    def test_save_without_tags(self, tmp_db):
        """Item без tags сохраняется и читается как None."""
        item = NewsItem(
            source="TestSource",
            url="https://example.com/no-tags",
            title="Без тегов",
            summary="Описание без тегов",
            published="2025-01-01",
            tags=None,
        )
        saved = tmp_db.save(item)
        assert saved

        items = tmp_db.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].tags is None

    def test_save_many_with_tags(self, tmp_db):
        """save_many: несколько items с разными tags."""
        items = [
            NewsItem(
                source="Source1",
                url=f"https://example.com/many-tags-{i}",
                title=f"Новость {i}",
                summary="Краткое описание",
                published="2025-01-01",
                tags=json.dumps([f"Tag{i}A", f"Tag{i}B"]),
            )
            for i in range(3)
        ]
        inserted = tmp_db.save_many(items)
        assert len(inserted) == 3

        read = tmp_db.get_all_by_state("new")
        assert len(read) == 3
        for i, ni in enumerate(read):
            expected = json.dumps([f"Tag{i}A", f"Tag{i}B"])
            assert ni.tags == expected, f"Item {i}: tags не совпали"

    def test_tags_in_deferred(self, tmp_db):
        """tags читаются из deferred-items."""
        item = NewsItem(
            source="DeferredSource",
            url="https://example.com/deferred-tags",
            title="Отложенная новость",
            summary="Описание",
            published="2025-01-01",
            tags=json.dumps(["DeferredTag"]),
        )
        tmp_db.save(item)
        # Переводим в deferred
        tmp_db.set_state(["https://example.com/deferred-tags"], "deferred")

        deferred = tmp_db.get_deferred_items()
        assert len(deferred) == 1
        assert deferred[0].tags == json.dumps(["DeferredTag"])

    def test_schema_has_tags_column(self, tmp_db):
        """Колонка tags существует в схеме."""
        conn = tmp_db._connect()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        assert "tags" in cols, "Колонка tags должна существовать в items"


class TestUrlNormalizationDedup:
    """
    URL-based дедупликация: нормализация склеивает utm/без utm,
    порядок параметров, fragment в одну запись (v14 миграция).
    """

    def test_utm_without_utm_collapsed(self, tmp_db):
        """Статья с utm и без — один и тот же ключ → дубликат."""
        url_bare = "https://www.aljazeera.com/news/politics/article"
        url_tracked = f"{url_bare}?utm_source=rss&traffic_source=rss&ref=newsletter"

        item1 = NewsItem(
            source="Al Jazeera",
            url=url_tracked,
            title="Test Article",
            summary="Summary",
            published="2025-10-23",
        )
        item2 = NewsItem(
            source="Al Jazeera 2",
            url=url_bare,
            title="Test Article",
            summary="Summary",
            published="2025-10-23",
        )

        # Первый сохраняется
        saved1 = tmp_db.save(item1)
        assert saved1, "Первый item должен быть сохранён"

        # Второй — дубликат (та же нормализованная ссылка)
        assert tmp_db.is_duplicate(item2), "item2 должен считаться дубликатом item1"
        saved2 = tmp_db.save(item2)
        assert not saved2, "Дубликат не должен вставляться"

    def test_param_order_not_matters(self, tmp_db):
        """Порядок query-параметров не влияет на дедуп."""
        u1 = "https://example.com/p?a=1&z=9&m=5"
        u2 = "https://example.com/p?m=5&a=1&z=9"

        item1 = NewsItem(source="S1", url=u1, title="T1", summary="X", published="2025-10-01")
        item2 = NewsItem(source="S1", url=u2, title="T2", summary="X", published="2025-10-01")

        tmp_db.save(item1)
        assert tmp_db.is_duplicate(item2)

    def test_fragment_ignored(self, tmp_db):
        """Fragment (#...) игнорируется при дедупе."""
        base = "https://example.com/article-42"
        url_frag = f"{base}#comments"

        item1 = NewsItem(source="S1", url=url_frag, title="T1", summary="X", published="2025-10-01")
        item2 = NewsItem(source="S1", url=base, title="T2", summary="X", published="2025-10-01")

        tmp_db.save(item1)
        assert tmp_db.is_duplicate(item2)

    def test_legit_params_preserved(self, tmp_db):
        """Легитимные параметры (p=, id=) сохраняются и используются в дедупе."""
        # Два разных id → разные нормализованные URL → не дубликаты
        item1 = NewsItem(
            source="Shop", url="https://shop.com/product?p=100", title="T1", summary="X", published="2025-10-01"
        )
        item2 = NewsItem(
            source="Shop", url="https://shop.com/product?p=200", title="T2", summary="X", published="2025-10-01"
        )

        tmp_db.save(item1)
        assert not tmp_db.is_duplicate(item2), "Разные легитимные параметры — разные URL"

    def test_utm_param_combinations_same_key(self, tmp_db):
        """Варианты utm_medium / utm_source склеиваются."""
        base = "https://news.site/article"

        item_bare = NewsItem(source="RSS", url=base, title="A", summary="X", published="2025-10-01")
        item_utm_email = NewsItem(
            source="Email", url=f"{base}?utm_source=email", title="B", summary="X", published="2025-10-01"
        )
        item_full_utm = NewsItem(
            source="Social",
            url=f"{base}?utm_source=twitter&utm_medium=post&fbclid=abc123",
            title="C",
            summary="X",
            published="2025-10-01",
        )

        tmp_db.save(item_bare)
        assert tmp_db.is_duplicate(item_utm_email)
        assert tmp_db.is_duplicate(item_full_utm)

    def test_case_preserved_in_norm_key(self, tmp_db):
        """
        nit-фикс: scheme+host lowercase (склейка), path case-sensitive (сохранение).
        HTTPS://EXAMPLE.COM/PATH и https://example.com/Path — РАЗНЫЕ нормализованные ключи
        (CMS может дать 404 при неправильном case пути).
        HTTPS://EXAMPLE.COM/PAGE и https://example.com/PAGE — один ключ.
        """
        # Host lower: HTTPS://WWW.EXAMPLE.COM/PAGE → https://www.example.com/PAGE
        url_host_diff = "HTTPS://WWW.EXAMPLE.COM/PAGE"
        url_host_same = "https://www.example.com/PAGE"
        assert normalize_url(url_host_diff) == normalize_url(url_host_same) == "https://www.example.com/PAGE"

        # Path case-sensitive: /PAGE ≠ /page
        url_page_upper = "https://www.example.com/PAGE"
        url_page_lower = "https://www.example.com/page"
        assert normalize_url(url_page_upper) == url_page_upper
        assert normalize_url(url_page_lower) == url_page_lower
        assert normalize_url(url_page_upper) != normalize_url(url_page_lower)

        item1 = NewsItem(
            source="S1", url=normalize_url(url_page_upper), title="T1", summary="X", published="2025-10-01"
        )
        item2 = NewsItem(
            source="S1", url=normalize_url(url_page_lower), title="T2", summary="X", published="2025-10-01"
        )

        tmp_db.save(item1)
        # Разный path → разные ключи → НЕ дубликаты
        assert not tmp_db.is_duplicate(item2), (
            "Different path case (/PAGE vs /page) should NOT be duplicates"
        )

    def test_url_norm_column_exists(self, tmp_db):
        """Колонка url_norm существует в схеме (v14)."""
        conn = tmp_db._connect()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        assert "url_norm" in cols, "Колонка url_norm должна существовать в items"

    def test_url_norm_index_created(self, tmp_db):
        """UNIQUE INDEX idx_items_url_norm существует."""
        conn = tmp_db._connect()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ).fetchall()
        index_names = {r[0] for r in rows}
        assert "idx_items_url_norm" in index_names, f"Индекс url_norm должен существовать: {index_names}"
        assert "idx_items_url" not in index_names, "Старый индекс url должен быть заменён"

    def test_trailing_slash_dedup(self, tmp_db):
        """URL с завершающим слэшом и без — дубликаты."""
        item1 = NewsItem(
            source="S1", url="https://example.com/page/", title="T1", summary="X", published="2025-10-01"
        )
        item2 = NewsItem(
            source="S1", url="https://example.com/page", title="T2", summary="X", published="2025-10-01"
        )

        tmp_db.save(item1)
        assert tmp_db.is_duplicate(item2)

    def test_utm_collector_ingest_normalized(self, tmp_db):
        """
        При сохранении из collector (URL уже нормализован) —
        запись сохраняется с url_norm.
        """
        # Симулируем collector-путь: URL нормализован ДО NewsItem
        raw_url = "https://EXAMPLE.COM/ARTICLE?utm_source=rss&fbclid=abc#section"
        norm_url = normalize_url(raw_url)

        item = NewsItem(
            source="RSS",
            url=norm_url,   # collector already normalized
            title="Article",
            summary="X",
            published="2025-10-01",
        )

        tmp_db.save(item)
        conn = tmp_db._connect()
        row = conn.execute(
            "SELECT url, url_norm FROM items WHERE source = ?", ("RSS",)
        ).fetchone()
        assert row is not None
        assert row["url"] == norm_url
        assert row["url_norm"] == norm_url
        assert "utm_source" not in row["url_norm"]


class TestV14Migration:
    """
    MAJOR 3-тесты: миграция v13→v14 (url_norm, keyset backfill, safe index swap).
    """

    def test_migration_keyset_pagination_no_nulls(self, tmp_path):
        """
        После миграции v14 НЕ должно остаться строк с url_norm IS NULL.
        (keyset-пагинация вместо open-cursor SELECT → SQLite undefined behaviour.)
        """
        # Создаём свежую v14 БД и сохраняем несколько items
        db_path = tmp_path / "v14_from_fresh.db"
        storage = Storage(db_path=db_path)
        conn = storage._connect()

        now = "2025-10-01T00:00:00"
        # Вставляем items с уникальными URL (разные страницы)
        for i in range(50):
            url = (f"https://example.com/item{i}?utm_source=feed"
                   if i % 2 == 0 else f"https://example.com/item{i}")
            conn.execute(
                "INSERT INTO items (url, state, published, title, source, summary, "
                "fetched_at, state_at, url_norm) "
                "VALUES (?, 'new', '2025-10-01', ?, 'Src', 'x', ?, ?, ?)",
                (url, f"Title {i}", now, now, url),  # url_norm уже заполнен
            )
        conn.commit()
        storage.close()

        # Подменяем user_version на 13 (имитируем v13 БД без url_norm)
        conn2 = Storage(db_path=db_path)._connect()
        conn2.execute("PRAGMA user_version = 13")
        conn2.execute("UPDATE items SET url_norm = NULL")  # симулируем v13 данные
        conn2.commit()
        conn2.close()

        # Закрываем и пересоздаём — триггер миграции
        storage2 = Storage(db_path=db_path)
        conn3 = storage2._connect()
        version = conn3.execute("PRAGMA user_version").fetchone()[0]
        assert version == 14, f"user_version должен быть 14, got {version}"

        # Проверяем: нет NULL в url_norm
        nulls = conn3.execute(
            "SELECT COUNT(*) FROM items WHERE url_norm IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"После миграции не должно быть url_norm IS NULL, found {nulls}"

        storage2.close()

    def test_migration_conflict_raises_integrity_error(self, tmp_path):
        """
        При конфликте url_norm (IntegrityError при CREATE UNIQUE INDEX)
        миграция поднимает исключение, user_version не меняется.
        Старый индекс idx_items_url сохранён (не дропается до успешного CREATE).

        Подход: создаём БД напрямую через sqlite3.connect (без Storage(),
        чтобы избежать auto-migration) и вставляем конфликтующие строки.
        Две записи с url_norm = 'https://a.com/' (пустой path нормализуется в '/',
        так что 'https://a.com?utm_source=rss' и 'https://a.com/' дают один url_norm).
        """
        import sqlite3

        db_path = tmp_path / "conflict_test.db"

        # Создаём v14-совместимую схему через sqlite3.connect напрямую
        # (минуя Storage.__init__ который вызывает миграцию).
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version = 14")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                url_norm TEXT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                published TEXT,
                tags TEXT,
                cluster_key TEXT,
                lang TEXT,
                fetched_at TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'new',
                state_at TEXT NOT NULL,
                llm_score INTEGER,
                llm_summary TEXT,
                llm_reason TEXT,
                llm_urgent INTEGER,
                filtered_at TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_items_url ON items(url);
            CREATE TABLE IF NOT EXISTS seen_titles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title_hash TEXT NOT NULL,
                source TEXT,
                fetched_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                title_normalized TEXT
            );
        """)
        now = "2025-10-01T00:00:00"
        # Вставляем две строки с одинаковым url_norm
        conn.execute(
            "INSERT INTO items (source, url, title, summary, published, "
            "fetched_at, state, state_at, url_norm) "
            "VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            ("S1", "https://a.com?utm_source=rss", "Title 1", "x", "2025-10-01", now, now, "https://a.com/"),
        )
        conn.execute(
            "INSERT INTO items (source, url, title, summary, published, "
            "fetched_at, state, state_at, url_norm) "
            "VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            ("S2", "https://a.com/", "Title 2", "x", "2025-10-01", now, now, "https://a.com/"),
        )
        conn.commit()
        conn.close()

        # Подменяем user_version на 13 (имитируем v13 БД)
        conn2 = sqlite3.connect(str(db_path))
        conn2.execute("PRAGMA user_version = 13")
        conn2.close()

        # Миграция должна поднять IntegrityError
        integrity_error_raised = False
        try:
            Storage(db_path=db_path)
        except sqlite3.IntegrityError:
            integrity_error_raised = True

        assert integrity_error_raised, (
            "Migration should raise IntegrityError when duplicate url_norm exists"
        )

        # user_version осталась 13, старый индекс сохранён
        conn_check = sqlite3.connect(str(db_path))
        version = conn_check.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            r[0] for r in conn_check.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn_check.close()

        assert version == 13, (
            f"user_version должен остаться 13 после IntegrityError, got {version}"
        )
        assert "idx_items_url" in indexes, (
            f"Старый индекс idx_items_url должен быть сохранён: {indexes}"
        )
        assert "idx_items_url_norm" not in indexes, (
            "Новый индекс НЕ должен быть создан при IntegrityError"
        )

    def test_migration_safe_dedup_on_conflict(self, tmp_path):
        """
        Миграция v13→v14 на БД с двумя записями нормализованно-одинакового URL.
        После миграции (с ручной дедупликацией) — unique-индекс существует,
        старый idx_items_url удалён, дубль безопасно обработан.
        """
        import sqlite3

        db_path = tmp_path / "dedup_conflict.db"

        # Создаём БД напрямую (минуя Storage.__init__)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version = 14")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                url_norm TEXT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                published TEXT,
                tags TEXT,
                cluster_key TEXT,
                lang TEXT,
                fetched_at TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'new',
                state_at TEXT NOT NULL,
                llm_score INTEGER,
                llm_summary TEXT,
                llm_reason TEXT,
                llm_urgent INTEGER,
                filtered_at TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_items_url ON items(url);
            CREATE TABLE IF NOT EXISTS seen_titles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title_hash TEXT NOT NULL,
                source TEXT,
                fetched_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                title_normalized TEXT
            );
        """)
        now = "2025-10-01T00:00:00"
        conn.execute(
            "INSERT INTO items (source, url, title, summary, published, "
            "fetched_at, state, state_at, url_norm) "
            "VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            ("S1", "https://b.com?utm_source=rss", "T1", "x", "2025-10-01", now, now, "https://b.com/"),
        )
        conn.execute(
            "INSERT INTO items (source, url, title, summary, published, "
            "fetched_at, state, state_at, url_norm) "
            "VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            ("S2", "https://b.com/", "T2", "x", "2025-10-01", now, now, "https://b.com/"),
        )
        conn.commit()
        conn.close()

        # Устанавливаем v13 — миграция выбросит IntegrityError
        conn2 = sqlite3.connect(str(db_path))
        conn2.execute("PRAGMA user_version = 13")
        conn2.close()

        error_raised = False
        try:
            Storage(db_path=db_path)
        except sqlite3.IntegrityError:
            error_raised = True
        assert error_raised, "Migration should fail on duplicate url_norm"

        # Дедуплицируем вручную (оставляем MIN rowid)
        conn3 = sqlite3.connect(str(db_path))
        conn3.execute(
            "DELETE FROM items WHERE rowid NOT IN ("
            "  SELECT MIN(rowid) FROM items GROUP BY url_norm"
            ")"
        )
        conn3.commit()
        conn3.close()

        # Повторная миграция — теперь успешна
        storage2 = Storage(db_path=db_path)
        conn4 = storage2._connect()
        version = conn4.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            r[0] for r in conn4.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        nulls = conn4.execute(
            "SELECT COUNT(*) FROM items WHERE url_norm IS NULL"
        ).fetchone()[0]
        conn4.close()
        storage2.close()

        assert version == 14, f"user_version должен быть 14, got {version}"
        assert "idx_items_url_norm" in indexes, (
            f"idx_items_url_norm должен существовать: {indexes}"
        )
        assert "idx_items_url" not in indexes, (
            f"Старый idx_items_url должен быть удалён: {indexes}"
        )
        assert nulls == 0, f"Не должно быть url_norm IS NULL, found {nulls}"

    def test_migration_idempotent(self, tmp_path):
        """
        Миграция v13→v14 идемпотентна: повторный вызов не ломает БД.
        """
        db_path = tmp_path / "idempotent_test.db"

        # Создаём v13 БД
        storage = Storage(db_path=db_path)
        conn = storage._connect()
        conn.execute("PRAGMA user_version = 13")
        now = "2025-10-01T00:00:00"
        conn.execute(
            "INSERT INTO items (url, state, published, title, source, summary, "
            "fetched_at, state_at, url_norm) "
            "VALUES (?, 'new', '2025-10-01', ?, 'Src', 'x', ?, ?, ?)",
            ("https://c.com/p1", "Title", now, now, "https://c.com/p1"),
        )
        conn.commit()
        storage.close()

        # Первая миграция (v13→v14)
        storage1 = Storage(db_path=db_path)
        conn1 = storage1._connect()
        v1 = conn1.execute("PRAGMA user_version").fetchone()[0]
        storage1.close()

        # Повторный вызов — Storage() не меняет schema, но убеждаемся что не падает
        storage2 = Storage(db_path=db_path)
        conn2 = storage2._connect()
        v2 = conn2.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            r[0] for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn2.close()
        storage2.close()

        assert v1 == 14, f"Первая миграция: user_version={v1}"
        assert v2 == 14, f"Повторный вызов: user_version={v2}"
        assert "idx_items_url_norm" in indexes, (
            f"Индекс после повторного вызова: {indexes}"
        )