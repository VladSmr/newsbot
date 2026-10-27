"""
test_storage_hardening.py — фиксы M1, M2, M3, Minor#3, Minor#4, MAJOR 4.

M1  — индекс idx_seen_fetched и idx_items_state_state_at (v11).
M2  — save/save_many логируют на уровне error, различают дедуп и реальную ошибку.
M3  — recover_stuck_pending снимает терминальные БЕЗ возрастного фильтра.
Minor#3 — TTL test: использовал фиктивный title_hash, теперь реальный title_normalized().
Minor#4 — UNIQUE(url, kind) на feedback_targets, INSERT OR IGNORE в get_or_create_*.
MAJOR 4 — тесты C: indexes, downgrade guard, sliding TTL, NULL fallback,
         terminal items between windows.
"""
import logging
import sqlite3
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from datetime import datetime, timedelta, timezone

from newsbot.pipeline import NewsPipeline
from newsbot.storage import Storage
from newsbot.collector import NewsItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _item(title="Test Title", url="https://example.com/1", source="test"):
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary="Test summary",
        published="2024-01-01",
        published_parsed=None,
        fetched_at=None,
    )


def _read_indexes(db_path):
    """Читает список имён индексов из БД (без row_factory, tuples)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    indexes = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()}
    conn.close()
    return indexes


# ---------------------------------------------------------------------------
# M1: индексы (v11)
# ---------------------------------------------------------------------------

def test_migration_v11_creates_indexes(tmp_path):
    """
    При инициализации БД (с нуля) SCHEMA_VERSION=11, user_version=11,
    индексы idx_seen_fetched и idx_items_state_state_at существуют.
    """
    db_path = tmp_path / "test_v11.db"
    s = Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    uv = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert uv == Storage.SCHEMA_VERSION, f"expected user_version={Storage.SCHEMA_VERSION}, got {uv}"

    indexes = _read_indexes(db_path)
    assert "idx_seen_fetched" in indexes, \
        f"idx_seen_fetched missing from indexes: {indexes}"
    assert "idx_items_state_state_at" in indexes, \
        f"idx_items_state_state_at missing from indexes: {indexes}"


def test_migration_v11_upgrades_existing_db(tmp_path):
    """
    Миграция v11 из v10: поднимает user_version и создаёт индексы
    на существующей БД.
    """
    db_path = tmp_path / "legacy.db"

    # Создаём минимальную БД v10 с needed-by-v9 state.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY, source TEXT, url TEXT UNIQUE,
            title TEXT, summary TEXT, published TEXT,
            fetched_at TEXT, state TEXT, state_at TEXT,
            llm_score INTEGER, llm_summary TEXT, llm_reason TEXT,
            llm_urgent INTEGER, filtered_at TEXT, retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY, title_hash TEXT, source TEXT,
            fetched_at TEXT, expires_at TEXT, title_normalized TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    conn.execute("""
        CREATE TABLE feedback_targets (
            id INTEGER PRIMARY KEY, url TEXT, source TEXT, title TEXT,
            summary TEXT, llm_score INTEGER, kind TEXT, chat_id TEXT,
            message_id INTEGER, created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY, target_id INTEGER, url TEXT,
            source TEXT, title TEXT, summary TEXT, llm_score INTEGER,
            verdict TEXT, chat_id TEXT, message_id INTEGER,
            created_at TEXT, updated_at TEXT
        )
    """)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()
    conn.close()

    # Открываем Storage — должен накатить v11.
    s = Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    uv = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert uv == Storage.SCHEMA_VERSION

    indexes = _read_indexes(db_path)
    assert "idx_seen_fetched" in indexes, f"idx_seen_fetched missing: {indexes}"
    assert "idx_items_state_state_at" in indexes


def test_migration_v11_idempotent(tmp_path):
    """Повторный запуск Storage на v11 не падает и не меняет user_version."""
    db_path = tmp_path / "idempotent.db"
    Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION
    conn.close()

    # Второй Storage на той же БД — без ошибок.
    Storage(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION
    conn.close()


# ---------------------------------------------------------------------------
# M2: логирование ошибок save / save_many
# ---------------------------------------------------------------------------

def test_save_logs_error_on_real_db_exception(tmp_path, caplog):
    """
    save() вызывает logger.exception (уровень ERROR) при реальной ошибке БД.
    M2: до фикса было logger.warning — после фикса logger.exception.
    """
    db_path = tmp_path / "save_error.db"
    s = Storage(db_path)

    item = _item(url="https://ok.example/1", title="OK")
    s.save(item)

    # Патчим Storage._connect — возвращаем соединение, которое бросает.
    class BadConn:
        row_factory = sqlite3.Row
        def execute(self, sql, *a, **kw):
            if "INSERT OR IGNORE INTO items" in sql:
                raise sqlite3.OperationalError("simulated I/O error")
            return self._real.execute(sql, *a, **kw)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def commit(self):
            pass

    real_conn = sqlite3.connect(str(db_path))
    real_conn.row_factory = sqlite3.Row
    bad = BadConn()
    bad._real = real_conn

    with mock.patch.object(s, "_connect", return_value=bad):
        item2 = _item(url="https://ok.example/2", title="Bad")
        result = s.save(item2)

    real_conn.close()

    assert result is False, "save() должен вернуть False при ошибке"
    error_records = [r for r in caplog.record_tuples if r[1] >= logging.ERROR]
    assert error_records, (
        f"M2: save() должна логировать на уровне ERROR, "
        f"логов ERROR нет. Записи: {caplog.record_tuples}"
    )


def test_save_ignores_duplicate_not_logged_as_error(tmp_path, caplog):
    """
    save() при INSERT OR IGNORE (дедуп) НЕ выбрасывает исключение —
    возвращает False без логирования на error.
    """
    db_path = tmp_path / "dedup_ok.db"
    s = Storage(db_path)

    item = _item(url="https://dup.example/1", title="Dup")
    r1 = s.save(item)
    assert r1 is True

    r2 = s.save(item)
    assert r2 is False

    error_records = [r for r in caplog.record_tuples if r[1] >= logging.ERROR]
    assert not error_records, f"M2: дедуп не должен логироваться на ERROR: {error_records}"


def test_save_many_logs_error_on_real_db_exception(tmp_path, caplog):
    """save_many() логирует на ERROR при ошибке БД."""
    db_path = tmp_path / "save_many_error.db"
    s = Storage(db_path)

    class BadConn:
        row_factory = sqlite3.Row
        def execute(self, sql, *a, **kw):
            if "INSERT OR IGNORE INTO items" in sql:
                raise sqlite3.OperationalError("simulated I/O error")
            return self._real.execute(sql, *a, **kw)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def commit(self):
            pass

    real_conn = sqlite3.connect(str(db_path))
    real_conn.row_factory = sqlite3.Row
    bad = BadConn()
    bad._real = real_conn

    items = [_item(url=f"https://bad.example/{i}", title=f"Bad {i}") for i in range(3)]
    with mock.patch.object(s, "_connect", return_value=bad):
        result = s.save_many(items)

    real_conn.close()

    assert result == [], "save_many() должна вернуть [] при ошибке"
    error_records = [r for r in caplog.record_tuples if r[1] >= logging.ERROR]
    assert error_records, (
        f"M2: save_many() должна логировать на уровне ERROR, "
        f"логов ERROR нет. Записи: {caplog.record_tuples}"
    )


# ---------------------------------------------------------------------------
# M3: recover_stuck_pending без возрастного фильтра для терминальных
# ---------------------------------------------------------------------------

def test_recover_stuck_pending_ignores_age_for_terminal(tmp_path):
    """
    recover_stuck_pending снимает pending с retry_count>=max НЕЗАВИСИМО от возраста.
    Это M3: до фикса условие AND state_at < cutoff отсекало «молодые» терминальные.
    """
    db_path = tmp_path / "m3_recover.db"
    s = Storage(db_path)

    item = _item(url="https://stuck.example/1", title="Stuck")
    s.save(item)
    s.set_state(["https://stuck.example/1"], "pending")

    # Устанавливаем retry_count >= max_retries (8) напрямую в БД.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE items SET retry_count = 8 WHERE url = ?",
        ("https://stuck.example/1",),
    )
    conn.commit()
    conn.close()

    count_before = s.count_by_state("pending")
    recovered = s.recover_stuck_pending(max_retries=8)
    count_after = s.count_by_state("pending")

    assert count_before == 1
    assert recovered == 1, (
        f"M3: recover_stuck_pending должен снять 1 терминальный (retry_count>=8), "
        f"даже если state_at < 60 с. recovered={recovered}"
    )
    assert count_after == 0


def test_get_stuck_pending_respects_60s_threshold(tmp_path):
    """
    get_stuck_pending возвращает pending старше 60 с с retry_count < max.
    Подтверждает, что порог 60 с не убран (защита «в работе»).
    """
    db_path = tmp_path / "m3_get_stuck.db"
    s = Storage(db_path)

    item = _item(url="https://young.example/1", title="Young Pending")
    s.save(item)
    s.set_state(["https://young.example/1"], "pending")

    stuck = s.get_stuck_pending(max_retries=8)
    assert len(stuck) == 0, (
        f"get_stuck_pending не должна возвращать молодые (<60 с) pending. "
        f"Вернула: {len(stuck)}"
    )


# ---------------------------------------------------------------------------
# Minor#4: UNIQUE(url, kind) на feedback_targets
# ---------------------------------------------------------------------------

def test_feedback_targets_unique_constraint_prevents_duplicate(tmp_path):
    """
    Minor#4: UNIQUE(url, kind) — повторный вызов get_or_create_feedback_target_from_item
    для того же (url, kind) возвращает тот же id.
    """
    db_path = tmp_path / "minor4_unique.db"
    s = Storage(db_path)

    item = _item(url="https://fb.example/1", title="FB")

    id1 = s.get_or_create_feedback_target_from_item(item, kind="push")
    id2 = s.get_or_create_feedback_target_from_item(item, kind="push")

    assert id1 == id2, (
        f"Minor#4: повторный get_or_create_feedback_target_from_item "
        f"должен вернуть тот же id. id1={id1}, id2={id2}"
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    count = conn.execute(
        "SELECT COUNT(*) FROM feedback_targets WHERE url = ? AND kind = ?",
        ("https://fb.example/1", "push"),
    ).fetchone()[0]
    conn.close()

    assert count == 1, f"Minor#4: ожидалась 1 запись, найдено {count}"


def test_feedback_targets_different_kinds_allowed(tmp_path):
    """UNIQUE(url, kind) разрешает разные kind для одного URL."""
    db_path = tmp_path / "minor4_diff_kind.db"
    s = Storage(db_path)

    item = _item(url="https://multi.example/1", title="Multi")

    id_push = s.get_or_create_feedback_target_from_item(item, kind="push")
    id_digest = s.get_or_create_feedback_target_from_item(item, kind="digest")

    assert id_push != id_digest

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    count = conn.execute(
        "SELECT COUNT(*) FROM feedback_targets WHERE url = ?",
        ("https://multi.example/1",),
    ).fetchone()[0]
    conn.close()

    assert count == 2, f"ожидалось 2 записи (push+digest), найдено {count}"


def test_feedback_targets_retry_returns_same_id(tmp_path):
    """Minor#4: 3 retry-вызова возвращают тот же id."""
    db_path = tmp_path / "minor4_concurrent.db"
    s = Storage(db_path)
    item = _item(url="https://concurrent.example/1", title="Concurrent")

    ids = [
        s.get_or_create_feedback_target_from_item(item, kind="fallback")
        for _ in range(3)
    ]

    assert ids[0] == ids[1] == ids[2], f"retry ids должны совпадать: {ids}"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    count = conn.execute("SELECT COUNT(*) FROM feedback_targets").fetchone()[0]
    conn.close()

    assert count == 1, f"Minor#4: ожидалась 1 запись, найдено {count}"


def test_feedback_targets_v11_migration_dedups_existing(tmp_path):
    """
    v11-миграция дедуплицирует существующие дубли (url, kind) перед
    созданием UNIQUE INDEX.
    """
    db_path = tmp_path / "minor4_legacy_dedup.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY, source TEXT, url TEXT UNIQUE,
            title TEXT, summary TEXT, published TEXT,
            fetched_at TEXT, state TEXT, state_at TEXT,
            llm_score INTEGER, llm_summary TEXT, llm_reason TEXT,
            llm_urgent INTEGER, filtered_at TEXT, retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY, title_hash TEXT, source TEXT,
            fetched_at TEXT, expires_at TEXT, title_normalized TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    conn.execute("""
        CREATE TABLE feedback_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, source TEXT, title TEXT,
            summary TEXT, llm_score INTEGER, kind TEXT NOT NULL,
            chat_id TEXT, message_id INTEGER, created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY, target_id INTEGER, url TEXT,
            source TEXT, title TEXT, summary TEXT, llm_score INTEGER,
            verdict TEXT, chat_id TEXT, message_id INTEGER,
            created_at TEXT, updated_at TEXT
        )
    """)
    # Вставляем дубли.
    for i in range(3):
        conn.execute(
            "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
            ("https://dup.example/1", "push", f"2024-01-0{i+1}T00:00:00"),
        )
    conn.execute("PRAGMA user_version = 10")
    conn.commit()
    conn.close()

    # Открываем Storage — v11-миграция дедуплицирует.
    s = Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    count = conn.execute(
        "SELECT COUNT(*) FROM feedback_targets WHERE url = ? AND kind = ?",
        ("https://dup.example/1", "push"),
    ).fetchone()[0]
    has_index = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        ("idx_feedback_targets_url_kind",),
    ).fetchone() is not None
    conn.close()

    assert count == 1, f"v11: ожидался 1 feedback_target, найдено {count}"
    assert has_index, "v11: UNIQUE INDEX не создан"


# ---------------------------------------------------------------------------
# MAJOR 4: заявленные тесты C (отсутствующие — добавить реальные)
# ---------------------------------------------------------------------------

def test_indexes_created(tmp_path):
    """
    MAJOR 4: после Storage init на свежей БД PRAGMA index_list(items)
    содержит idx_items_source_fetched и idx_items_cluster_key_fetched.
    """
    db_path = tmp_path / "indexes.db"
    s = Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='items'"
    ).fetchall()
    conn.close()

    index_names = {r[0] for r in rows}
    assert "idx_items_source_fetched" in index_names, (
        f"idx_items_source_fetched missing: {index_names}"
    )
    assert "idx_items_cluster_key_fetched" in index_names, (
        f"idx_items_cluster_key_fetched missing: {index_names}"
    )


def test_downgrade_guard_raises(tmp_path):
    """
    MAJOR 4: создаём БД с user_version=99 (выше текущей SCHEMA_VERSION),
    переоткрываем Storage → RuntimeError (downgrade запрещён).
    """
    db_path = tmp_path / "downgrade.db"
    Storage(db_path)  # инициализируем нормально

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()

    # Создаём новый Storage — должен выбросить RuntimeError.
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        Storage(db_path)


def test_seen_titles_ttl_semantics(tmp_path):
    """
    MAJOR 4: документирует семантику title-dedup в seen_titles:
    - dedup по (title_hash, source): title_normalized одинаков, но разные
      источники → разные title_hash → НЕ дедупятся (разные clusters).
    - expires_at вычисляется как fetched_at + _TITLE_DEDUP_TTL_DAYS (30 дней).
    NOTE: title_hash = md5(title_normalized + source), поэтому source — ключ.
    """
    db_path = tmp_path / "ttl_semantics.db"
    s = Storage(db_path)

    # 3 items: одинаковый title_normalized ("testheadline"), РАЗНЫЕ sources.
    # title_hash отличается → НЕ дедуплятся, все сохраняются.
    results = []
    for i in range(3):
        item = _item(
            title="Test Headline",
            url=f"https://example.com/{i}",
            source=f"Source{i}",  # разные источники → разные title_hash
        )
        results.append(s.save(item))

    # Все 3 save вернули True (нет дедупа из-за разных sources)
    assert all(r is True for r in results), (
        f"Все save с разными source должны вернуть True, got {results}"
    )

    # seen_titles: ровно 3 записи (нет дедупа между sources)
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM seen_titles").fetchone()[0]
    conn.close()
    assert count == 3, f"seen_titles должна содержать 3 записи (разные sources), got {count}"

    # expires_at во всех 3 записях в будущем (TTL active, fresh data)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT expires_at FROM seen_titles").fetchall()
    conn.close()
    for row in rows:
        expires_dt = datetime.fromisoformat(row[0])
        assert expires_dt > now, (
            f"TTL должен быть active: expires={expires_dt}, now={now}"
        )


def test_seen_titles_expired_title_not_deduped(tmp_path):
    """
    Minor 3-фикс (wave 2): протухший title в seen_titles (expires_at в прошлом)
    считается НОВЫМ. is_duplicate должно игнорировать истёкшие записи.

    Bug: оригинальный тест вставлял title_hash="already_seen_hash" (hardcoded
    строка), а is_duplicate ищет по item.title_normalized() — они никогда не
    совпадают, тест зелёный без фильтра expires_at.

    Фикс: вставляем с РЕАЛЬНЫМ title_hash = item.title_normalized().
    save() не вызывает is_duplicate (только URL-проверка), поэтому проверяем
    is_duplicate() напрямую — это API который реализует TTL-фильтр.

    Шаг 1: expires_at в прошлом → is_duplicate==False (TTL истёк).
    Шаг 2: expires_at в будущем → is_duplicate==True (TTL активен).
    """
    from newsbot.storage import Storage

    db_path = tmp_path / "ttl_expired.db"
    s = Storage(db_path)
    now_utc = datetime.now(timezone.utc)

    # Шаг 1: вставляем запись с истёкшим TTL
    expired_iso = (now_utc - timedelta(days=1)).isoformat()
    future_iso = (now_utc + timedelta(days=30)).isoformat()

    # Создаём item ДО вставки — чтобы title_normalized() вычислился.
    item = _item(
        title="TTL Expired Test Title",
        url="https://example.com/ttl_expired",
        source="TTLTestSrc",
    )
    actual_hash = item.title_normalized()  # реальный хэш, который ищет is_duplicate

    # Вставляем запись с истёкшим expires_at и РЕАЛЬНЫМ хэшем.
    # NOTE: fresh DB не имеет колонки title_normalized (v9 миграция не выполняется
    # для from_version=0), поэтому вставляем только в колонки, которые гарантированно
    # существуют: title_hash, source, fetched_at, expires_at.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR IGNORE INTO seen_titles "
        "(title_hash, source, fetched_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (actual_hash, "TTLTestSrc", now_utc.isoformat(), expired_iso),
    )
    conn.commit()
    conn.close()

    # Шаг 1: истёкший TTL — is_duplicate должно вернуть False.
    # URL ещё не в items → проверяется только title-dedup (is_duplicate).
    assert s.is_duplicate(item) is False, (
        "Протухший title_hash не должен считаться дубликатом: "
        "is_duplicate должно игнорировать истёкшие записи"
    )
    # save() вставляет item в items; только после этого is_duplicate
    # начнёт видеть URL (URL-dedup независим от title-dedup).
    assert s.save(item) is True, "First save should succeed (not blocked by expired TTL)"
    # Теперь тот же URL в items → is_duplicate по URL вернёт True.
    assert s.is_duplicate(item) is True, "URL deduplication works after save()"

    # Шаг 2: обновляем expires_at на будущее → TTL активен.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE seen_titles SET expires_at = ? WHERE title_hash = ?",
        (future_iso, actual_hash),
    )
    conn.commit()
    conn.close()

    # Новый item с тем же title+source, но новый URL (чтобы не сработал URL-dedup).
    item2 = _item(
        title="TTL Expired Test Title",
        url="https://example.com/ttl_expired_NEW_URL",
        source="TTLTestSrc",
    )
    # TTL активен: is_duplicate должно вернуть True (блокировка по title_hash).
    assert s.is_duplicate(item2) is True, (
        "Активный TTL (expires_at в будущем) должен блокировать по title_hash"
    )

    s.close()


def test_rows_to_items_null_fetched_at_fallback(tmp_path, caplog):
    """
    MAJOR 4: _rows_to_items получает строку с fetched_at=NULL →
    возвращает item с fallback datetime.min + warning в логах.
    NOTE: get_deferred_items НЕ вернёт такие строки — SQL-фильтр
    fetched_at>=since_iso отсекает NULL (NULL >= anything = NULL).
    Тестируем через _rows_to_items напрямую (минуя SQL-where).
    """
    db_path = tmp_path / "null_fetched.db"
    s = Storage(db_path)

    # Вставляем deferred item с NULL fetched_at
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, state, state_at, llm_score, fetched_at)
        VALUES (?, ?, ?, ?, ?, 'deferred', ?, ?, ?)
        """,
        (
            "TestSrc", "https://null.example/1", "Null Fetched Title",
            "Test", "2025-01-01", "2025-01-01T00:00:00+00:00", 6, None,
        ),
    )
    conn.commit()
    conn.close()

    import logging
    # NOTE: get_deferred_items НЕ вернёт NULL-fetched_at строки
    # (SQL WHERE fetched_at >= ... отсекает NULL).
    # Проверяем это документирующим assertion.
    items_via_query = s.get_deferred_items(since_iso="2000-01-01T00:00:00+00:00")
    assert not any(i.url == "https://null.example/1" for i in items_via_query), (
        "get_deferred_items не должна возвращать NULL-fetched_at items "
        "(SQL-фильтр отсекает NULL). Это ожидаемо и документировано."
    )

    # Тестируем _rows_to_items напрямую — эмулируем строку с NULL fetched_at
    null_row = {
        "source": "TestSrc",
        "url": "https://null.example/1",
        "title": "Null Fetched Title",
        "summary": "Test",
        "published": "2025-01-01",
        "fetched_at": None,
        "state": "deferred",
        "state_at": "2025-01-01T00:00:00+00:00",
        "llm_score": 6,
        "llm_summary": None,
        "llm_reason": None,
        "llm_urgent": None,
        "cluster_key": None,
    }

    with caplog.at_level(logging.WARNING, logger="newsbot.storage"):
        items_direct = s._rows_to_items([null_row])

    assert len(items_direct) == 1
    item = items_direct[0]
    # fallback: datetime.min (минимальный aware UTC datetime)
    from datetime import datetime, timezone
    assert item.fetched_at == datetime.min.replace(tzinfo=timezone.utc), (
        f"fetched_at должен быть datetime.min при NULL: {item.fetched_at}"
    )
    # warning в логах (про NULL fetched_at)
    warn_messages = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any("fetched_at" in m.lower() for m in warn_messages), (
        f"Ожидался WARNING про NULL fetched_at: {warn_messages}"
    )


def test_delete_items_before_removes_terminal_older_than_cutoff(tmp_path):
    """
    MAJOR 4: delete_items_before(window_start_iso) удаляет ТЕРМИНАЛЬНЫЕ
    items (delivered/skipped/failed) с fetched_at < window_start_iso.
    'deferred' НЕ удаляются delete_items_before.
    Тестируем напрямую через Storage (минуя pipeline), чтобы избежать
    сложностей с digest workflow и outdated-fresh fixtures.
    """
    from datetime import datetime, timezone, timedelta

    db_path = tmp_path / "delete_items.db"
    s = Storage(db_path)

    now_str = datetime.now(timezone.utc).isoformat()
    old_str = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

    # Вставляем 3 terminal items: 2 старых (внутри окна), 1 свежий (за пределами)
    for state, url, fa in [
        ("delivered", "https://ex.com/old1", old_str),
        ("skipped", "https://ex.com/old2", old_str),
        ("failed", "https://ex.com/old3", old_str),
        ("delivered", "https://ex.com/fresh", now_str),
    ]:
        _insert_hardening_item(db_path, state=state, url=url, title=f"Item {url}",
                              fetched_at=fa)

    # delete_items_before с cutoff = 24 часа назад (old items внутри, fresh — снаружи)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    deleted = s.delete_items_before(cutoff)

    # Удалены: 3 old (terminal, fetched < cutoff). Fresh: NOT deleted.
    assert deleted == 3, f"Expected 3 deleted, got {deleted}"

    # Fresh item остался
    all_urls = {i.url for i in s.get_all_by_state("delivered")}
    assert "https://ex.com/fresh" in all_urls, "Fresh item должен остаться"
    assert "https://ex.com/old1" not in all_urls, "old1 должен быть удалён"
    assert "https://ex.com/old2" not in all_urls, "old2 должен быть удалён"
    assert "https://ex.com/old3" not in all_urls, "old3 должен быть удалён"

    s.close()

    # Deferred НЕ удаляются delete_items_before
    db_path2 = tmp_path / "deferred_no_delete.db"
    s2 = Storage(db_path2)
    s2.save(_item(title="Def", url="https://ex.com/def", source="test"))
    s2.set_state(["https://ex.com/def"], "deferred")
    deleted2 = s2.delete_items_before(cutoff)
    assert deleted2 == 0, (
        f"deferred не должны удаляться delete_items_before, got {deleted2}"
    )
    assert "https://ex.com/def" in {i.url for i in s2.get_all_by_state("deferred")}
    s2.close()


# ---------------------------------------------------------------------------
# helpers для MAJOR 4
# ---------------------------------------------------------------------------

def _mk_hardening_pipeline(db_path, digest_cfg=None):
    """Pipeline для тестов storage_hardening."""
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": 15,
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
        },
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    p.storage = Storage(db_path)
    return p


def _insert_hardening_item(db_path, state, url, title, fetched_at):
    """Вставляет item с заданным state."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "TestSrc", url, title, "Test summary",
            "2025-01-01", fetched_at, state, fetched_at, 6,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Ремонт по вердикту критика: Storage Minors 1-4 + тесты (а),(б),(в)
# ---------------------------------------------------------------------------

class _TracingStorage(Storage):
    """
    Storage с sqlite3 trace_callback: пишет все выполненные SQL-операторы.
    Нужен, чтобы проверить, что DROP INDEX не вызывается (Minor 1).
    """

    def __init__(self, db_path, traced):
        self._traced = traced
        super().__init__(db_path)

    def _init_db_body(self, conn):
        conn.set_trace_callback(self._traced.append)
        super()._init_db_body(conn)


def _fetch_item_row(db_path, url):
    """Читает строку items по url через отдельное соединение."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM items WHERE url = ?", (url,)).fetchone()
    conn.close()
    return row


def _index_sql(db_path, name="idx_feedback_targets_url_kind"):
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (name,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def test_unique_feedback_index_not_dropped_on_reinit(tmp_path):
    """
    Storage Minor 1: существующий UNIQUE-индекс не дропается на каждом старте
    (раньше DROP+CREATE выполнялись всегда → окно без уникальности).
    """
    db_path = tmp_path / "unique_keep.db"
    Storage(db_path).close()

    traced = []
    s = _TracingStorage(db_path, traced)
    s.close()

    # NIT: проверяем, что trace_callback действительно что-то записал —
    # иначе "нет DROP" мог бы означать «трассировка не работает».
    assert any("CREATE UNIQUE INDEX" in s for s in traced), (
        f"trace_callback не зафиксировал CREATE UNIQUE INDEX: {traced[:5]}"
    )

    drops = [
        stmt for stmt in traced
        if stmt.strip().upper().startswith("DROP INDEX")
        and "IDX_FEEDBACK_TARGETS_URL_KIND" in stmt.upper()
    ]
    assert not drops, f"UNIQUE-индекс не должен дропаться: {drops}"

    # Уникальность по-прежнему работает.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
        ("https://uniq.example/1", "push", "2024-01-01T00:00:00"),
    )
    integrity_error = False
    try:
        conn.execute(
            "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
            ("https://uniq.example/1", "push", "2024-01-02T00:00:00"),
        )
    except sqlite3.IntegrityError:
        integrity_error = True
    conn.close()
    assert integrity_error, "UNIQUE(url, kind) должен остаться в силе"


def test_feedback_targets_unique_index_recovers_after_integrity_error(tmp_path):
    """
    Storage Minor 1: если UNIQUE-индекса нет и в таблице есть дубли,
    IntegrityError при CREATE не глотается: выполняется одноразовый дедуп
    (min(id)) и индекс всё равно создаётся.
    """
    db_path = tmp_path / "unique_recover.db"
    Storage(db_path).close()

    # Ломаем состояние: убираем UNIQUE-индекс и вставляем дубли.
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP INDEX IF EXISTS idx_feedback_targets_url_kind")
    for i in range(3):
        conn.execute(
            "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
            ("https://dup-fix.example/1", "digest", f"2024-01-0{i + 1}T00:00:00"),
        )
    conn.commit()
    conn.close()

    s = Storage(db_path)  # должен дедуплицировать и создать UNIQUE-индекс
    s.close()

    sql = _index_sql(db_path)
    assert sql is not None, "UNIQUE-индекс не создан после дедупа"
    assert "UNIQUE" in sql.upper(), f"индекс должен быть UNIQUE: {sql}"

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
        ("https://dup-fix.example/1", "digest"),
    ).fetchall()
    conn.close()
    assert len(rows) == 1, f"дубли должны быть удалены, осталось {len(rows)}"


def test_feedback_targets_dedupe_redirects_feedback(tmp_path):
    """
    Storage Minor: дедуп feedback_targets не оставляет осиротевших feedback.
    Голос, привязанный к удаляемому дублю target, перенаправляется на
    выживший MIN(id) — поздний клик не теряется.
    """
    db_path = tmp_path / "dedupe_feedback.db"
    Storage(db_path).close()

    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP INDEX IF EXISTS idx_feedback_targets_url_kind")
    url = "https://dedupe-fb.example/1"
    for i in range(3):
        conn.execute(
            "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
            (url, "push", f"2024-01-0{i + 1}T00:00:00"),
        )
    dup_target_id = conn.execute(
        "SELECT MAX(id) FROM feedback_targets WHERE url = ? AND kind = ?",
        (url, "push"),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO feedback (target_id, url, verdict, created_at) "
        "VALUES (?, ?, ?, ?)",
        (dup_target_id, url, "like", "2024-01-03T00:00:00"),
    )
    conn.commit()
    conn.close()

    s = Storage(db_path)  # дедуп + UNIQUE-индекс
    try:
        rows = s._connect().execute("SELECT target_id FROM feedback").fetchall()
        assert [r[0] for r in rows] == [1], (
            f"feedback должен переехать на выживший target id=1, "
            f"получено {[r[0] for r in rows]}"
        )
        assert s.get_feedback_target(rows[0][0]) is not None, (
            "get_feedback_target для выжившего target не должен возвращать None"
        )
        conn = sqlite3.connect(str(db_path))
        targets = conn.execute(
            "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
            (url, "push"),
        ).fetchall()
        conn.close()
        assert len(targets) == 1, f"должен остаться один target, {targets}"
    finally:
        s.close()


def test_feedback_targets_dedupe_keeps_newer_feedback_on_conflict(tmp_path):
    """
    Storage Minor: UNIQUE(target_id) — при перенаправлении голоса на выживший
    target конфликт разрешается в пользу более свежего feedback
    (COALESCE(updated_at, created_at)).
    """
    db_path = tmp_path / "dedupe_feedback_conflict.db"
    Storage(db_path).close()

    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP INDEX IF EXISTS idx_feedback_targets_url_kind")
    url = "https://dedupe-fb2.example/1"
    for i in range(2):
        conn.execute(
            "INSERT INTO feedback_targets (url, kind, created_at) VALUES (?, ?, ?)",
            (url, "push", f"2024-01-0{i + 1}T00:00:00"),
        )
    keeper_id, dup_id = 1, 2
    # Старый голос — на выжившем target, свежий — на дубле.
    conn.execute(
        "INSERT INTO feedback (target_id, url, verdict, created_at) "
        "VALUES (?, ?, ?, ?)",
        (keeper_id, url, "dislike", "2024-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO feedback (target_id, url, verdict, created_at) "
        "VALUES (?, ?, ?, ?)",
        (dup_id, url, "like", "2024-02-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    s = Storage(db_path)
    try:
        rows = s._connect().execute(
            "SELECT target_id, verdict FROM feedback"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(keeper_id, "like")], (
            f"должен остаться один более свежий голос на выжившем target: "
            f"{[(r[0], r[1]) for r in rows]}"
        )
    finally:
        s.close()


def test_add_feedback_target_is_idempotent(tmp_path):
    """
    Storage Minor 4: legacy add_feedback_target на повторе не падает
    IntegrityError, а возвращает id существующей записи.
    """
    db_path = tmp_path / "legacy_add.db"
    s = Storage(db_path)

    kwargs = dict(
        url="https://legacy.example/1", source="S", title="T",
        summary="Sum", llm_score=7, kind="push",
    )
    id1 = s.add_feedback_target(**kwargs)
    id2 = s.add_feedback_target(**kwargs)  # не должно бросать IntegrityError

    assert id1 == id2, f"ожидался тот же id, получено {id1} != {id2}"
    s.close()


def test_v13_post_check_adds_lang_column(tmp_path):
    """
    Тест (а): user_version=13, но колонки items.lang нет →
    post-check досоздаёт колонку, существующие данные целы.
    """
    db_path = tmp_path / "v13_nolang.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 13")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source TEXT, url TEXT UNIQUE, title TEXT, summary TEXT,
            published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
            llm_score INTEGER, llm_summary TEXT, llm_reason TEXT,
            llm_urgent INTEGER, filtered_at TEXT, retry_count INTEGER,
            cluster_key TEXT, tags TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY, title_hash TEXT, source TEXT,
            fetched_at TEXT, expires_at TEXT, title_normalized TEXT
        )
    """)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """
        INSERT INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, retry_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("OldSrc", "https://old.example/1", "Old title", "Old summary",
         "2024-01-01", "2024-01-01T00:00:00", "new", "2024-01-01T00:00:00", 0),
    )
    conn.commit()
    conn.close()

    s = Storage(db_path)

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn2.execute("PRAGMA table_info(items)").fetchall()}
    old_row = conn2.execute(
        "SELECT url, title FROM items WHERE url = ?", ("https://old.example/1",)
    ).fetchone()
    conn2.close()

    assert "lang" in cols, f"lang должна быть добавлена post-check: {cols}"
    assert old_row is not None, "существующая строка потеряна"
    assert old_row["title"] == "Old title", "данные существующей строки изменились"

    # Колонка рабочая: сохранение item с lang проходит.
    s.save(NewsItem(
        source="S", url="https://v13post.example/1", title="V13",
        summary="S", published="2024-01-01", published_parsed=None, lang="ru",
    ))
    saved = _fetch_item_row(db_path, "https://v13post.example/1")
    assert saved is not None and saved["lang"] == "ru"
    s.close()


def test_concurrent_initialization_two_storages(tmp_path):
    """
    Тест (б): два Storage на одном файле инициализируются одновременно
    (барьер), оба рабочие — каждый сохраняет свой item в своём потоке.
    """
    import threading

    db_path = tmp_path / "concurrent.db"
    Storage(db_path).close()  # файл существует, journal_mode=WAL уже выставлен

    barrier = threading.Barrier(2)
    results = {}

    def _open_and_save(idx):
        try:
            barrier.wait(timeout=10)
            st = Storage(db_path)  # конкурентная инициализация
            st.save(NewsItem(
                source="S", url=f"https://conc.example/{idx}",
                title=f"Conc {idx}", summary="S",
                published="2024-01-01", published_parsed=None,
            ))
            st.close()
            results[idx] = True
        except Exception as e:  # noqa: BLE001 — тест должен показать ошибку явно
            results[idx] = e

    threads = [threading.Thread(target=_open_and_save, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    errors = {k: v for k, v in results.items() if isinstance(v, Exception)}
    assert not errors, f"конкурентная инициализация упала: {errors}"
    assert results == {0: True, 1: True}

    for i in range(2):
        assert _fetch_item_row(db_path, f"https://conc.example/{i}") is not None

    conn = sqlite3.connect(str(db_path))
    count = conn.execute(
        "SELECT COUNT(*) FROM items WHERE url LIKE 'https://conc.example/%'"
    ).fetchone()[0]
    uv = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert count == 2, f"ожидалось 2 items, найдено {count}"
    assert uv == Storage.SCHEMA_VERSION


def test_purge_items_keeps_old_pending(tmp_path):
    """
    Тест (в): purge_items не УДАЛЯЕТ pending — застрявший pending
    (state_at старше PENDING_STALE_DAYS) терминализируется в 'failed',
    но строка остаётся в БД и удалится обычным retention-окном для failed.
    """
    db_path = tmp_path / "purge_pending.db"
    s = Storage(db_path)

    old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    _insert_hardening_item(db_path, "pending", "https://pend.example/1", "Pending", old_ts)
    _insert_hardening_item(db_path, "delivered", "https://deliv.example/1", "Delivered", old_ts)

    deleted = s.purge_items(retention_days=1, deferred_retention_days=1)

    assert deleted == 1, f"должен быть удалён только delivered, deleted={deleted}"
    pend_row = _fetch_item_row(db_path, "https://pend.example/1")
    assert pend_row is not None, \
        "старый pending НЕ должен удаляться purge_items"
    assert pend_row["state"] == "failed", \
        "застрявший pending должен терминализироваться в 'failed', " \
        f"а не остаться {pend_row['state']}"
    assert _fetch_item_row(db_path, "https://deliv.example/1") is None
    s.close()


def test_purge_items_keeps_fresh_pending_as_pending(tmp_path):
    """Свежий pending (в работе) терминализации не подлежит."""
    db_path = tmp_path / "purge_pending_fresh.db"
    s = Storage(db_path)

    fresh_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _insert_hardening_item(db_path, "pending", "https://pend2.example/1", "Pending", fresh_ts)

    assert s.purge_items(retention_days=1, deferred_retention_days=1) == 0
    row = _fetch_item_row(db_path, "https://pend2.example/1")
    assert row is not None and row["state"] == "pending", \
        "свежий pending должен остаться pending"
    s.close()


def test_purge_items_negative_retention_does_not_wipe(tmp_path):
    """
    Major: retention_days=-1 (опечатка) → default (DEFAULT_RETENTION_DAYS),
    НЕ падает в исключение и НЕ использует cutoff=now (что снесло бы ВСЕ данные).
    С DEFAULT_RETENTION_DAYS=1: items старше 1 дня удаляются (expected),
    но deferred по default=3 сохраняется (отдельный параметр deferred_retention_days).
    """
    db_path = tmp_path / "purge_negative.db"
    s = Storage(db_path)

    # 100 дней назад: старше DEFAULT_RETENTION_DAYS=1 → УДАЛЯЮТСЯ (expected)
    ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    for i, state in enumerate(("delivered", "skipped", "failed", "digested")):
        _insert_hardening_item(
            db_path, state, f"https://neg.example/{i}", f"{state}-item", ts,
        )
    # deferred 1 день назад: default deferred_retention=3 → сохраняется (отказ на -1 → default 3).
    recent_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _insert_hardening_item(
        db_path, "deferred", "https://neg.example/def", "def", recent_ts,
    )

    deleted = s.purge_items(retention_days=-1, deferred_retention_days=-1)

    # 4 терминальных старше 1 дня — удалились. deferred по default=3 сохранился.
    assert deleted == 4, f"retention_days=-1 → default 1, 4 items старше 1 дня удалились, deleted={deleted}"
    for i, state in enumerate(("delivered", "skipped", "failed", "digested")):
        row = _fetch_item_row(db_path, f"https://neg.example/{i}")
        assert row is None, f"{state} должен быть удалён (старше 1 дня, default=1)"
    # deferred живёт по deferred_retention_days=3 (отказ на -1 → default 3)
    assert _fetch_item_row(db_path, "https://neg.example/def") is not None, \
        "deferred не должен удаляться при deferred_retention_days=-1 (→ default 3)"
    s.close()


# ---------------------------------------------------------------------------
# v1.3.0: retention-параметры из config.yaml (строки) + cap title
# ---------------------------------------------------------------------------

def test_purge_items_accepts_string_retention(tmp_path):
    """retention_days: "1" (строка из config.yaml) не должен валить purge."""
    db_path = tmp_path / "purge_string_retention.db"
    s = Storage(db_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    _insert_hardening_item(db_path, "delivered", "https://str.example/1", "Old", old_ts)

    deleted = s.purge_items(retention_days="1", deferred_retention_days="1")

    assert deleted == 1, f"строка retention_days должна работать, deleted={deleted}"
    assert _fetch_item_row(db_path, "https://str.example/1") is None
    s.close()


def test_purge_items_invalid_retention_does_not_raise(tmp_path):
    """Невалидный retention_days → fallback, purge не падает."""
    s = Storage(tmp_path / "purge_bad_retention.db")
    assert s.purge_items(retention_days="abc", deferred_retention_days=None) == 0
    s.close()


def test_prune_feedback_targets_invalid_days_skips(tmp_path):
    """Невалидный older_than_days → 0 (данные не сносим), без исключения."""
    s = Storage(tmp_path / "prune_bad.db")
    assert s.prune_feedback_targets("abc") == 0
    s.close()


def test_prune_feedback_targets_valid_days_keeps_fresh_targets(tmp_path):
    """
    prune на НЕпустой БД: валидный "30" (строка из config.yaml) не сносит
    свежие targets. Раньше тест шёл на пустой БД и был тривиальным.
    """
    s = Storage(tmp_path / "prune_fresh.db")
    tid = s.add_feedback_target(
        url="https://fresh-prune.example/1", source="TestSrc", title="Fresh",
        summary="s", llm_score=6, kind="push",
    )
    assert s.prune_feedback_targets("30") == 0
    assert s.get_feedback_target(tid) is not None, "свежий target не должен удаляться"
    s.close()


def test_prune_feedback_targets_negative_days_does_not_wipe(tmp_path):
    """
    Major: older_than_days=-1 (опечатка в конфиге) не должен сносить ВСЕ
    targets — кнопки под старыми сообщениями умерли бы. days<=0 → пропуск.
    """
    s = Storage(tmp_path / "prune_negative.db")
    tid = s.add_feedback_target(
        url="https://old-prune.example/1", source="TestSrc", title="Old",
        summary="s", llm_score=6, kind="push",
    )
    old_date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    conn = s._connect()
    conn.execute(
        "UPDATE feedback_targets SET created_at = ? WHERE id = ?", (old_date, tid),
    )
    conn.commit()

    assert s.prune_feedback_targets(-1) == 0, "отрицательный retention не должен удалять"
    assert s.prune_feedback_targets(0) == 0, "0 трактуется как «не чистить»"
    assert s.get_feedback_target(tid) is not None, "target не должен быть снесён"
    s.close()


def test_save_caps_huge_title(tmp_path):
    """Враждебный RSS: title в 100К символов капится в БД."""
    from newsbot.storage import MAX_TITLE_CHARS

    db_path = tmp_path / "title_cap.db"
    s = Storage(db_path)
    item = NewsItem(
        source="TestSrc",
        url="https://cap.example/1",
        title="X" * 100_000,
        summary="s",
        published="01.01.2026",
    )
    assert s.save(item) is True

    row = _fetch_item_row(db_path, "https://cap.example/1")
    assert row is not None
    assert len(row["title"]) == MAX_TITLE_CHARS
    s.close()


def test_save_many_caps_huge_title(tmp_path):
    """save_many тоже капит title (batch-путь записи)."""
    from newsbot.storage import MAX_TITLE_CHARS

    db_path = tmp_path / "title_cap_many.db"
    s = Storage(db_path)
    item = NewsItem(
        source="TestSrc",
        url="https://cap.example/2",
        title="Y" * 50_000,
        summary="s",
        published="01.01.2026",
    )
    s.save_many([item])

    row = _fetch_item_row(db_path, "https://cap.example/2")
    assert row is not None
    assert len(row["title"]) == MAX_TITLE_CHARS
    s.close()


class _FlakyCommitConn:
    """
    Прокси вокруг sqlite3.Connection: первый commit() падает (имитация
    disk I/O error / сбоя fsync), остальные делегируются реальному соединению.
    Нужен, чтобы проверить rollback-семантику _tx().
    """

    def __init__(self, real):
        self._real = real
        self.fail_next_commit = True

    def execute(self, *args, **kwargs):
        return self._real.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("disk I/O error (simulated)")
        self._real.commit()


def test_set_meta_uses_transaction(tmp_path):
    """
    set_meta в _tx: падение commit'а откатывается (ROLLBACK), значение НЕ
    записано, и следующая транзакция открывается нормально (иначе было бы
    «cannot start a transaction within a transaction»).
    """
    s = Storage(tmp_path / "meta_tx.db")
    s.set_meta("k", "v")
    assert s.get_meta("k") == "v"

    proxy = _FlakyCommitConn(s._connect())
    with mock.patch.object(s, "_connect", return_value=proxy):
        with pytest.raises(sqlite3.OperationalError):
            s.set_meta("k", "broken")
        # после сбойного commit соединение НЕ осталось в транзакции:
        # следующая запись проходит (в старом коде упала бы на BEGIN)
        s.set_meta("k", "v2")

    assert s.get_meta("k") == "v2", "успешная запись после сбоя должна примениться"
    s.close()


# ---------------------------------------------------------------------------
# Minor (волна 4): батчевая терминализация pending + отравленное соединение
# ---------------------------------------------------------------------------

def test_purge_items_terminalizes_large_pending_backlog_in_batches(
    tmp_path, monkeypatch, caplog,
):
    """
    Minor (волна 4): 12k застрявших pending терминализируются ПОРЦИЯМИ
    (короткие транзакции), а не одним UPDATE под общим WAL-lock'ом.
    Батч уменьшен, чтобы проверить саму итеративность цикла.
    """
    monkeypatch.setattr(Storage, "PURGE_PENDING_BATCH_SIZE", 1000)

    db_path = tmp_path / "purge_pending_batches.db"
    s = Storage(db_path)

    old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    total = 12_000
    conn = sqlite3.connect(str(db_path))
    conn.executemany(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "TestSrc", f"https://batch.example/{i}", f"Batch {i}", "s",
                "2025-01-01", old_ts, "pending", old_ts, 0,
            )
            for i in range(total)
        ),
    )
    conn.commit()
    conn.close()

    traced = []
    s._connect().set_trace_callback(traced.append)

    with caplog.at_level(logging.WARNING, logger="newsbot.storage"):
        deleted = s.purge_items(retention_days=1, deferred_retention_days=1)

    s._connect().set_trace_callback(None)

    assert deleted == 0, f"pending не удаляются, deleted={deleted}"

    conn = sqlite3.connect(str(db_path))
    pending_left = conn.execute(
        "SELECT COUNT(*) FROM items WHERE state = 'pending'"
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM items WHERE state = 'failed'"
    ).fetchone()[0]
    conn.close()

    assert pending_left == 0, f"осталось pending: {pending_left}"
    assert failed == total, f"в failed должно быть {total}, got {failed}"

    # Батчинг реально работал: 12 порций по 1000 + завершающая пустая +
    # отдельная транзакция DELETE-фазы.
    # Minor-5: _tx() использует BEGIN IMMEDIATE, поэтому проверяем оба варианта.
    begins = [
        stmt for stmt in traced
        if stmt.strip().upper() in ("BEGIN", "BEGIN IMMEDIATE")
    ]
    assert len(begins) >= 12, (
        f"ожидалось ≥12 коротких транзакций (батчи), получено {len(begins)}"
    )

    # Warning агрегированный: один, с общим числом.
    warn_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and "pending" in r.message
    ]
    assert len(warn_msgs) == 1, f"ожидался один агрегированный warning: {warn_msgs}"
    assert str(total) in warn_msgs[0], (
        f"warning должен содержать общее число {total}: {warn_msgs[0]}"
    )

    s.close()


class _PoisonedConn:
    """
    Соединение, у которого падают и commit(), и ROLLBACK, и close():
    имитация мёртвого fd / disk I/O error. Все прочие вызовы — no-op.
    Minor-5: также добавлено in_transaction и execute возвращает объект
    с rowcount для совместимости с BEGIN IMMEDIATE + _tx().
    """

    row_factory = sqlite3.Row
    in_transaction = False

    def execute(self, sql, *args, **kwargs):
        if sql.strip().upper().startswith("ROLLBACK"):
            raise sqlite3.OperationalError("disk I/O error (rollback)")
        # Minor-5: возвращаем dummy cursor для BEGIN IMMEDIATE
        class DummyCursor:
            rowcount = 0
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def fetchone(self): return None
            def fetchall(self): return []
        return DummyCursor()

    def commit(self):
        raise sqlite3.OperationalError("disk I/O error (commit)")

    def rollback(self):
        raise sqlite3.OperationalError("disk I/O error (rollback)")

    def close(self):
        raise sqlite3.OperationalError("disk I/O error (close)")


def test_tx_resets_connection_when_rollback_fails(tmp_path):
    """
    Minor (волна 4): если в _tx падает и commit, и ROLLBACK — соединение
    отравлено, self._conn сбрасывается, и следующая запись идёт через
    заново созданное соединение (до фикса BEGIN падал до рестарта).
    """
    db_path = tmp_path / "poisoned_conn.db"
    s = Storage(db_path)
    real = s._conn

    s._conn = _PoisonedConn()
    with pytest.raises(sqlite3.OperationalError):
        s.set_meta("k", "broken")

    assert s._conn is None, (
        "после провала ROLLBACK соединение должно быть сброшено (self._conn = None)"
    )

    real.close()

    # Ленивое пересоздание в _connect() работает — запись проходит.
    s.set_meta("k", "v")
    assert s.get_meta("k") == "v", "запись после пересоздания соединения должна пройти"
    assert s._conn is not None
    s.close()