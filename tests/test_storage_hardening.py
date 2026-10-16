"""
test_storage_hardening.py — фиксы M1, M2, M3, Minor#4.

M1  — индекс idx_seen_fetched и idx_items_state_state_at (v11).
M2  — save/save_many логируют на уровне error, различают дедуп и реальную ошибку.
M3  — recover_stuck_pending снимает терминальные БЕЗ возрастного фильтра.
Minor#4 — UNIQUE(url, kind) на feedback_targets, INSERT OR IGNORE в get_or_create_*.
"""
import logging
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

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
    assert uv == 11, f"expected user_version=11, got {uv}"

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
    assert uv == 11

    indexes = _read_indexes(db_path)
    assert "idx_seen_fetched" in indexes, f"idx_seen_fetched missing: {indexes}"
    assert "idx_items_state_state_at" in indexes


def test_migration_v11_idempotent(tmp_path):
    """Повторный запуск Storage на v11 не падает и не меняет user_version."""
    db_path = tmp_path / "idempotent.db"
    Storage(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 11
    conn.close()

    # Второй Storage на той же БД — без ошибок.
    Storage(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 11
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