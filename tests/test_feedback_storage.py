"""
tests/test_feedback_storage.py
Фаза 2a (доработка после read-only ревью): тесты миграции v6→v8,
feedback_targets, feedback.

Изменения после ревью:
  N1     — SCHEMA_VERSION 7→8; миграция v7→v8 добавляет updated_at + CHECK(verdict)
  N1-CHECK — CHECK(verdict) во всех трёх DDL (fresh / v7 / RECREATE)
  N2     — CREATE TABLE IF NOT EXISTS feedback* в _init_db (F7); транзакция RECREATE
  N3     — record_feedback: INSERT OR IGNORE + rowcount + UPDATE (без TOCTOU)
  N3     — миграция v7→v8: сохранение данных (id/target_id/verdict/created_at)
  N4     — проверка CHECK через DDL из sqlite_master (без записи в таблицу)
  NIT    — INSERT OR IGNORE вместо INSERT...ON CONFLICT; улучшенный docstring
"""
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import mkdtemp
import shutil

import pytest

from newsbot.storage import Storage
from newsbot.collector import NewsItem


# ---------------------------------------------------------------------------
# фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_path():
    d = Path(mkdtemp(prefix="nb_test_fb_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "test.db")


@pytest.fixture
def news_item():
    """Базовый NewsItem для тестов (fresh-объект каждый вызов)."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    return NewsItem(
        source="TestSource",
        url=f"https://example.com/{uid}",
        title="Тестовая статья",
        summary="Краткое содержание тестовой статьи.",
        published="2026-01-01",
        published_parsed=None,
    )


# ---------------------------------------------------------------------------
# миграция v6 → v8 (полный цикл)
# ---------------------------------------------------------------------------

def test_migration_v8_schema_version(storage):
    """user_version после инициализации = 8."""
    conn = sqlite3.connect(str(storage.db_path))
    conn.row_factory = sqlite3.Row
    assert conn.execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION
    conn.close()


def test_migration_v8_fresh_db_has_both_tables(tmp_path):
    """Свежая БД сразу получает обе таблицы (v8-схема)."""
    s = Storage(tmp_path / "fresh.db")
    conn = sqlite3.connect(str(s.db_path))
    conn.row_factory = sqlite3.Row
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "feedback_targets" in tables
    assert "feedback" in tables


def test_migration_v8_idempotent_reopen(tmp_path, news_item):
    """NIT: повторное открытие той же БД не падает (открываем 2 раза)."""
    db = tmp_path / "reopen.db"
    s1 = Storage(db)
    v1 = sqlite3.connect(str(db)).execute("PRAGMA user_version").fetchone()[0]
    assert v1 == Storage.SCHEMA_VERSION

    # повторное открытие (idempotent)
    s2 = Storage(db)
    v2 = sqlite3.connect(str(db)).execute("PRAGMA user_version").fetchone()[0]
    assert v2 == Storage.SCHEMA_VERSION

    # третий раз — тоже
    s3 = Storage(db)
    v3 = sqlite3.connect(str(db)).execute("PRAGMA user_version").fetchone()[0]
    assert v3 == Storage.SCHEMA_VERSION

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "items" in tables
    assert "feedback_targets" in tables
    assert "feedback" in tables


def test_migration_v6_to_v8_full(tmp_path):
    """Эмуляция: v6-БД + миграция → v10, все таблицы на месте."""
    db_path = tmp_path / "v6.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 6")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source TEXT, url TEXT, title TEXT, summary TEXT,
            published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
            retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY,
            title_hash TEXT, source TEXT, fetched_at TEXT, expires_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    conn.commit()
    conn.close()

    s = Storage(db_path)
    assert sqlite3.connect(str(db_path)).execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    tables = {r["name"] for r in conn2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn2.close()
    assert "feedback_targets" in tables
    assert "feedback" in tables


def test_migration_v7_to_v8_adds_updated_at(tmp_path):
    """N1: v7-БД (без updated_at) получает колонку при миграции v7→v9."""
    db_path = tmp_path / "v7.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 7")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source TEXT, url TEXT, title TEXT, summary TEXT,
            published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
            retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY,
            title_hash TEXT, source TEXT, fetched_at TEXT, expires_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    # v7-schema feedback БЕЗ updated_at и БЕЗ CHECK (как было после первого запуска)
    conn.execute("""
        CREATE TABLE feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            target_id   INTEGER NOT NULL,
            url         TEXT NOT NULL,
            source      TEXT,
            title       TEXT,
            summary     TEXT,
            llm_score   INTEGER,
            verdict     TEXT NOT NULL,
            chat_id     TEXT,
            message_id  INTEGER,
            created_at  TEXT NOT NULL,
            UNIQUE(target_id)
        )
    """)
    conn.execute("""
        CREATE TABLE feedback_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, source TEXT, title TEXT, summary TEXT,
            llm_score INTEGER, kind TEXT NOT NULL,
            chat_id TEXT, message_id INTEGER, created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # Миграция v7→v8
    s = Storage(db_path)
    assert sqlite3.connect(str(db_path)).execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION

    # updated_at добавлена
    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    fb_cols = {r["name"] for r in conn2.execute("PRAGMA table_info(feedback)").fetchall()}
    conn2.close()
    assert "updated_at" in fb_cols, "updated_at должна быть добавлена при миграции v7→v8"


def test_migration_v7_to_v8_preserves_data(tmp_path):
    """N3: миграция v7→v8 сохраняет данные (id/target_id/verdict/created_at), updated_at=NULL."""
    db_path = tmp_path / "v7_data.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 7")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source TEXT, url TEXT, title TEXT, summary TEXT,
            published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
            retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY,
            title_hash TEXT, source TEXT, fetched_at TEXT, expires_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    # v7-schema: feedback БЕЗ updated_at (чистая схема)
    conn.execute("""
        CREATE TABLE feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            target_id   INTEGER NOT NULL,
            url         TEXT NOT NULL,
            source      TEXT,
            title       TEXT,
            summary     TEXT,
            llm_score   INTEGER,
            verdict     TEXT NOT NULL,
            chat_id     TEXT,
            message_id  INTEGER,
            created_at  TEXT NOT NULL,
            UNIQUE(target_id)
        )
    """)
    conn.execute("""
        CREATE TABLE feedback_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, source TEXT, title TEXT, summary TEXT,
            llm_score INTEGER, kind TEXT NOT NULL,
            chat_id TEXT, message_id INTEGER, created_at TEXT NOT NULL
        )
    """)
    # Вставляем 2 строки ДО миграции
    conn.execute("""
        INSERT INTO feedback (target_id, url, source, title, summary, verdict, created_at)
        VALUES (1, 'https://x.com/1', 'Src1', 'Title1', 'Sum1', 'like', '2026-01-01T10:00:00+00:00')
    """)
    conn.execute("""
        INSERT INTO feedback (target_id, url, source, title, summary, verdict, created_at)
        VALUES (2, 'https://x.com/2', 'Src2', 'Title2', 'Sum2', 'dislike', '2026-01-02T12:00:00+00:00')
    """)
    conn.commit()
    conn.close()

    # Миграция v7→v8
    s = Storage(db_path)

    # Проверяем: все 2 строки на месте, данные не искажены
    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    rows = conn2.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    conn2.close()
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["target_id"] == 1
    assert rows[0]["verdict"] == "like"
    assert rows[0]["created_at"] == "2026-01-01T10:00:00+00:00"
    assert rows[0]["updated_at"] is None, "updated_at должен быть NULL (первый голос)"
    assert rows[1]["id"] == 2
    assert rows[1]["verdict"] == "dislike"
    assert rows[1]["created_at"] == "2026-01-02T12:00:00+00:00"
    assert rows[1]["updated_at"] is None


# ---------------------------------------------------------------------------
# N2: отсутствующие таблицы создаются при актуальной версии
# ---------------------------------------------------------------------------

def test_feedback_tables_created_at_v8_reopen(tmp_path):
    """N2: при user_version=8 отсутствующие feedback-таблицы пересоздаются."""
    db_path = tmp_path / "v8_missing.db"
    # создаём v8-БД но удаляем feedback-таблицы
    s_pre = Storage(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS feedback")
    conn.execute("DROP TABLE IF EXISTS feedback_targets")
    conn.commit()
    conn.close()

    # повторное открытие — таблицы должны появиться
    s_reopen = Storage(db_path)
    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    tables = {r["name"] for r in conn2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn2.close()
    assert "feedback_targets" in tables
    assert "feedback" in tables


def test_feedback_check_constraint_rejects_invalid_verdict(tmp_path):
    """N1-CHECK: прямой INSERT невалидного verdict вызывает IntegrityError (CHECK на БД)."""
    s = Storage(tmp_path / "check.db")
    conn = sqlite3.connect(str(s.db_path))
    # Вставляем через INSERT INTO feedback напрямую (минуя record_feedback)
    try:
        conn.execute(
            "INSERT INTO feedback (target_id, url, verdict, created_at) "
            "VALUES (?, ?, ?, ?)",
            (1, "https://x.com/check", "invalid_verdict", "2026-01-01"),
        )
        conn.commit()
        conn.close()
        assert False, "CHECK должен был отклонить invalid_verdict"
    except sqlite3.IntegrityError:
        conn.close()
        # CHECK сработал — ожидаемое поведение


# ---------------------------------------------------------------------------
# feedback_targets
# ---------------------------------------------------------------------------

def test_add_feedback_target_from_item(storage, news_item):
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    assert isinstance(tid, int)
    assert tid > 0


def test_add_feedback_target_fields(storage):
    tid = storage.add_feedback_target(
        url="https://x.com/p/1", source="XSource",
        title="Заголовок", summary="Суть", llm_score=9, kind="digest",
    )
    row = storage.get_feedback_target(tid)
    assert row["url"] == "https://x.com/p/1"
    assert row["source"] == "XSource"
    assert row["llm_score"] == 9
    assert row["kind"] == "digest"


def test_set_feedback_target_message(storage, news_item):
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.set_feedback_target_message(tid, chat_id="123456", message_id=999)
    row = storage.get_feedback_target(tid)
    assert row["chat_id"] == "123456"
    assert row["message_id"] == 999


def test_get_feedback_target_not_found(storage):
    assert storage.get_feedback_target(99999) is None


# ---------------------------------------------------------------------------
# feedback (record_feedback, upsert, N3)
# ---------------------------------------------------------------------------

def test_record_feedback_like(storage, news_item):
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    assert storage.record_feedback(tid, "like") is True
    rows = storage.get_recent_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "like"


def test_record_feedback_dislike(storage, news_item):
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    assert storage.record_feedback(tid, "dislike") is True
    rows = storage.get_recent_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "dislike"


def test_record_feedback_returns_bool(storage, news_item):
    """record_feedback возвращает True при успехе."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    assert storage.record_feedback(tid, "like") is True


def test_record_feedback_upsert_updates_verdict(storage, news_item):
    """Повторный record_feedback — UPDATE, не дубликат."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")
    storage.record_feedback(tid, "dislike")
    rows = storage.get_recent_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] == "dislike"


def test_record_feedback_preserves_created_at(storage, news_item):
    """F5: created_at неизменяем — re-vote не двигает запись."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")
    rows_before = storage.get_recent_feedback()
    created_before = rows_before[0]["created_at"]

    time.sleep(0.01)
    storage.record_feedback(tid, "dislike")
    rows_after = storage.get_recent_feedback()
    assert rows_after[0]["created_at"] == created_before


def test_record_feedback_sets_updated_at_on_revote(storage, news_item):
    """F5: updated_at ставится при re-vote, NULL при первом голосе."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")
    row_first = storage.get_recent_feedback()[0]
    assert row_first["updated_at"] is None

    storage.record_feedback(tid, "dislike")
    row_revote = storage.get_recent_feedback()[0]
    assert row_revote["updated_at"] is not None
    assert row_revote["verdict"] == "dislike"


def test_record_feedback_no_toctou_concurrent_insert(tmp_path, news_item):
    """N3: параллельные первые голоса не дают IntegrityError (ON CONFLICT)."""
    db_path = tmp_path / "race.db"
    s = Storage(db_path)
    tid = s.add_feedback_target_from_item(news_item, kind="push")

    errors = []
    results = []

    def vote(verdict):
        s2 = Storage(db_path)
        try:
            r = s2.record_feedback(tid, verdict)
            results.append(r)
        except Exception as e:
            errors.append(e)

    import threading
    t1 = threading.Thread(target=lambda: vote("like"))
    t2 = threading.Thread(target=lambda: vote("dislike"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Не должно быть IntegrityError: {errors}"
    # Один из голосов записался, целостность БД сохранена
    assert len(results) == 2
    rows = s.get_recent_feedback()
    assert len(rows) == 1
    assert rows[0]["verdict"] in ("like", "dislike")


def test_record_feedback_invalid_verdict_raises(storage, news_item):
    """Невалидный verdict вызывает ValueError."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    with pytest.raises(ValueError, match="verdict"):
        storage.record_feedback(tid, "unknown")


def test_migration_probe_checks_ddl_from_sqlite_master(tmp_path):
    """N4: миграция определяет наличие CHECK по DDL из sqlite_master (без записи в таблицу)."""
    db_path = tmp_path / "probe.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 7")
    conn.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source TEXT, url TEXT, title TEXT, summary TEXT,
            published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
            retry_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE seen_titles (
            id INTEGER PRIMARY KEY,
            title_hash TEXT, source TEXT, fetched_at TEXT, expires_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
    """)
    # v7-schema feedback БЕЗ updated_at, БЕЗ CHECK
    conn.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_id INTEGER NOT NULL UNIQUE,
            url TEXT NOT NULL, source TEXT, title TEXT, summary TEXT,
            llm_score INTEGER, verdict TEXT NOT NULL,
            chat_id TEXT, message_id INTEGER, created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE feedback_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, source TEXT, title TEXT, summary TEXT,
            llm_score INTEGER, kind TEXT NOT NULL,
            chat_id TEXT, message_id INTEGER, created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # DDL без CHECK — probe должен обнаружить это и trigger RECREATE
    ddl = sqlite3.connect(str(db_path)).execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'"
    ).fetchone()[0]
    assert "CHECK" not in ddl, "Исходная v7 БД не имеет CHECK в DDL"

    # Миграция v7→v8
    s = Storage(db_path)

    # После миграции DDL содержит CHECK
    ddl2 = sqlite3.connect(str(db_path)).execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'"
    ).fetchone()[0]
    assert "CHECK" in ddl2, "После миграции v8 DDL feedback должен содержать CHECK"


def test_record_feedback_unknown_target_returns_false(storage):
    """F6: неизвестный target_id возвращает False."""
    assert storage.record_feedback(99999, "like") is False


# ---------------------------------------------------------------------------
# get_recent_feedback
# ---------------------------------------------------------------------------

def test_get_recent_feedback_order_by_updated_at_desc(storage, news_item):
    """F5: re-vote поднимает запись наверх (updated_at DESC)."""
    t1 = storage.add_feedback_target_from_item(news_item, kind="push")
    time.sleep(0.01)   # разнести timestamps для детерминированного порядка
    storage.record_feedback(t1, "like")

    item2 = NewsItem(
        source="S2", url="https://x.com/2",
        title="T2", summary="S2", published="2026",
        published_parsed=None,
    )
    t2 = storage.add_feedback_target_from_item(item2, kind="push")
    time.sleep(0.01)
    storage.record_feedback(t2, "dislike")

    # re-vote на первом item
    time.sleep(0.01)
    storage.record_feedback(t1, "like")

    rows = storage.get_recent_feedback()
    assert len(rows) == 2
    assert rows[0]["target_id"] == t1, "re-vote на t1 должен поднять его наверх"


def test_get_recent_feedback_filter_verdict(storage, news_item):
    """Фильтр по verdict."""
    t1 = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(t1, "like")
    item2 = NewsItem(source="S2", url="https://x.com/2", title="T2",
                     summary="S2", published="2026", published_parsed=None)
    t2 = storage.add_feedback_target_from_item(item2, kind="push")
    storage.record_feedback(t2, "dislike")

    likes = storage.get_recent_feedback(verdict="like")
    assert len(likes) == 1 and likes[0]["verdict"] == "like"
    dislikes = storage.get_recent_feedback(verdict="dislike")
    assert len(dislikes) == 1 and dislikes[0]["verdict"] == "dislike"


def test_get_recent_feedback_limit(storage, news_item):
    """Лимит работает."""
    ids = []
    for i in range(20):
        item = NewsItem(source="S", url=f"https://x.com/{i}",
                        title=f"T{i}", summary="S", published="2026",
                        published_parsed=None)
        tid = storage.add_feedback_target_from_item(item, kind="push")
        storage.record_feedback(tid, "like")
        ids.append(tid)

    rows = storage.get_recent_feedback(limit=5)
    assert len(rows) == 5
    assert rows[0]["target_id"] == ids[-1]


def test_get_recent_feedback_no_chat_id_in_result(storage, news_item):
    """NIT: chat_id/message_id НЕ попадают в результат (не для few-shot)."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")
    rows = storage.get_recent_feedback()
    r = rows[0]
    assert "title" in r and "summary" in r and "url" in r and "source" in r
    assert "chat_id" not in r
    assert "message_id" not in r


def test_get_recent_feedback_empty(storage):
    assert storage.get_recent_feedback() == []


# ---------------------------------------------------------------------------
# meta + tg_updates_offset
# ---------------------------------------------------------------------------

def test_meta_get_set(storage):
    assert storage.get_meta("key") is None
    storage.set_meta("key", "val")
    assert storage.get_meta("key") == "val"


def test_meta_get_default(storage):
    assert storage.get_meta("nonexistent", default="fallback") == "fallback"


def test_meta_overwrite(storage):
    storage.set_meta("k", "v1")
    storage.set_meta("k", "v2")
    assert storage.get_meta("k") == "v2"


def test_tg_updates_offset_meta(storage):
    storage.set_meta("tg_updates_offset", "1234567890")
    assert storage.get_meta("tg_updates_offset") == "1234567890"


# ---------------------------------------------------------------------------
# prune_feedback_targets + F1 интеграция
# ---------------------------------------------------------------------------

def test_prune_deletes_old_targets_preserves_feedback(storage, news_item):
    """F3: prune удаляет старые targets, feedback сохраняется."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")

    # подменим created_at на 100 дней назад
    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    conn = storage._connect()
    conn.execute("UPDATE feedback_targets SET created_at = ? WHERE id = ?", (old_date, tid))
    conn.commit()

    deleted = storage.prune_feedback_targets(older_than_days=30)
    assert deleted == 1
    assert storage.get_feedback_target(tid) is None
    rows = storage.get_recent_feedback()
    assert len(rows) == 1 and rows[0]["target_id"] == tid


def test_prune_noop_when_fresh(storage, news_item):
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    assert storage.prune_feedback_targets(older_than_days=30) == 0


def test_purge_items_calls_prune_feedback_targets(storage, news_item):
    """F1: purge_items с feedback_retention_days вызывает prune."""
    tid = storage.add_feedback_target_from_item(news_item, kind="push")
    storage.record_feedback(tid, "like")

    old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    conn = storage._connect()
    conn.execute("UPDATE feedback_targets SET created_at = ? WHERE id = ?", (old_date, tid))
    conn.commit()

    deleted = storage.purge_items(
        retention_days=30,
        feedback_retention_days=30,
    )
    assert deleted == 0
    assert storage.get_feedback_target(tid) is None
    assert len(storage.get_recent_feedback()) == 1


# ---------------------------------------------------------------------------
# существующие тесты (sanity)
# ---------------------------------------------------------------------------

def test_existing_save_and_retrieve(storage, news_item):
    inserted = storage.save(news_item)
    assert inserted is True


def test_existing_state_transitions(storage, news_item):
    storage.save(news_item)
    updated = storage.set_state([news_item.url], "pending")
    assert updated == 1


def test_existing_recover_stuck_pending(storage):
    assert storage.recover_stuck_pending(max_retries=8) == 0


def test_existing_purge_items_empty(storage):
    assert storage.purge_items(retention_days=30) == 0


def test_existing_purge_seen_titles(storage):
    assert storage.purge_seen_titles() == 0