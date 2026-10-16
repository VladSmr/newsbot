"""
SQLite-хранилище: items + seen_titles.
state-машина:
  new       — только что обнаружен, ещё не обработан
  pending   — попытка обработки/доставки начата (защита от повторной в том же цикле)
  delivered — успешно обработан и доставлен
  skipped   — LLM отфильтровал (retry не трогает)
  deferred  — score 5–6, отложен на дайджест (retry не трогает)
  failed    — терминальный; retry_count исчерпан или recovery на старте цикла
  digested  — терминальный; отправлен в дайджесте

Дедуп:
  (a) точный URL — ВСЕ записи (не зависит от state)
  (b) нормализованный заголовок + source — TTL 30 дней

INSERT OR IGNORE + UNIQUE-констрейнты защищают от дубликатов.
"""
import contextlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, List, Optional, Set

from .collector import NewsItem

logger = logging.getLogger("newsbot.storage")

_TITLE_DEDUP_TTL_DAYS = 30
# N4: минимальная длина нормализованного заголовка для title-dedup.
# Пустые/короткие (<3 симв.) не индексируются в seen_titles
# и не проверяются по title-dedup во избежание ложного схлопывания.
MIN_TITLE_LEN = 3


class Storage:
    """
    SQLite-файл с двумя таблицами: items (state-машина), seen_titles.
    Схема мигрирует автоматически.
    Идемпотентность: user_version выставляется ПОСЛЕ успешной миграции.
    """

    SCHEMA_VERSION = 10
    STATES = ("new", "pending", "delivered", "skipped", "deferred", "failed", "digested")
    DEFAULT_RETENTION_DAYS = 90
    DEFAULT_RETRY_THRESHOLD_SECONDS = 60
    # Лимит попыток для retry: terminal state при исчерпании.
    DEFAULT_MAX_RETRIES = 8
    # Порог для warning по deferred-items.
    DEFERRED_WARN_THRESHOLD = 500
    # Окно хранения deferred-items: старые неотправленные новости удаляются.
    # 14 дней достаточно: deferred накапливаются между дайджестами (09:00/21:00),
    # при паузе или длительной недоступности дайджеста — не более 14 суток.
    # Для VPS с 1 GB RAM это критично: низкоско́рые deferred накапливались
    # бесконечно (digest берёт топ-30, остаток копился вечно).
    DEFAULT_DEFERRED_RETENTION_DAYS = 14

    def __init__(self, db_path: Path, dedup_config: Optional[dict] = None):
        """
        Args:
            db_path: путь к БД.
            dedup_config: конфиг дедупликации (опционально, дефолт=пустой dict).
                          Принимается для обратной совместимости вызовов извне,
                          НЕ используется внутри Storage; параметр хранится
                          только чтобы не сломать уже существующие вызовы.
                          Кросс-источниковая дедупликация реализована в dedup.filter_new.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._has_title_normalized = False  # перезаписывается в _init_db после миграций
        self._init_db()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._conn.row_factory = sqlite3.Row
            # busy_timeout: ждать до 5 с при блокировке записи (WAL readers/writers).
            self._conn.execute("PRAGMA busy_timeout = 5000")
        return self._conn

    @contextlib.contextmanager
    def _tx(self) -> Generator:
        """
        Контекстный менеджер транзакции.
        Устраняет хрупкие ROLLBACK-в-except.
        """
        conn = self._connect()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def _current_version(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        current_version = self._current_version()

        # ── 1. Миграция ──
        self._migrate(conn, current_version)

        # ── 2. Таблицы (CREATE IF NOT EXISTS — идемпотентны) ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT    NOT NULL,
                url         TEXT    NOT NULL,
                title       TEXT    NOT NULL,
                summary     TEXT    NOT NULL,
                published   TEXT,
                fetched_at  TEXT    NOT NULL,
                state       TEXT    NOT NULL DEFAULT 'new',
                state_at    TEXT    NOT NULL,
                llm_score   INTEGER,
                llm_summary TEXT,
                llm_reason  TEXT,
                llm_urgent  INTEGER,
                filtered_at TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_titles (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                title_hash       TEXT    NOT NULL,
                source           TEXT,
                fetched_at       TEXT    NOT NULL,
                expires_at       TEXT    NOT NULL,
                title_normalized TEXT
            )
        """)

        # ── 2b. v9: добавить title_normalized если отсутствует (идемпотентно) ──
        # Вызывается после CREATE TABLE, поэтому seen_titles гарантированно существует.
        # Также вызывается из миграции v9 для старых БД, где таблица уже есть.
        # Идемпотентно: try/except OperationalError (duplicate column).
        try:
            st_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(seen_titles)"
            ).fetchall()}
            if "title_normalized" not in st_cols:
                conn.execute(
                    "ALTER TABLE seen_titles ADD COLUMN title_normalized TEXT"
                )
                logger.info("v9: добавлена колонка seen_titles.title_normalized")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
            logger.info("v9: колонка title_normalized уже существует")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # ── 2a. feedback-таблицы (N2: всегда, даже при актуальной версии) ──
        # Перенесены из миграции v7 сюда, чтобы отсутствующие таблицы
        # создавались при ЛЮБОЙ user_version >= 7 (F7).
        for stmt in (
            """
            CREATE TABLE IF NOT EXISTS feedback_targets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL,
                source      TEXT,
                title       TEXT,
                summary     TEXT,
                llm_score   INTEGER,
                kind        TEXT NOT NULL,
                chat_id     TEXT,
                message_id  INTEGER,
                created_at  TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id   INTEGER NOT NULL,
                url         TEXT NOT NULL,
                source      TEXT,
                title       TEXT,
                summary     TEXT,
                llm_score   INTEGER,
                verdict     TEXT NOT NULL CHECK(verdict IN ('like','dislike')),
                chat_id     TEXT,
                message_id  INTEGER,
                created_at  TEXT NOT NULL,
                updated_at  TEXT,
                UNIQUE(target_id)
            )
            """,
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # already exists или конкурентная миграция

        # ── 3. Индексы ──
        self._create_indexes(conn)

        # ── 3a. feedback-индексы (Фаза 2a / F7: нужны даже при актуальной версии) ──
        self._create_feedback_indexes(conn)

        # ── 4. Версия — только после успешной миграции и CREATE TABLE ──
        conn.execute("PRAGMA user_version = %d" % self.SCHEMA_VERSION)
        conn.commit()

        # ── 5. Определить наличие title_normalized в seen_titles (v9) ──
        # Проверяем ПОСЛЕ всех миграций (включая v9), после коммита.
        # Для старых БД (v<9) — колонки нет; для свежих v9 и после миграции — есть.
        st_cols = {r[1] for r in conn.execute("PRAGMA table_info(seen_titles)").fetchall()}
        self._has_title_normalized = "title_normalized" in st_cols

        # ── 5b. v10: defensive post-check для llm_urgent ───────────────────
        # При user_version=10 без колонки (редкий edge case при конкурентной
        # миграции) — добавляем колонку идемпотентно.
        # Это защита: если миграция v10 ещё не выполнена для этого соединения,
        # но user_version уже 10, SELECT по llm_urgent упадёт.
        # Идемпотентно: try/except OperationalError (duplicate column).
        it_cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        if "llm_urgent" not in it_cols:
            try:
                conn.execute("ALTER TABLE items ADD COLUMN llm_urgent INTEGER")
                logger.info("v10 post-check: добавлена колонка items.llm_urgent")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                logger.info("v10 post-check: колонка llm_urgent уже существует")

        logger.debug("БД инициализирована (v%d): %s", self.SCHEMA_VERSION, self.db_path)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Создаёт индексы. IntegrityError = дубли в old data, логируем явно."""
        for stmt in (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url ON items(url)",
            "CREATE INDEX IF NOT EXISTS idx_items_state ON items(state)",
            "CREATE INDEX IF NOT EXISTS idx_items_state_at ON items(state_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_seen_title_source "
            "ON seen_titles(title_hash, source)",
            "CREATE INDEX IF NOT EXISTS idx_seen_ttl ON seen_titles(expires_at)",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.IntegrityError as e:
                logger.warning(
                    "UNIQUE index пропущен — дубли в существующих данных: %s. "
                    "Команда: %s",
                    e, stmt[:80],
                )
            except sqlite3.OperationalError:
                pass

    def _create_feedback_indexes(self, conn: sqlite3.Connection) -> None:
        """
        Создаёт индексы для feedback_targets и feedback.
        F7: вызывается из _init_db, чтобы создать индексы даже при актуальной версии.
        CREATE INDEX IF NOT EXISTS = идемпотентно. Проверка наличия таблицы —
        по sqlite_master без row_factory (используем str(r[0]) на tuple).

        F1: idx_feedback_targets_url_kind ускоряет get_or_create_feedback_target
        (SELECT WHERE url=? AND kind=?). Составной, а не два отдельных,
        потому что в БД бывают и push, и digest для одного URL.
        """
        existing_tables = {str(r[0]) for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "feedback_targets" in existing_tables:
            # F1: для get_or_create (составной: url + kind)
            self._create_index_ignore_errors(
                conn, "idx_feedback_targets_url_kind",
                "feedback_targets", "url, kind",
            )
            self._create_index_ignore_errors(
                conn, "idx_feedback_targets_created",
                "feedback_targets", "created_at",
            )

        if "feedback" in existing_tables:
            self._create_index_ignore_errors(
                conn, "idx_feedback_verdict_created",
                "feedback", "verdict, created_at",
            )
            self._create_index_ignore_errors(
                conn, "idx_feedback_created",
                "feedback", "created_at",
            )
            self._create_index_ignore_errors(
                conn, "idx_feedback_updated",
                "feedback", "updated_at",
            )

    def _create_index_ignore_errors(
        self, conn: sqlite3.Connection, name: str, table: str, cols: str,
    ) -> None:
        """
        Создаёт индекс IF NOT EXISTS; пропускает IntegrityError / OperationalError.
        Не проверяет существование индекса заранее — IF NOT EXISTS достаточно.
        """
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {name} ON {table}({cols})"
            )
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            pass

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """
        Миграция from_version → SCHEMA_VERSION.
        user_version обновляется ТОЛЬКО в _init_db после успешного завершения.
        При падении посередине — повторный старт продолжит с того же места
        (идемпотентность: IF NOT EXISTS, IF NOT EXISTS, etc.).
        """
        existing_tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        # cols читается ВСЕГДА когда таблица items существует — не только
        # для ветки v3-v4, но и для v6, чтобы избежать UnboundLocalError.
        if "items" in existing_tables:
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
        else:
            cols = set()  # таблицы нет — колонок тоже нет

        # st_cols читается ВСЕГДА когда seen_titles существует (v9)
        if "seen_titles" in existing_tables:
            st_cols = {r[1] for r in conn.execute("PRAGMA table_info(seen_titles)").fetchall()}
        else:
            st_cols = set()

        # Ранний return: если версия актуальна — миграция не нужна.
        # Все миграционные шаги идемпотентны, но return экономит работу.
        # При падении до return — повторный старт накатит миграцию с нуля.
        if from_version >= self.SCHEMA_VERSION:
            conn.commit()  # checkpoint WAL
            return

        logger.info("Миграция БД: v%d → v%d", from_version, self.SCHEMA_VERSION)
        now = datetime.now(timezone.utc).isoformat()

        # (a) колонка state — только если items уже существует
        if "items" in existing_tables:
            # (a) колонка state
            if "state" not in cols:
                try:
                    conn.execute(
                        "ALTER TABLE items ADD COLUMN state TEXT NOT NULL DEFAULT 'new'"
                    )
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("state column already exists (concurrent migration)")
                else:
                    # мигрируем delivered=1 → delivered
                    if "delivered" in cols:
                        conn.execute(
                            "UPDATE items SET state = 'delivered' WHERE delivered = 1"
                        )
                        # DROP COLUMN delivered требует SQLite >= 3.35.0
                        # Старая колонка просто остаётся (не используется, harmless).
                        try:
                            conn.execute("ALTER TABLE items DROP COLUMN delivered")
                        except sqlite3.OperationalError as e:
                            logger.info(
                                "DROP COLUMN delivered пропущен (%s). "
                                "Колонка не используется, harmless.",
                                e,
                            )
                    logger.info("Миграция: добавлена колонка state, delivered -> state")

            # (б) колонка state_at — литеральный DEFAULT (SQLite не принимает ?)
            if "state_at" not in cols:
                try:
                    conn.execute(
                        "ALTER TABLE items ADD COLUMN state_at TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("state_at column already exists (concurrent migration)")
                else:
                    # backfill: все строки получают now как state_at
                    conn.execute(
                        "UPDATE items SET state_at = ? WHERE state_at = '' OR state_at IS NULL",
                        (now,),
                    )
                    logger.info("Миграция: добавлена колонка state_at")

            # (в) title_hashes → seen_titles
            if "title_hashes" in existing_tables and "seen_titles" not in existing_tables:
                try:
                    conn.execute("ALTER TABLE title_hashes RENAME TO seen_titles")
                    logger.info("Миграция: title_hashes → seen_titles")
                except sqlite3.OperationalError:
                    pass

            # (г) v4: llm_score, llm_summary, llm_reason, filtered_at
            for col_name, col_def in (
                ("llm_score", "INTEGER"),
                ("llm_summary", "TEXT"),
                ("llm_reason", "TEXT"),
                ("filtered_at", "TEXT"),
            ):
                if col_name not in cols:
                    try:
                        conn.execute(
                            f"ALTER TABLE items ADD COLUMN {col_name} {col_def}"
                        )
                        logger.info("Миграция v4: добавлена колонка %s", col_name)
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise
                        logger.info("v4 column %s already exists (concurrent migration)", col_name)

        # (д) v5: таблица meta (идемпотентна — IF NOT EXISTS в _init_db)
        if "meta" not in existing_tables:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            logger.info("Миграция v5: создана таблица meta")

        # (е) v6: retry_count — только для существующих таблиц с версией < 6
        # cols читается выше, гарантированно определён.
        if from_version < 6 and "items" in existing_tables:
            for col_name, col_def in (
                ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if col_name not in cols:
                    try:
                        conn.execute(
                            f"ALTER TABLE items ADD COLUMN {col_name} {col_def}"
                        )
                        logger.info("Миграция v6: добавлена колонка %s", col_name)
                    except sqlite3.OperationalError as e:
                        # Конкурентная миграция: колонка уже добавлена параллельным процессом.
                        if "duplicate column" not in str(e).lower():
                            raise
                        logger.info(
                            "Миграция v6: колонка %s уже существует (конкурентный процесс).",
                            col_name,
                        )

        # (ж) v7: feedback_targets + feedback
        # Идемпотентно: IF NOT EXISTS + try/except для конкурентной миграции.
        if from_version < 7:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_targets (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        url         TEXT NOT NULL,
                        source      TEXT,
                        title       TEXT,
                        summary     TEXT,
                        llm_score   INTEGER,
                        kind        TEXT NOT NULL,
                        chat_id     TEXT,
                        message_id  INTEGER,
                        created_at  TEXT NOT NULL
                    )
                """)
                logger.info("Миграция v7: создана таблица feedback_targets")
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e).lower():
                    raise
                logger.info("Миграция v7: таблица feedback_targets уже существует")

            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        target_id   INTEGER NOT NULL,
                        url         TEXT NOT NULL,
                        source      TEXT,
                        title       TEXT,
                        summary     TEXT,
                        llm_score   INTEGER,
                        verdict     TEXT NOT NULL CHECK(verdict IN ('like','dislike')),
                        chat_id     TEXT,
                        message_id  INTEGER,
                        created_at  TEXT NOT NULL,
                        updated_at  TEXT,
                        UNIQUE(target_id)
                    )
                """)
                logger.info("Миграция v7: создана таблица feedback")
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e).lower():
                    raise
                logger.info("Миграция v7: таблица feedback уже существует")

            # Индексы feedback_targets + feedback создаются ВЕЗДЕ из _init_db
            # через _create_feedback_indexes (F7), здесь не дублируем.

        # (з) v8: доводка таблицы feedback:
        #   — добавить updated_at (если нет)
        #   — добавить CHECK(verdict) через RECREATE TABLE (SQLite ALTER TABLE
        #     не умеет добавлять CHECK; штатный паттерн — пересоздание с данными)
        #
        # existing_tables снята ДО шага v7, поэтому feedback из текущего
        # запуска там нет. Используем CURRENT список таблиц через PRAGMA.
        if from_version >= 7:
            current_tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "feedback" in current_tables:
                fb_cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()}
                needs_updated_at = "updated_at" not in fb_cols

                # N4: определяем CHECK по DDL из sqlite_master (без записи в таблицу)
                fb_ddl = (conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'"
                ).fetchone() or [""])[0]
                has_check = "CHECK" in fb_ddl

                if needs_updated_at or not has_check:
                    # N3: читаем данные ДО пересоздания; нормализуем verdict
                    old_rows = [tuple(row) for row in conn.execute("SELECT * FROM feedback").fetchall()]
                    old_cols = [r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()]

                    # N2: явная транзакция для RECREATE (атомарность)
                    conn.execute("BEGIN")
                    try:
                        conn.execute("DROP TABLE IF EXISTS _feedback_new")
                        conn.execute("""
                            CREATE TABLE _feedback_new (
                                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                                target_id   INTEGER NOT NULL,
                                url         TEXT NOT NULL,
                                source      TEXT,
                                title       TEXT,
                                summary     TEXT,
                                llm_score   INTEGER,
                                verdict     TEXT NOT NULL CHECK(verdict IN ('like','dislike')),
                                chat_id     TEXT,
                                message_id  INTEGER,
                                created_at  TEXT NOT NULL,
                                updated_at  TEXT,
                                UNIQUE(target_id)
                            )
                        """)

                        if old_rows:
                            # N1-CHECK: фильтруем строки с невалидным verdict;
                            # если CHECK нарушится при копировании — логируем и пропускаем
                            placeholders = ",".join(["?"] * len(old_cols))
                            for row in old_rows:
                                # определяем позицию verdict по old_cols
                                try:
                                    conn.execute(
                                        f"INSERT INTO _feedback_new ({','.join(old_cols)}) "
                                        f"VALUES ({placeholders})",
                                        row,
                                    )
                                except sqlite3.IntegrityError:
                                    # N1-CHECK: пропускаем строку с невалидным verdict
                                    verdict_val = row[old_cols.index("verdict")] if "verdict" in old_cols else "?"
                                    logger.warning(
                                        "Миграция v8: пропущена строка feedback id=%s "
                                        "(невалидный verdict=%r). "
                                        "Данные удалены; рекомендуется проверить источник.",
                                        row[old_cols.index("id")] if "id" in old_cols else "?",
                                        verdict_val,
                                    )

                        conn.execute("DROP TABLE feedback")
                        conn.execute("ALTER TABLE _feedback_new RENAME TO feedback")
                        conn.execute("COMMIT")
                        logger.info("Миграция v8: feedback пересоздана с updated_at + CHECK(verdict), "
                                    "строк: %d", len(old_rows))
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

        # (и) v9: seen_titles.title_normalized для fuzzy-дедупа (кросс-источниковый)
        # Идемпотентно: try/except для конкурентной миграции.
        # existing_tables снята ДО v7; проверяем текущее состояние.
        if from_version >= 8:
            current_st_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(seen_titles)"
            ).fetchall()} if "seen_titles" in {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()} else set()
            if "title_normalized" not in current_st_cols:
                try:
                    conn.execute(
                        "ALTER TABLE seen_titles ADD COLUMN title_normalized TEXT"
                    )
                    logger.info("Миграция v9: добавлена колонка seen_titles.title_normalized")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v9: колонка title_normalized уже существует (конкурентный процесс)")

        # (й) v10: llm_urgent INTEGER — персистентный флаг срочности из LLM.
        # Идемпотентно: try/except для конкурентной миграции.
        # ВЫНЕСЕН из блока `if from_version >= 8`: должен выполняться для ВСЕХ
        # from_version < 10, включая v3, v4, v5 (миграция v8 для них пропущена,
        # поэтому ранний return выше не блокирует).
        # Для свежей БД (v0) таблица items ещё НЕ создана — guard `if "items" in existing_tables`.
        if "items" in existing_tables:
            current_items_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            if "llm_urgent" not in current_items_cols:
                try:
                    conn.execute(
                        "ALTER TABLE items ADD COLUMN llm_urgent INTEGER"
                    )
                    logger.info("Миграция v10: добавлена колонка items.llm_urgent")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v10: колонка llm_urgent уже существует (конкурентный процесс)")

        conn.commit()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_duplicate(self, item: NewsItem) -> bool:
        """
        (a) точный URL — ВСЕ записи (permanent dedup)
        (b) нормализованный заголовок + source в пределах TTL
            (per-source дедуп, существующее поведение)

        NOTE: кросс-источниковый точный дедуп (уровень 1) НЕ здесь —
        он обрабатывается в filter_new (dedup.py), где при срабатывании
        выполняется save_skipped. is_duplicate — только read-only проверка.

        Возвращает True если дубликат.
        """
        conn = self._connect()

        row = conn.execute(
            "SELECT 1 FROM items WHERE url = ? LIMIT 1",
            (item.url,),
        ).fetchone()
        if row:
            return True

        normalized = item.title_normalized()
        # Не проверяем title-dedup для пустых / слишком коротких заголовков.
        if len(normalized) < MIN_TITLE_LEN:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()

        # (b) per-source дедуп — всегда (существующее поведение)
        row2 = conn.execute(
            """
            SELECT 1 FROM seen_titles
            WHERE  title_hash = ?
              AND  source = ?
              AND  expires_at > ?
            LIMIT 1
            """,
            (normalized, item.source, now_iso),
        ).fetchone()
        return row2 is not None

    def save(self, item: NewsItem) -> bool:
        """
        Вставляет item (если URL ещё не в БД).
        Возвращает True если ВСТАВЛЕН, False если IGNORE.

        Summary обрезается до MAX_SUMMARY_CHARS (6000) для защиты LLM-бюджета.
        """
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        expires_iso = (now_utc + timedelta(days=_TITLE_DEDUP_TTL_DAYS)).isoformat()
        normalized = item.title_normalized()

        # Cap summary — защита от чрезмерных LLM-токенов и БД-размера.
        MAX_SUMMARY_CHARS = 6000
        summary = item.summary[:MAX_SUMMARY_CHARS]

        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, title, summary, published, fetched_at, state, state_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'new', ?)
                    """,
                    (
                        item.source,
                        item.url,
                        item.title,
                        summary,
                        item.published,
                        now_iso,
                        now_iso,
                    ),
                )
                inserted = cur.rowcount

                # seen_titles индексируется только для осмысленных заголовков
                # (normalized >= MIN_TITLE_LEN) — чтобы пустые title не ложно схлопывались.
                if len(normalized) >= MIN_TITLE_LEN:
                    if self._has_title_normalized:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO seen_titles
                            (title_hash, source, fetched_at, expires_at, title_normalized)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (normalized, item.source, now_iso, expires_iso, normalized),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO seen_titles
                            (title_hash, source, fetched_at, expires_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (normalized, item.source, now_iso, expires_iso),
                        )

            return inserted > 0
        except Exception as e:
            logger.warning("Ошибка save(%s): %s", item.title[:60], e)
            return False

    def save_many(self, items: List[NewsItem]) -> List[NewsItem]:
        """
        Батчевое сохранение items в ОДНОЙ транзакции (INSERT OR IGNORE).
        N1: возвращает список РЕАЛЬНО вставленных NewsItem (для pipeline).

        Summary обрезается до MAX_SUMMARY_CHARS (6000).
        """
        if not items:
            return []
        MAX_SUMMARY_CHARS = 6000
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        expires_iso = (now_utc + timedelta(days=_TITLE_DEDUP_TTL_DAYS)).isoformat()

        inserted_items: List[NewsItem] = []
        seen_keys: Set[str] = set()  # для дедупа по title+source внутри батча

        try:
            with self._tx() as conn:
                for item in items:
                    summary = item.summary[:MAX_SUMMARY_CHARS]
                    normalized = item.title_normalized()

                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO items
                        (source, url, title, summary, published, fetched_at,
                         state, state_at, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?, 'new', ?, 0)
                        """,
                        (
                            item.source,
                            item.url,
                            item.title,
                            summary,
                            item.published,
                            now_iso,
                            now_iso,
                        ),
                    )
                    if cur.rowcount:
                        inserted_items.append(item)

                        # seen_titles — только для осмысленных заголовков
                        if len(normalized) >= MIN_TITLE_LEN:
                            # внутрибатчевый дедуп: не добавлять повторный key
                            key = f"{normalized}|{item.source}"
                            if key not in seen_keys:
                                if self._has_title_normalized:
                                    conn.execute(
                                        """
                                        INSERT OR IGNORE INTO seen_titles
                                        (title_hash, source, fetched_at, expires_at, title_normalized)
                                        VALUES (?, ?, ?, ?, ?)
                                        """,
                                        (normalized, item.source, now_iso, expires_iso, normalized),
                                    )
                                else:
                                    conn.execute(
                                        """
                                        INSERT OR IGNORE INTO seen_titles
                                        (title_hash, source, fetched_at, expires_at)
                                        VALUES (?, ?, ?, ?)
                                        """,
                                        (normalized, item.source, now_iso, expires_iso),
                                    )
                                seen_keys.add(key)

            return inserted_items
        except Exception as e:
            logger.warning("Ошибка save_many(%d items): %s", len(items), e)
            return []

    def set_state(self, urls: List[str], new_state: str) -> int:
        """
        Меняет state и state_at для списка URLs батчами по 200.
        state_at обновляется ВСЕГДА (включая pending→pending и new→new) —
        это гарантирует корректный расчёт get_stuck_pending.age.
        state='failed' — терминальный; при достижении лимита retry.

        Возвращает общее количество обновлённых строк.
        """
        if not urls or new_state not in self.STATES:
            return 0
        total = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        batch_size = 200

        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            try:
                with self._tx() as conn:
                    cur = conn.execute(
                        f"""
                        UPDATE items
                        SET state = ?, state_at = ?
                        WHERE url IN ({placeholders})
                        """,
                        [new_state, now_iso] + batch,
                    )
                total += cur.rowcount
            except Exception as e:
                logger.warning(
                    "set_state(%s, batch %d..%d): %s",
                    new_state, i, i + len(batch), e,
                )
        return total

    def get_stuck_pending(
        self,
        max_age_seconds: int = None,
        max_retries: int = None,
    ) -> List[NewsItem]:
        """
        Возвращает items state='pending' старше max_age_seconds
        с retry_count < max_retries (не терминальные).
        'failed' items сюда не попадают: они терминальные, не retry'ятся.
        """
        if max_age_seconds is None:
            max_age_seconds = self.DEFAULT_RETRY_THRESHOLD_SECONDS
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_RETRIES
        conn = self._connect()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        ).isoformat()
        rows = conn.execute(
            """
            SELECT source, url, title, summary, published, fetched_at, state_at,
                   llm_score, llm_summary, llm_reason, llm_urgent, retry_count
            FROM items
            WHERE state = 'pending'
              AND state_at < ?
              AND retry_count < ?
            ORDER BY id ASC
            """,
            (cutoff, max_retries),
        ).fetchall()
        return self._rows_to_items(rows, with_retry_count=True)

    def increment_retry_count(self, urls: List[str]) -> int:
        """
        Инкрементирует retry_count для списка URLs.
        Возвращает количество обновлённых строк.
        """
        if not urls:
            return 0
        total = 0
        batch_size = 200
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            try:
                with self._tx() as conn:
                    cur = conn.execute(
                        f"""
                        UPDATE items
                        SET retry_count = retry_count + 1
                        WHERE url IN ({placeholders})
                        """,
                        batch,
                    )
                total += cur.rowcount
            except Exception as e:
                logger.warning(
                    "increment_retry_count batch %d..%d: %s",
                    i, i + len(batch), e,
                )
        return total

    def set_failed(self, urls: List[str]) -> int:
        """
        Переводит URLs в терминальный state='failed'.
        Возвращает количество обновлённых строк.
        """
        return self.set_state(urls, "failed")

    def get_urls_with_retry_gte(self, urls: List[str], threshold: int) -> List[str]:
        """
        N5: возвращает из переданного списка URLs только те,
        у которых retry_count >= threshold.
        Батчинг по ~500 для надёжности при большом списке.
        """
        if not urls:
            return []
        result: List[str] = []
        batch_size = 500
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            try:
                conn = self._connect()
                rows = conn.execute(
                    f"""
                    SELECT url FROM items
                    WHERE url IN ({placeholders})
                      AND retry_count >= ?
                    """,
                    batch + [threshold],
                ).fetchall()
                result.extend(r[0] for r in rows)
            except Exception as e:
                logger.warning(
                    "get_urls_with_retry_gte batch %d..%d: %s",
                    i, i + len(batch), e,
                )
        return result

    def mark_failed_if_retries_exhausted(self, urls: List[str], max_retries: int) -> int:
        """
        N5: переводит URLs с retry_count >= max_retries в 'failed'.
        Батчинг по ~500 для надёжности при большом списке.
        Возвращает количество маркированных строк.
        """
        if not urls:
            return 0
        total = 0
        batch_size = 500
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            try:
                with self._tx() as conn:
                    cur = conn.execute(
                        f"""
                        UPDATE items
                        SET state = 'failed', state_at = ?
                        WHERE url IN ({placeholders})
                          AND retry_count >= ?
                        """,
                        [datetime.now(timezone.utc).isoformat()] + batch + [max_retries],
                    )
                total += cur.rowcount
            except Exception as e:
                logger.warning(
                    "mark_failed_if_retries_exhausted batch %d..%d: %s",
                    i, i + len(batch), e,
                )
        if total:
            logger.info(
                "mark_failed_if_retries_exhausted: %d items → 'failed'",
                total,
            )
        return total

    def recover_stuck_pending(self, max_retries: int, retry_threshold_sec: int = None) -> int:
        """
        C2: на старте цикла переводит зависшие 'pending' с retry_count >= max_retries
        в терминальный 'failed'. Защита от зависания: если max_retries уменьшили
        в конфиге — старые items с retry_count >= нового лимита получат 'failed'.

        Args:
            max_retries: лимит попыток
            retry_threshold_sec: окно «зависания» — pending старше этого
                                 порога. Если None — DEFAULT_RETRY_THRESHOLD_SECONDS.
        """
        if retry_threshold_sec is None:
            retry_threshold_sec = self.DEFAULT_RETRY_THRESHOLD_SECONDS
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=retry_threshold_sec)
        ).isoformat()
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    UPDATE items
                    SET state = 'failed', state_at = ?
                    WHERE state = 'pending'
                      AND retry_count >= ?
                      AND state_at < ?
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        max_retries,
                        cutoff,
                    ),
                )
            if cur.rowcount:
                logger.warning(
                    "C2-recovery: %d зависших items (pending, retry_count>=%d) "
                    "переведены в 'failed'",
                    cur.rowcount, max_retries,
                )
            return cur.rowcount
        except Exception as e:
            logger.warning("recover_stuck_pending: %s", e)
            return 0

    def get_all_by_state(self, state: str, limit: Optional[int] = None) -> List[NewsItem]:
        conn = self._connect()
        sql = """
            SELECT source, url, title, summary, published, fetched_at, state_at,
                   llm_score, llm_summary, llm_reason, llm_urgent
            FROM items
            WHERE state = ?
            ORDER BY id ASC
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (state,)).fetchall()
        return self._rows_to_items(rows)

    def count_by_state(self, state: str) -> int:
        """
        Возвращает количество items с заданным state.
        Не загружает объекты в память — использует COUNT(*) в БД.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM items WHERE state = ?",
            (state,),
        ).fetchone()
        return row[0] if row else 0

    def get_deferred_items(self, limit: Optional[int] = None) -> List[NewsItem]:
        """
        Возвращает items со state='deferred', отсортированные:
          llm_score DESC (лучшие первыми), затем fetched_at ASC.
        Лимит опционален (для дайджеста: top-N по score).
        """
        conn = self._connect()
        sql = """
            SELECT source, url, title, summary, published, fetched_at, state_at,
                   llm_score, llm_summary, llm_reason, llm_urgent
            FROM items
            WHERE state = 'deferred'
            ORDER BY llm_score DESC, fetched_at ASC
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        return self._rows_to_items(rows)

    def set_digested(self, urls: List[str]) -> int:
        """
        Переводит URLs в терминальный state='digested'.
        Возвращает количество обновлённых строк.
        """
        return self.set_state(urls, "digested")

    def _rows_to_items(self, rows: List, with_retry_count: bool = False) -> List[NewsItem]:
        """
        Конвертирует sqlite3.Row в NewsItem.
        sqlite3.Row поддерживает только целочисленное индексирование и
        строковое (row["col"]) — НЕ .get(). Проверяем None явно.

        Args:
            rows: список sqlite3.Row
            with_retry_count: если True — читает колонку retry_count (может
                              отсутствовать в старых записях до миграции v6)
        """
        results = []
        for row in rows:
            fetched_at_raw = row["fetched_at"]
            if fetched_at_raw:
                try:
                    fetched_at = datetime.fromisoformat(fetched_at_raw)
                except Exception:
                    fetched_at = datetime.now(timezone.utc)
            else:
                fetched_at = datetime.now(timezone.utc)

            llm_score = None
            llm_summary = None
            llm_reason = None
            retry_count = 0

            if with_retry_count:
                try:
                    retry_count = int(row["retry_count"] or 0)
                except Exception:
                    retry_count = 0

            # LLM-поля: читаем безопасно (могут отсутствовать в старых записях).
            # v10: llm_urgent читается для retry (C1-фикс).
            llm_urgent: Optional[bool] = None
            for val_name, attr_name in (
                ("llm_score", "llm_score"),
                ("llm_summary", "llm_summary"),
                ("llm_reason", "llm_reason"),
                ("llm_urgent", "llm_urgent"),
            ):
                try:
                    val = row[val_name]
                    if val is not None:
                        if val_name == "llm_score":
                            llm_score = val
                        elif val_name == "llm_summary":
                            llm_summary = str(val)
                        elif val_name == "llm_reason":
                            llm_reason = str(val)
                        elif val_name == "llm_urgent":
                            llm_urgent = bool(val)  # INTEGER 0/1 → bool
                except Exception:
                    pass

            ni = NewsItem(
                source=row["source"],
                url=row["url"],
                title=row["title"],
                summary=row["summary"],
                published=row["published"] or "—",
                published_parsed=None,
                fetched_at=fetched_at,
                llm_score=llm_score,
                llm_summary=llm_summary,
                llm_reason=llm_reason,
                llm_urgent=llm_urgent,
            )
            if with_retry_count:
                ni.retry_count = retry_count  # type: ignore
            results.append(ni)
        return results

    def update_llm_result(
        self,
        url: str,
        score: int,
        summary: str,
        reason: str,
        filtered_at: Optional[str] = None,
        urgent: Optional[bool] = None,
    ) -> bool:
        """
        Сохраняет результат LLM-оценки для одного item по URL.
        Атомарно через _tx. urgent=None означает «не менять» (совместимость
        с legacy-вызовами без этого поля).
        """
        if filtered_at is None:
            filtered_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._tx() as conn:
                if urgent is not None:
                    cur = conn.execute(
                        """
                        UPDATE items
                        SET llm_score = ?, llm_summary = ?, llm_reason = ?,
                            llm_urgent = ?, filtered_at = ?
                        WHERE url = ?
                        """,
                        (score, summary, reason, 1 if urgent else 0, filtered_at, url),
                    )
                else:
                    cur = conn.execute(
                        """
                        UPDATE items
                        SET llm_score = ?, llm_summary = ?, llm_reason = ?,
                            filtered_at = ?
                        WHERE url = ?
                        """,
                        (score, summary, reason, filtered_at, url),
                    )
            return cur.rowcount > 0
        except Exception as e:
            logger.warning("update_llm_result(%s): %s", url[:60], e)
            return False

    def update_llm_results_batch(
        self,
        records: List[dict],
    ) -> int:
        """
        Батчевое сохранение результатов LLM-оценки.
        records — список dict: {url, score, summary, reason, urgent?, filtered_at?}
        Если record содержит ключ 'urgent' (bool) — он пишется в llm_urgent (INTEGER 0/1).
        Для обратной совместимости отсутствие ключа 'urgent' не меняет llm_urgent.
        Возвращает количество обновлённых строк.
        """
        if not records:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        total = 0
        try:
            with self._tx() as conn:
                for rec in records:
                    urgent = rec.get("urgent")
                    if urgent is not None:
                        cur = conn.execute(
                            """
                            UPDATE items
                            SET llm_score = ?, llm_summary = ?, llm_reason = ?,
                                llm_urgent = ?, filtered_at = ?
                            WHERE url = ?
                            """,
                            (
                                rec["score"],
                                rec["summary"],
                                rec["reason"],
                                1 if urgent else 0,
                                rec.get("filtered_at") or now_iso,
                                rec["url"],
                            ),
                        )
                    else:
                        cur = conn.execute(
                            """
                            UPDATE items
                            SET llm_score = ?, llm_summary = ?, llm_reason = ?,
                                filtered_at = ?
                            WHERE url = ?
                            """,
                            (
                                rec["score"],
                                rec["summary"],
                                rec["reason"],
                                rec.get("filtered_at") or now_iso,
                                rec["url"],
                            ),
                        )
                    total += cur.rowcount
            return total
        except Exception as e:
            logger.warning("update_llm_results_batch: %s", e)
            return 0

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Читает значение ключа из meta. Возвращает default если ключ отсутствует.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        """Пишет или обновляет ключ в meta. Полностью идемпотентна."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def purge_items(
        self,
        retention_days: int = None,
        feedback_retention_days: int = None,
        deferred_retention_days: int = None,
    ) -> int:
        """
        Удаляет items старше retention_days:
          - 'new': необработанные новости (при долгой паузе/ошибке накапливаются
            вечно; K1-recovery вызывает get_all_by_state('new') каждый цикл,
            что может загрузить неограниченный объём в RAM).
          - 'deferred': старые отложенные, которые не попали ни в один дайджест
            (накапливались бесконечно при паузе или дайджест-окнах < max_items;
            digest берёт топ-30 по score, остаток копился вечно).
          - 'delivered', 'skipped', 'failed', 'digested': терминальные items
            старше retention_days (классический purge).
        'pending' — не удаляется (активный retry).

        Также вызывает prune_feedback_targets для очистки старых targets
        (feedback при этом СОХРАНЯЕТСЯ — обучающие данные).
        """
        if retention_days is None:
            retention_days = self.DEFAULT_RETENTION_DAYS
        if deferred_retention_days is None:
            deferred_retention_days = self.DEFAULT_DEFERRED_RETENTION_DAYS
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        deferred_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=deferred_retention_days)
        ).isoformat()

        total_deleted = 0
        try:
            with self._tx() as conn:
                # new + терминальные (retention_days)
                cur = conn.execute(
                    """
                    DELETE FROM items
                    WHERE state IN ('new', 'delivered', 'skipped', 'failed', 'digested')
                      AND state_at < ?
                    """,
                    (cutoff,),
                )
                new_and_terminal = cur.rowcount

                # deferred: отдельное окно (deferred_retention_days)
                cur_def = conn.execute(
                    """
                    DELETE FROM items
                    WHERE state = 'deferred'
                      AND state_at < ?
                    """,
                    (deferred_cutoff,),
                )
                deferred_del = cur_def.rowcount

                total_deleted = new_and_terminal + deferred_del
            if total_deleted:
                parts = []
                if new_and_terminal:
                    parts.append(f"new+terminal={new_and_terminal}")
                if deferred_del:
                    parts.append(f"deferred={deferred_del}")
                logger.info("purge_items: удалено %s", " + ".join(parts))
        except Exception as e:
            logger.warning("purge_items: %s", e)

        # prune feedback_targets отдельно (свой retention)
        if feedback_retention_days is not None:
            self.prune_feedback_targets(feedback_retention_days)

        return total_deleted

    def purge_seen_titles(self) -> int:
        """Удаляет просроченные записи из seen_titles."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "DELETE FROM seen_titles WHERE expires_at < ?",
                    (now_iso,),
                )
            deleted = cur.rowcount
            if deleted:
                logger.info("purge_seen_titles: удалено %d записей", deleted)
            return deleted
        except Exception as e:
            logger.warning("purge_seen_titles: %s", e)
            return 0

    # ------------------------------------------------------------------
    # feedback_targets (Фаза 2b: кнопки)
    # ------------------------------------------------------------------

    def add_feedback_target(
        self,
        url: str,
        source: Optional[str],
        title: Optional[str],
        summary: Optional[str],
        llm_score: Optional[int],
        kind: str,
    ) -> int:
        """
        Создаёт запись target для будущего фидбэка.
        kind: 'push' или 'digest'.

        Returns:
            id новой записи feedback_targets.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback_targets
                (url, source, title, summary, llm_score, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, source, title, summary, llm_score, kind, now_iso),
            )
            return cur.lastrowid  # type: ignore

    def add_feedback_target_from_item(self, item, kind: str) -> int:
        """
        Ярлык над add_feedback_target через NewsItem.
        item: NewsItem
        kind: 'push', 'fallback' или 'digest'

        F1: предпочитайте get_or_create_feedback_target_from_item() —
        он идемпотентен и не создаёт дубли при retry.
        """
        return self.add_feedback_target(
            url=item.url,
            source=item.source,
            title=item.title,
            summary=item.summary,
            llm_score=item.llm_score,
            kind=kind,
        )

    def get_or_create_feedback_target_from_item(self, item, kind: str) -> int:
        """
        F1: идемпотентное создание feedback_target.

        Сначала ищет существующий target по (url, kind).
        Если найден — возвращает его id (retry-safe, не создаёт дублей).
        Если не найден — создаёт новый и возвращает id.

        Для retry/recovery-элементов: повторный проход вернёт тот же id,
        что был создан при первой попытке доставки.

        Args:
            item: NewsItem
            kind: 'push', 'fallback' или 'digest'

        Returns:
            id существующего или нового feedback_targets.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
            (item.url, kind),
        ).fetchone()
        if row is not None:
            return row[0]
        return self.add_feedback_target_from_item(item, kind)

    def get_or_create_feedback_target(
        self,
        url: str,
        source: Optional[str],
        title: Optional[str],
        summary: Optional[str],
        llm_score: Optional[int],
        kind: str,
    ) -> int:
        """
        F1: идемпотентное создание feedback_target (plain-вариант без NewsItem).
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
            (url, kind),
        ).fetchone()
        if row is not None:
            return row[0]
        return self.add_feedback_target(
            url=url,
            source=source,
            title=title,
            summary=summary,
            llm_score=llm_score,
            kind=kind,
        )

    def set_feedback_target_message(
        self,
        target_id: int,
        chat_id: str,
        message_id: int,
    ) -> None:
        """
        Записывает chat_id + message_id после отправки сообщения в Telegram.
        Нужно для обновления фидбэка через callback (Фаза 2b).
        """
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE feedback_targets
                SET chat_id = ?, message_id = ?
                WHERE id = ?
                """,
                (chat_id, message_id, target_id),
            )

    def get_feedback_target(self, target_id: int) -> Optional[dict]:
        """
        Возвращает dict с данными target или None.
        Нужен поллеру (Фаза 2b) для связки callback → фидбэк.
        """
        conn = self._connect()
        row = conn.execute(
            """
            SELECT id, url, source, title, summary, llm_score, kind,
                   chat_id, message_id, created_at
            FROM feedback_targets
            WHERE id = ?
            """,
            (target_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def record_feedback(
        self,
        target_id: int,
        verdict: str,
        chat_id: Optional[str] = None,
        message_id: Optional[int] = None,
    ) -> bool:
        """
        Фиксирует голос пользователя (like / dislike) по одному target.

        Поведение:
          — Первый голос по target: INSERT, created_at = now, updated_at = NULL.
          — Re-vote: UPDATE verdict + updated_at = now. created_at неизменяем.
          — Unknown target_id (поздний клик после prune) → возврат False, логируется.
          — Невалидный verdict → ValueError (программная ошибка, не пользователь).

        N3/NIT: механизм INSERT OR IGNORE + UPDATE — атомарный upsert без TOCTOU.
        Первый голос: INSERT, updated_at=NULL. Re-vote: UPDATE verdict+updated_at.
        Параллельные первые голоса: OR IGNORE пропускает конфликт без ошибки;
        IntegrityError невозможен — целостность БД сохранена, один голос побеждает.

        Returns:
            True — записано/обновлено. False — target не найден.
        Raises:
            ValueError — невалидный verdict (программная ошибка).
        """
        if verdict not in ("like", "dislike"):
            raise ValueError(
                f"verdict must be 'like' or 'dislike', got: {verdict!r}"
            )
        now_iso = datetime.now(timezone.utc).isoformat()

        target = self.get_feedback_target(target_id)
        if target is None:
            logger.warning(
                "record_feedback: feedback_target id=%d не найден (возможно, "
                "уже удалён prune). Пропускаем.",
                target_id,
            )
            return False

        with self._tx() as conn:
            # N3/NIT: INSERT OR IGNORE + UPDATE — атомарный upsert без TOCTOU.
            # Первый голос: INSERT, rowcount=1 → updated_at ставится в NULL
            #   (updated_at = None для first vote, чтобы сортировка по
            #   COALESCE(updated_at, created_at) давала created_at).
            # Re-vote: INSERT пропускается (UNIQUE конфликт), rowcount=0 →
            #   UPDATE ставит verdict + updated_at = now.
            # Параллельный первый голос: OR IGNORE пропускает конфликт без
            #   ошибки, rowcount=0 → UPDATE ставит updated_at = now
            #   (допустимо — одновременные голоса, данные целы, IntegrityError
            #   невозможен благодаря OR IGNORE).
            # created_at никогда не перезаписывается (только в INSERT).
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO feedback
                (target_id, url, source, title, summary, llm_score,
                 verdict, chat_id, message_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    target["url"],
                    target["source"],
                    target["title"],
                    target["summary"],
                    target["llm_score"],
                    verdict,
                    chat_id or target["chat_id"],
                    message_id if message_id is not None else target["message_id"],
                    now_iso,
                ),
            )
            inserted = cur.rowcount == 1
            conn.execute(
                """
                UPDATE feedback
                SET verdict = ?,
                    updated_at = CASE WHEN ? THEN updated_at ELSE ? END,
                    chat_id = ?, message_id = ?
                WHERE target_id = ?
                """,
                (
                    verdict,
                    inserted,     # [2]: if first vote → keep existing updated_at (NULL)
                    now_iso,      # [3]: if re-vote → set updated_at = now
                    chat_id or target["chat_id"],
                    message_id if message_id is not None else target["message_id"],
                    target_id,
                ),
            )
        return True

    def get_recent_feedback(
        self,
        verdict: Optional[str] = None,
        limit: int = 10,
    ) -> List[dict]:
        """
        Возвращает последние записи feedback для few-shot в промпт (Фаза 2c).
        Сортировка: updated_at DESC (показывает актуальные re-votes первыми),
        с fallback на created_at DESC для записей без updated_at.

        F5: chat_id/message_id НЕ возвращаются — для few-shot не нужны
        и не должны попасть в промпт.

        verdict фильтрует по like/dislike (опционально).
        limit ограничивает результат.
        """
        conn = self._connect()
        sql = """
            SELECT target_id, url, source, title, summary, llm_score,
                   verdict, created_at, updated_at
            FROM feedback
        """
        params: list = []
        if verdict:
            sql += " WHERE verdict = ?"
            params.append(verdict)
        sql += (
            " ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC "
            "LIMIT ?"
        )
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def prune_feedback_targets(self, older_than_days: int) -> int:
        """
        Удаляет устаревшие feedback_targets (кнопки больше не нужны).
        feedback_targets.created_at < now - older_than_days → удаляются.

        F3 (документировано): таблица feedback НЕ чистится.
        Причина: один пользователь, мало данных, это обучающие данные для
        few-shot (Фаза 2c). Практически permanent storage.

        Returns:
            Количество удалённых feedback_targets. feedback не затрагивается.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=older_than_days)
        ).isoformat()
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "DELETE FROM feedback_targets WHERE created_at < ?",
                    (cutoff,),
                )
            deleted = cur.rowcount
            if deleted:
                logger.info(
                    "prune_feedback_targets: удалено %d targets (старше %d дней)",
                    deleted, older_than_days,
                )
            return deleted
        except Exception as e:
            logger.warning("prune_feedback_targets: %s", e)
            return 0

    # ─────────────────────────────────────────────────────────────────────────────
    # Кросс-источниковая дедупликация
    # ─────────────────────────────────────────────────────────────────────────────

    def get_recent_normalized_titles(
        self,
        window_days: int,
        limit: int = 500,
        exclude_source: Optional[str] = None,
    ) -> List[tuple]:
        """
        Возвращает список недавних нормализованных заголовков из seen_titles
        для fuzzy-сравнения.

        Returns:
            List[(title_hash, title_normalized, source, fetched_at)].
            title_normalized может быть NULL для записей из старых БД (v<9).

        Args:
            window_days: окно в днях
            limit: лимит возвращаемых записей
            exclude_source: исключить source при поиске кросс-источниковых дублей
        """
        conn = self._connect()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()

        params: list = [cutoff]
        source_filter = ""
        if exclude_source:
            source_filter = " AND source != ?"
            params.append(exclude_source)

        # title_normalized может отсутствовать (старые записи до v9)
        rows = conn.execute(
            f"""
            SELECT title_hash,
                   COALESCE(title_normalized, title_hash) AS norm,
                   source,
                   fetched_at
            FROM seen_titles
            WHERE fetched_at > ?{source_filter}
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def get_recent_titles(
        self,
        window_days: int,
        limit: int = 20,
        exclude_source: Optional[str] = None,
    ) -> List[tuple]:
        """
        Возвращает недавние новости из БД с читаемыми заголовками
        (для LLM-дедупликации, кросс-цикловый дедуп).

        M1-фикс: возвращает ТОЛЬКО новости, которые УЖЕ УШЛИ — сохранены,
        отправлены или отложены в ПРОШЛЫХ циклах. Состояния
        ('delivered', 'deferred', 'digested') — это прошедшие обработку,
        они не потеряются при пометке их дубликатами.

        НЕ возвращаются: 'new' (кандидаты текущего цикла, ещё не обработаны),
        'pending' (в процессе обработки), 'skipped'/'failed' (отклонены ранее).

        Это гарантирует: кандидат текущего цикла (state='new') не может быть
        помечен дубликатом самого себя или другого кандидата из того же цикла.
        При этом cross-cycle дедуп сохраняется полностью.

        Returns:
            List[(title, source)].

        Args:
            window_days: окно в днях
            limit: лимит возвращаемых записей (default 20)
            exclude_source: исключить конкретный source
        """
        conn = self._connect()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()

        params: list = [cutoff]
        source_filter = ""
        if exclude_source:
            source_filter = " AND source != ?"
            params.append(exclude_source)

        rows = conn.execute(
            f"""
            SELECT title, source
            FROM items
            WHERE state_at > ?{source_filter}
              AND state IN ('delivered', 'deferred', 'digested')
            ORDER BY state_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def save_skipped(self, item: NewsItem) -> bool:
        """
        Сохраняет item в items со state='skipped' (кросс-источниковый дубль).
        Это нужно чтобы:
          (a) URL-дедуп ловил его в следующих циклах
          (б) он не попал в пуш/дайджест

        Returns True если ВСТАВЛЕН (URL ещё не было), False если IGNORE.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        MAX_SUMMARY_CHARS = 6000
        summary = item.summary[:MAX_SUMMARY_CHARS]

        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, title, summary, published, fetched_at,
                     state, state_at, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, 0)
                    """,
                    (
                        item.source,
                        item.url,
                        item.title,
                        summary,
                        item.published,
                        now_iso,
                        now_iso,
                    ),
                )
                return cur.rowcount > 0
        except Exception as e:
            logger.warning("Ошибка save_skipped(%s): %s", item.title[:60], e)
            return False

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None