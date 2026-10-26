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
  (a) точный URL — ВСЕ записи (не зависит от state; навсегда, до purge).
      URL уникальность защищена UNIQUE INDEX.
  (b) нормализованный заголовок + source — TTL 30 дней (см. _TITLE_DEDUP_TTL_DAYS).
      seen_titles периодически purge'ится по expires_at.

  NOTE: штатный дедуп — навсегда (URL хранится, пока существует запись).
  Автоочистка: purge_* процедуры удаляют записи старше retention (default 1 день).
  Для больших таблиц (сотни тысяч строк) рекомендуется периодический
  PRAGMA incremental_vacuum для высвобождения pages — штатный VACUUM заблокирует
  БД на время перезаписи. auto_vacuum=INCREMENTAL выставляется при создании БД.
  На существующих БД с auto_vacuum=FULL рекомендуется ручной переход:
  PRAGMA wal_checkpoint(TRUNCATE); PRAGMA incremental_vacuum — без блокировки записи.

INSERT OR IGNORE + UNIQUE-констрейнты защищают от дубликатов.
"""
import contextlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, List, Optional, Set

from .collector import NewsItem
from .urlnorm import normalize_url

logger = logging.getLogger("newsbot.storage")

_TITLE_DEDUP_TTL_DAYS = 30
# N4: минимальная длина нормализованного заголовка для title-dedup.
# Пустые/короткие (<3 симв.) не индексируются в seen_titles
# и не проверяются по title-dedup во избежание ложного схлопывания.
MIN_TITLE_LEN = 3

# Максимальная длина title, записываемого в БД (защита от враждебного RSS).
# Раньше капился только summary — заголовок в сотни КБ раздувал строку БД,
# UNIQUE-индекс по url/title и шёл в LLM (оплата токенов).
# Дублируется в collector.MAX_TITLE_CHARS — держать значения синхронными.
MAX_TITLE_CHARS = 500


def _coerce_days(value, default: int, param_name: str = "retention_days") -> int:
    """
    Приводит retention-параметр (может прийти строкой из config.yaml) к int.

    config.yaml читается как YAML/строка → retention_days: "90" давал
    TypeError в timedelta() ещё ДО try-purge, из-за чего очистка не
    выполнялась ни разу. Невалидное значение → осмысленный fallback (default).

    Отрицательное значение (опечатка вида `retention_days: -1`) → ОТКАЗ
    от применения + default. Раньше здесь стоял max(days, 0): -1 превращался
    в 0, cutoff = now, и первый же purge сносил ВСЕ терминальные items
    (delivered/skipped/failed/digested/deferred) — потеря данных из-за одной
    опечатки в конфиге. Теперь такой конфиг логирует warning и работает как
    default.

    Ноль остаётся допустимым ЯВНЫМ значением «удалять сразу»: это осознанный
    путь, используемый тестами и (теоретически) оператором для ручной чистки.
    В prod-конфиге 0 попадает только при явном `retention_days: 0` — purge_items
    логирует об этом warning, чтобы такой конфиг был заметен в логах.
    """
    if value is None:
        return int(default)
    try:
        days = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "%s: невалидное значение %r — используем %d",
            param_name, value, int(default),
        )
        return int(default)
    if days < 0:
        logger.warning(
            "%s: отрицательное значение %r — используем %d "
            "(иначе cutoff=now и purge снёс бы все данные)",
            param_name, value, int(default),
        )
        return int(default)
    return days


class Storage:
    """
    SQLite-файл с двумя таблицами: items (state-машина), seen_titles.
    Схема мигрирует автоматически.
    Идемпотентность: user_version выставляется ПОСЛЕ успешной миграции.
    """

    SCHEMA_VERSION = 15
    STATES = ("new", "pending", "delivered", "skipped", "deferred", "failed", "digested")
    DEFAULT_RETENTION_DAYS = 1
    DEFAULT_RETRY_THRESHOLD_SECONDS = 60
    # Лимит попыток для retry: terminal state при исчерпании.
    DEFAULT_MAX_RETRIES = 8
    # Порог для warning по deferred-items.
    DEFERRED_WARN_THRESHOLD = 500
    # Окно хранения deferred-items: старые неотправленные новости удаляются.
    # Minor 1-фикс: 14→3 дня — deferred живут между дайджестами (09:00/21:00),
    # digest.max_age_hours (24 ч) ограничивает выборку; items старше 3 дней
    # — либо уже отправлены (digested/delivered), либо потеряли актуальность.
    # 3 суток = запас на 1 пропущенное окно + 12 ч grace на retry/Telegram.
    DEFAULT_DEFERRED_RETENTION_DAYS = 3
    # 'new' не прошли скоринг — держим их дольше, чтобы пауза/долгий цикл
    # не уничтожил ещё не обработанные новости. Терминальные состояния
    # чистятся по retention_days (сутки).
    DEFAULT_NEW_RETENTION_DAYS = 7
    # Предельный возраст 'pending' (сутки). Pending — «в работе», но при
    # ChatIdNotBoundError/BotBlocked retry_count НЕ растёт (P1-фикс не
    # инкрементирует его для непоправимых ошибок), поэтому recover_stuck_pending
    # (фильтр retry_count >= max_retries) такие items НЕ снимет, а purge_items
    # раньше не трогал pending вообще → бэклог рос неограниченно, и после
    # починки бота get_stuck_pending (без LIMIT) выливал его флудом.
    # 7 суток: заведомо больше любого штатного цикла (порог retry — 60 с,
    # окно 'new' — те же 7 дней), т.е. всё, что висит в pending неделю,
    # уже не «в работе», а мёртвый груз. Терминализация в 'failed' (НЕ
    # удаление): failed живёт по обычному retention_days, т.е. данные
    # остаются доступны для разбора, а из retry-выборки item уходит.
    PENDING_STALE_DAYS = 7
    # Minor (волна 4): размер батча pending-терминализации в purge_items.
    # Раньше один UPDATE на весь бэклог шёл в ОДНОЙ write-транзакции вместе с
    # тремя DELETE: при десятках тысяч застрявших pending (недели блокировки
    # бота) транзакция держала WAL write-lock секундами/минутами, и
    # параллельные writers (poller) отваливались на «database is locked».
    # Теперь pending переводится в 'failed' порциями по N строк, каждая
    # порция — короткая отдельная транзакция; DELETE-фаза остаётся отдельной
    # транзакцией (она отсекается индексом state+state_at и обычно мала).
    PURGE_PENDING_BATCH_SIZE = 5000

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
                # Minor (волна 4): ROLLBACK тоже упал — соединение отравлено
                # (disk I/O error, сломанный файл, закрытый fd). Оставлять
                # self._conn открытым нельзя: незавершённая транзакция держит
                # write-lock, и все последующие BEGIN падают с «cannot start a
                # transaction within a transaction» до рестарта процесса.
                # Сбрасываем ссылку — _connect() лениво пересоздаст соединение.
                try:
                    conn.close()
                except Exception:
                    pass
                self._conn = None
            raise

    def _current_version(self) -> int:
        """
        Читает PRAGMA user_version.

        Storage Minor 2: fail-closed. Раньше любая ошибка → return 0, что
        обезоруживало downgrade-guard (0 < SCHEMA_VERSION → «старая БД»),
        и более новая БД могла быть мигрирована старым кодом. Теперь
        пробрасываем RuntimeError с внятным сообщением — Storage.__init__
        падает на старте, демон перезапускается (лучше явный отказ,
        чем тихая порча схемы).
        """
        conn = self._connect()
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
        except Exception as e:
            raise RuntimeError(
                f"Не удалось прочитать PRAGMA user_version для БД "
                f"{self.db_path}: {e}. Проверьте целостность файла БД и права "
                f"доступа — миграция без версии небезопасна."
            ) from e
        return int(row[0]) if row else 0

    def _init_db(self) -> None:
        """
        Инициализация/миграция схемы.

        Storage Minor 3: при исключении транзакция откатывается (ROLLBACK
        в try — она могла быть уже откачена/не начата) и соединение
        закрывается, иначе остаётся открытая транзакция и write-lock на файле.
        """
        conn = self._connect()
        try:
            self._init_db_body(conn)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass  # транзакция могла не начаться или уже откачена
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None
            raise

    def _init_db_body(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        current_version = self._current_version()

        # ── 0. Downgrade guard ──────────────────────────────────────────────
        if current_version > self.SCHEMA_VERSION:
            raise RuntimeError(
                f"БД создана более новой версией newsbot (v{current_version}), "
                f"запущена v{self.SCHEMA_VERSION} — откат невозможен. "
                f"Обновите newsbot до v{current_version} или новее."
            )

        # ── 1. Миграция ──
        self._migrate(conn, current_version)

        # ── 2. Таблицы (CREATE IF NOT EXISTS — идемпотентны) ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT    NOT NULL,
                url         TEXT    NOT NULL,
                url_norm    TEXT,
                title       TEXT    NOT NULL,
                summary     TEXT    NOT NULL,
                published   TEXT,
                tags        TEXT,
                cluster_key TEXT,
                lang        TEXT,
                fetched_at  TEXT    NOT NULL,
                state       TEXT    NOT NULL DEFAULT 'new',
                state_at    TEXT    NOT NULL,
                llm_score   INTEGER,
                llm_summary TEXT,
                llm_reason  TEXT,
                llm_urgent  INTEGER,
                llm_title   TEXT,
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

        # ── 5c. v13: defensive post-check для lang ──────────────────
        # Аналогично v10: при user_version=13 без колонки (конкурентная
        # миграция) — добавляем идемпотентно.
        it_cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        if "lang" not in it_cols:
            try:
                conn.execute("ALTER TABLE items ADD COLUMN lang TEXT")
                logger.info("v13 post-check: добавлена колонка items.lang")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                logger.info("v13 post-check: колонка lang уже существует")

        # ── 5d. v15: defensive post-check для llm_title ─────────────
        # Аналогично v10/v13: при user_version=15 без колонки
        # (конкурентная миграция) — добавляем идемпотентно.
        it_cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        if "llm_title" not in it_cols:
            try:
                conn.execute("ALTER TABLE items ADD COLUMN llm_title TEXT")
                logger.info("v15 post-check: добавлена колонка items.llm_title")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
                logger.info("v15 post-check: колонка llm_title уже существует")

        logger.debug("БД инициализирована (v%d): %s", self.SCHEMA_VERSION, self.db_path)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Создаёт индексы. IntegrityError = дубли в old data, логируем явно."""
        for stmt in (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url_norm ON items(url_norm)",
            # M4: не-unique индекс по url заменяет дропнутый idx_items_url.
            # UNIQUE-констрейнт теперь на url_norm (v14). Индекс по url
            # ускоряет set_state, increment_retry_count, update_llm_results_batch,
            # get_urls_with_retry_gte, mark_failed_if_retries_exhausted
            # (WHERE url IN (...)), которые иначе деградируют до полного скана.
            "CREATE INDEX IF NOT EXISTS idx_items_url ON items(url)",
            "CREATE INDEX IF NOT EXISTS idx_items_state ON items(state)",
            "CREATE INDEX IF NOT EXISTS idx_items_state_at ON items(state_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_seen_title_source "
            "ON seen_titles(title_hash, source)",
            "CREATE INDEX IF NOT EXISTS idx_seen_ttl ON seen_titles(expires_at)",
            # M1: ускоряет get_recent_normalized_titles — запрос WHERE fetched_at > ?
            "CREATE INDEX IF NOT EXISTS idx_seen_fetched ON seen_titles(fetched_at)",
            # Minor#7: составной индекс для purge и get_recent_titles
            "CREATE INDEX IF NOT EXISTS idx_items_state_state_at ON items(state, state_at)",
            # Волна 2: составные индексы для горячего пути дедупа и get_recent_titles.
            # (source, fetched_at) — SELECT WHERE source=? AND fetched_at>? из dedup.py.
            # (cluster_key, fetched_at) — SELECT WHERE cluster_key=? для кластеризации.
            "CREATE INDEX IF NOT EXISTS idx_items_source_fetched ON items(source, fetched_at)",
            "CREATE INDEX IF NOT EXISTS idx_items_cluster_key_fetched ON items(cluster_key, fetched_at)",
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
            # F1/Minor#4: UNIQUE-индекс (url, kind) для feedback_targets.
            # Для старых БД: дедуплицируем дубли и создаём UNIQUE INDEX.
            #
            # Storage Minor 1: DROP только если существующий индекс НЕ unique.
            # Раньше индекс безусловно дропался и пересоздавался на каждом
            # старте, а ошибки глотались → при неудачном CREATE уникальность
            # терялась молча (и при следующем старте дубли уже успевали
            # накопиться). Теперь: (а) DROP только для non-unique индекса,
            # (б) IntegrityError при CREATE → warning + одноразовый дедуп
            # (по образцу v11-миграции) + повторный CREATE.
            name = "idx_feedback_targets_url_kind"
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name = ?",
                (name,),
            ).fetchone()
            existing_sql = str(row[0]) if row and row[0] else ""
            if existing_sql and "UNIQUE" not in existing_sql.upper():
                try:
                    conn.execute(f"DROP INDEX IF EXISTS {name}")
                    logger.info(
                        "feedback_targets: non-unique %s удалён, "
                        "пересоздаём как UNIQUE", name,
                    )
                except sqlite3.OperationalError as e:
                    logger.warning(
                        "feedback_targets: не удалось удалить non-unique %s: %s",
                        name, e,
                    )
            self._ensure_unique_feedback_targets_index(conn, name)
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

    def _ensure_unique_feedback_targets_index(
        self, conn: sqlite3.Connection, name: str,
    ) -> None:
        """
        Storage Minor 1: создаёт UNIQUE INDEX (url, kind) на feedback_targets.
        При IntegrityError (дубли в данных — например, БД, где уникальность
        когда-то потерялась) — одноразовый дедуп по образцу v11-миграции
        и повторный CREATE. Ошибку не глотаем молча: пишем warning.
        """
        stmt = f"CREATE UNIQUE INDEX IF NOT EXISTS {name} ON feedback_targets(url, kind)"
        try:
            conn.execute(stmt)
            return
        except sqlite3.IntegrityError as e:
            logger.warning(
                "feedback_targets: UNIQUE %s не создан (дубли url,kind): %s. "
                "Выполняем одноразовый дедуп и повторяем.",
                name, e,
            )
        except sqlite3.OperationalError as e:
            logger.warning(
                "feedback_targets: UNIQUE %s не создан (OperationalError): %s",
                name, e,
            )
            return

        self._dedupe_feedback_targets(conn)
        try:
            conn.execute(stmt)
            logger.info("feedback_targets: UNIQUE %s создан после дедупа", name)
        except (sqlite3.IntegrityError, sqlite3.OperationalError) as e:
            logger.warning(
                "feedback_targets: UNIQUE %s так и не создан: %s. "
                "Дедуп не помог — проверьте данные вручную.",
                name, e,
            )

    def _dedupe_feedback_targets(self, conn: sqlite3.Connection) -> None:
        """
        Удаляет дубли feedback_targets(url, kind), оставляя запись
        с НАИМЕНЬШИМ id (первую по времени) — тот же паттерн, что в
        v11-миграции. Нужен перед созданием UNIQUE-индекса.

        Storage Minor: перед удалением «младших» target-строк перенаправляем
        привязанные к ним feedback.target_id на выживший MIN(id). FK между
        таблицами не включён, поэтому без перенаправления feedback становился
        бы осиротевшим: get_feedback_target() вернул бы None и поздний клик
        пользователя терялся бы. У feedback действует UNIQUE(target_id),
        поэтому при конфликте оставляем более свежий голос
        (COALESCE(updated_at, created_at)). Уже осиротевшие feedback
        (target удалён ранее, например prune'ом) не трогаем, но фиксируем
        warning с их числом — молча терять данные нельзя.
        """
        try:
            tables = {str(r[0]) for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            has_feedback = "feedback" in tables

            redirect_count = 0
            removed_feedback = 0
            if has_feedback:
                rows = conn.execute("""
                    SELECT f.id AS feedback_id,
                           COALESCE(f.updated_at, f.created_at) AS f_ts,
                           keeper.keeper_id AS keeper_id
                    FROM feedback f
                    JOIN feedback_targets t ON t.id = f.target_id
                    JOIN (
                        SELECT url, kind, MIN(id) AS keeper_id
                        FROM feedback_targets
                        GROUP BY url, kind
                    ) keeper ON keeper.url = t.url AND keeper.kind = t.kind
                    WHERE f.target_id <> keeper.keeper_id
                """).fetchall()
                for row in rows:
                    feedback_id = row["feedback_id"]
                    keeper_id = row["keeper_id"]
                    existing = conn.execute(
                        "SELECT id, COALESCE(updated_at, created_at) AS ts "
                        "FROM feedback WHERE target_id = ? AND id <> ?",
                        (keeper_id, feedback_id),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            "UPDATE feedback SET target_id = ? WHERE id = ?",
                            (keeper_id, feedback_id),
                        )
                        redirect_count += 1
                    elif (row["f_ts"] or "") > (existing["ts"] or ""):
                        # более свежий голос переезжает на выживший target
                        conn.execute(
                            "DELETE FROM feedback WHERE id = ?", (existing["id"],)
                        )
                        conn.execute(
                            "UPDATE feedback SET target_id = ? WHERE id = ?",
                            (keeper_id, feedback_id),
                        )
                        redirect_count += 1
                        removed_feedback += 1
                    else:
                        conn.execute(
                            "DELETE FROM feedback WHERE id = ?", (feedback_id,)
                        )
                        removed_feedback += 1

            cur = conn.execute("""
                DELETE FROM feedback_targets
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM feedback_targets
                    GROUP BY url, kind
                )
            """)
            deleted = cur.rowcount
            if deleted:
                logger.info(
                    "feedback_targets: дедуплицировано %d дублей (url, kind)",
                    deleted,
                )
            if redirect_count or removed_feedback:
                logger.info(
                    "feedback: перенаправлено %d голосов на выживший target, "
                    "удалено %d дублей голосов",
                    redirect_count, removed_feedback,
                )
            if has_feedback:
                orphans = conn.execute("""
                    SELECT COUNT(*) FROM feedback
                    WHERE target_id NOT IN (SELECT id FROM feedback_targets)
                """).fetchone()[0]
                if orphans:
                    logger.warning(
                        "feedback: %d записей ссылаются на несуществующий "
                        "target_id (осиротели до дедупа или после prune) — "
                        "get_feedback_target() вернёт для них None, поздний "
                        "клик не запишется.",
                        orphans,
                    )
        except sqlite3.OperationalError as e:
            logger.warning("feedback_targets: дедуп пропущен: %s", e)

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

            # (г) v4: llm_score, llm_summary, llm_reason, filtered_at, tags, cluster_key
            for col_name, col_def in (
                ("llm_score", "INTEGER"),
                ("llm_summary", "TEXT"),
                ("llm_reason", "TEXT"),
                ("filtered_at", "TEXT"),
                ("tags", "TEXT"),
                ("cluster_key", "TEXT"),
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

        # (е) v6: retry_count, cluster_key — только для существующих таблиц с версией < 6
        # cols читается выше, гарантированно определён.
        if from_version < 6 and "items" in existing_tables:
            for col_name, col_def in (
                ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
                ("cluster_key", "TEXT"),
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

        # (й) v11: индекс seen_titles(fetched_at) для get_recent_normalized_titles
        # и составной индекс (state, state_at) для purge и get_recent_titles.
        # Для новых БД индексы создаются через _create_indexes (идемпотентно).
        # Для старых — поднимаем user_version и применяем идемпотентно.
        # IF NOT EXISTS = безопасно при повторном запуске.
        if "seen_titles" in {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}:
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_seen_fetched "
                    "ON seen_titles(fetched_at)"
                )
                logger.info("Миграция v11: создан индекс idx_seen_fetched")
            except sqlite3.OperationalError:
                pass  # уже существует или конкурентный процесс
        # Составной индекс на items: безопасен даже без таблицы
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_state_state_at "
                "ON items(state, state_at)"
            )
            logger.info("Миграция v11: создан индекс idx_items_state_state_at")
        except sqlite3.OperationalError:
            pass

        # (й) v11: UNIQUE(url, kind) на feedback_targets (Minor#4).
        # Для старых БД: дедуплицируем дубли (удаляем младшие id) перед
        # созданием уникального индекса. SQLite не поддерживает ALTER TABLE
        # ADD CONSTRAINT → создаём UNIQUE INDEX. Для свежей БД — в _create_indexes.
        if "feedback_targets" in {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}:
            # Дедуп: удаляем строки с дублирующимися (url, kind),
            # оставляя запись с НАИМЕНЬШИМ id (первая по времени).
            # Storage Minor: общий helper сначала перенаправляет feedback
            # дублей на выживший target, чтобы не плодить осиротевшие голоса.
            self._dedupe_feedback_targets(conn)
            # Создаём UNIQUE INDEX. DROP INDEX IF EXISTS для безопасности
            # (в старых БД может быть не-unique idx_feedback_targets_url_kind).
            try:
                conn.execute(
                    "DROP INDEX IF EXISTS idx_feedback_targets_url_kind"
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_targets_url_kind "
                    "ON feedback_targets(url, kind)"
                )
                logger.info("Миграция v11: создан UNIQUE idx_feedback_targets_url_kind")
            except sqlite3.OperationalError:
                pass  # конкурентный процесс уже создал

        # (й) v12: cluster_key TEXT для кластеризации новостей.
        # Идемпотентно: try/except для конкурентной миграции.
        if "items" in existing_tables:
            current_items_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            if "cluster_key" not in current_items_cols:
                try:
                    conn.execute(
                        "ALTER TABLE items ADD COLUMN cluster_key TEXT"
                    )
                    logger.info("Миграция v12: добавлена колонка items.cluster_key")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v12: колонка cluster_key уже существует (конкурентный процесс)")

        # (й) v13: lang TEXT для языка фида (проводка config → БД → prompt).
        # Идемпотентно: try/except для конкурентной миграции.
        # Для свежей БД колонка добавляется через post-check (как llm_urgent v10).
        if "items" in existing_tables:
            current_items_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            if "lang" not in current_items_cols:
                try:
                    conn.execute("ALTER TABLE items ADD COLUMN lang TEXT")
                    logger.info("Миграция v13: добавлена колонка items.lang")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v13: колонка lang уже существует (конкурентный процесс)")

        # (к) v14: url_norm TEXT — нормализованный URL для URL-based дедупа.
        # UNIQUE INDEX по url_norm склеивает utm/без utm в одну запись.
        # Для свежих БД колонка + индекс создаются в _create_indexes (в новой схеме).
        #
        # B1-фикс (crash-loop): если миграция падает с IntegrityError (конфликт
        # дублирующихся url_norm), Вариант A восстановления работает благодаря
        # best-effort backfill: при старте бота после ручного Варианта A
        # url_norm IS NULL у конфликтных строк; backfill пытается заполнить их
        # заново, но при конфликте (другое значение url_norm уже существует)
        # оставляет IS NULL — UNIQUE-индекс создаётся, бот стартует.
        # Конфликтные строки с url_norm IS NULL не участвуют в URL-дедупе.
        if "items" in existing_tables:
            current_items_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            if "url_norm" not in current_items_cols:
                try:
                    conn.execute("ALTER TABLE items ADD COLUMN url_norm TEXT")
                    logger.info("Миграция v14: добавлена колонка items.url_norm")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v14: колонка url_norm уже существует (конкурентный процесс)")

            # Backfill: заполняем url_norm для существующих записей.
            # B1-фикс: best-effort INSERT semantics — если нормализованное
            # значение уже занято другой строкой, НЕ перезаписываем,
            # оставляем url_norm IS NULL (SQLite UNIQUE-индекс с NULL не
            # конфликтует). Это размыкает crash-loop: после ручного Варианта A
            # рестарт → backfill пропускает конфликтные строки → UNIQUE-индекс
            # создаётся → бот стартует.
            #
            # B1-фикс (Variant A): O(N²) → O(N).
            # Старый код: per-row `SELECT 1 FROM items WHERE url_norm = ?`
            # — full scan на каждую строку, всё в одной транзакции.
            # На 100k+ legacy-строках → десятки минут старта.
            # Исправление: один раз загружаем занятые url_norm в Python-set,
            # затем O(1) lookup per row. При успешном UPDATE добавляем в set,
            # чтобы строки батча «видели» занятые значения от предыдущих строк.
            #
            # Идемпотентен (при повторном запуске — уже заполнено).
            # MAJOR 3-фикс: keyset-пагинация вместо open cursor на UPDATE-таблице.
            # SQLite undefined behaviour при SELECT по открытому курсору той же
            # таблицы с последующим UPDATE (строки могут быть пропущены).
            # При rowid > last_rowid — детерминированный порядок, без пропусков.
            # Пачками по 10000 — безопасно для больших таблиц.
            #
            # Вариант B (предварительный индекс) отклонён: память ~10 МБ
            # на 100k URL приемлема, а индекс до backfill не ускоряет
            # INSERT INTO items (url_norm IS NULL) — индекс по NULL не работает.
            occupied_norm = {
                row[0] for row in conn.execute(
                    "SELECT url_norm FROM items WHERE url_norm IS NOT NULL"
                ).fetchall()
            }
            last_rowid = 0
            while True:
                batch = conn.execute(
                    "SELECT rowid, url FROM items "
                    "WHERE rowid > ? AND url_norm IS NULL "
                    "ORDER BY rowid LIMIT 10000",
                    (last_rowid,),
                ).fetchall()
                if not batch:
                    break
                for rowid, orig_url in batch:
                    url_norm = normalize_url(orig_url)
                    try:
                        if url_norm in occupied_norm:
                            # url_norm уже занят другой строкой — пропускаем.
                            continue
                        conn.execute(
                            "UPDATE items SET url_norm = ? WHERE rowid = ? AND url_norm IS NULL",
                            (url_norm, rowid),
                        )
                        occupied_norm.add(url_norm)
                    except sqlite3.IntegrityError:
                        # Конфликт с concurrently-вставленной строкой (между
                        # occupied_norm load и UPDATE). Пропускаем.
                        pass
                last_rowid = batch[-1][0]

            # MAJOR 3-фикс: создание индекса ДО удаления старого.
            # Если CREATE UNIQUE INDEX падает (дубли по нормализованному URL),
            # старый индекс idx_items_url остаётся нетронутым — защита от
            # «БД без unique-индекса по URL». При IntegrityError:
            # логируем actionable-сообщение (какие именно дубли мешают) и
            # НЕ устанавливаем user_version=14 — миграция повторится после
            # ручной дедупликации (DROP дубликатов вручную или автоматически).
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url_norm "
                    "ON items(url_norm)"
                )
                # Индекс успешно создан — безопасно дропать старый.
                conn.execute("DROP INDEX IF EXISTS idx_items_url")
                logger.info("v14: idx_items_url → idx_items_url_norm")
                # NOTE: user_version=14 выставляется ВЫЗЫВАЮЩИМ кодом
                # после успешного commit. Здесь мы только меняем схему.
            except sqlite3.IntegrityError as e:
                # Приводим url_norm к множеству чтобы найти дубликата.
                dup_rows = conn.execute(
                    "SELECT url_norm, COUNT(*) AS cnt, MIN(rowid) AS keep_rowid "
                    "FROM items WHERE url_norm IS NOT NULL "
                    "GROUP BY url_norm HAVING cnt > 1"
                ).fetchall()
                dup_list = [f"'{r[0]}' ({r[1]} rows, keep rowid={r[2]})"
                            for r in dup_rows]
                logger.error(
                    "v14: CREATE UNIQUE INDEX idx_items_url_norm НЕ создан "
                    "(конфликт данных: нормализованные URL неуникальны). "
                    "Дубликаты по нормализованному URL: %s. "
                    "Старый индекс idx_items_url сохранён. "
                    "Для продолжения — дедуплицируйте вручную: "
                    "Вариант A (безопасный, рекомендуется): обнулите url_norm "
                    "у дублирующихся строк (не уникальных), чтобы UNIQUE-индекс "
                    "мог быть создан: "
                    "UPDATE items SET url_norm = NULL WHERE rowid NOT IN ("
                    "  SELECT MIN(rowid) FROM items GROUP BY url_norm"
                    "  HAVING url_norm IS NOT NULL"
                    "); — после этого миграция применится: backfill при старте "
                    "заполнит url_norm заново, но для конфликтных строк "
                    "url_norm останется NULL (best-effort, не конфликтует с UNIQUE). "
                    "Строки с url_norm IS NULL не участвуют в URL-дедупликации "
                    "(продедуп по ним не работает). "
                    "Вариант B (радикальный DELETE): удалить дубликаты полностью: "
                    "DELETE FROM items WHERE rowid NOT IN ("
                    "  SELECT MIN(rowid) FROM items GROUP BY url_norm"
                    "  HAVING url_norm IS NOT NULL"
                    ") AND url_norm IS NOT NULL; "
                    "Затем повторите миграцию (user_version=13). "
                    "Исключение поднято — user_version НЕ обновлён.",
                    "; ".join(dup_list) if dup_list else str(e),
                )
                # Re-raise: user_version останется 13 → миграция повторится
                # после ручной дедупликации. НЕ удаляем старый индекс
                # (IF EXISTS), чтобы при повторном apply старый idx всё ещё был.
                raise

        # (л) v15: llm_title TEXT — EN-перевод заголовка от LLM.
        # Идемпотентно: try/except для конкурентной миграции.
        if "items" in existing_tables:
            current_items_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            if "llm_title" not in current_items_cols:
                try:
                    conn.execute("ALTER TABLE items ADD COLUMN llm_title TEXT")
                    logger.info("Миграция v15: добавлена колонка items.llm_title")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                    logger.info("v15: колонка llm_title уже существует (конкурентный процесс)")

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

        # (a) URL dedup по нормализованному ключу: склеивает utm/без utm,
        # порядок параметров, fragment и т.п. в одну запись в БД.
        url_norm = normalize_url(item.url)
        row = conn.execute(
            "SELECT 1 FROM items WHERE url_norm = ? LIMIT 1",
            (url_norm,),
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
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        now_iso = now_utc.isoformat()
        expires_iso = (now_utc + timedelta(days=_TITLE_DEDUP_TTL_DAYS)).isoformat()
        normalized = item.title_normalized()

        # Cap summary — защита от чрезмерных LLM-токенов и БД-размера.
        MAX_SUMMARY_CHARS = 6000
        summary = item.summary[:MAX_SUMMARY_CHARS]
        # Cap title — враждебный RSS с заголовком в сотни КБ раздувал строку
        # БД/UNIQUE-индекс и уходил в LLM (оплата токенов).
        title = item.title[:MAX_TITLE_CHARS] if item.title else item.title

        # Нормализованный URL (ключ дедупа — см. is_duplicate и v14 миграцию).
        url_norm = normalize_url(item.url)

        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, url_norm, title, summary, published, tags, cluster_key,
                     lang, fetched_at, state, state_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)
                    """,
                    (
                        item.source,
                        item.url,
                        url_norm,
                        title,
                        summary,
                        item.published,
                        getattr(item, "tags", None),
                        getattr(item, "cluster_key", None),
                        getattr(item, "lang", None),
                        now_iso,
                        now_iso,
                    ),
                )
                inserted = cur.rowcount

                # seen_titles индексируется только для осмысленных заголовков
                # (normalized >= MIN_TITLE_LEN) — чтобы пустые title не ложно схлопывались.
                # Sliding TTL: при повторной встрече title_hash обновляем
                # fetched_at/expires_at, чтобы активные темы не выпадали из дедупа
                # при длинных окнах (30 дней). Поведение: upsert, не INSERT OR REPLACE
                # (сохраняет id/title_normalized, не теряет UNIQUE-конфликты для отладки).
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
                        conn.execute(
                            """
                            UPDATE seen_titles
                            SET fetched_at = ?, expires_at = ?
                            WHERE title_hash = ? AND source = ?
                            """,
                            (now_iso, expires_iso, normalized, item.source),
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
                        conn.execute(
                            """
                            UPDATE seen_titles
                            SET fetched_at = ?, expires_at = ?
                            WHERE title_hash = ? AND source = ?
                            """,
                            (now_iso, expires_iso, normalized, item.source),
                        )

            return inserted > 0
        except sqlite3.IntegrityError:
            # UNIQUE constraint / CHECK — штатный дедуп (URL уже есть), не ошибка.
            return False
        except Exception as e:
            # M2: логируем на уровне error, чтобы отличать баг от дедупа.
            # INSERT OR IGNORE при дедупе НЕ выбрасывает исключение — только
            # IntegrityError при реальных constraint-нарушениях, что невозможно
            # при INSERT OR IGNORE с UNIQUE(url). Сюда попадают I/O, диск full,
            #corruption — это баги, а не штатное поведение.
            logger.exception("Ошибка save(%s): %s", item.title[:60], e)
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
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        now_iso = now_utc.isoformat()
        expires_iso = (now_utc + timedelta(days=_TITLE_DEDUP_TTL_DAYS)).isoformat()

        inserted_items: List[NewsItem] = []
        seen_keys: Set[str] = set()  # для дедупа по title+source внутри батча

        try:
            with self._tx() as conn:
                for item in items:
                    summary = item.summary[:MAX_SUMMARY_CHARS]
                    title = item.title[:MAX_TITLE_CHARS] if item.title else item.title
                    normalized = item.title_normalized()
                    url_norm = normalize_url(item.url)

                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO items
                        (source, url, url_norm, title, summary, published, tags, cluster_key,
                         lang, fetched_at, state, state_at, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, 0)
                        """,
                        (
                            item.source,
                            item.url,
                            url_norm,
                            title,
                            summary,
                            item.published,
                            getattr(item, "tags", None),
                            getattr(item, "cluster_key", None),
                            getattr(item, "lang", None),
                            now_iso,          # fetched_at
                            now_iso,          # state_at
                        ),
                    )
                    if cur.rowcount:
                        inserted_items.append(item)

                        # seen_titles — только для осмысленных заголовков
                        # Sliding TTL: обновляем fetched_at/expires_at при повторном
                        # появлении, чтобы активные темы не выпадали из дедупа.
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
                                    conn.execute(
                                        """
                                        UPDATE seen_titles
                                        SET fetched_at = ?, expires_at = ?
                                        WHERE title_hash = ? AND source = ?
                                        """,
                                        (now_iso, expires_iso, normalized, item.source),
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
                                    conn.execute(
                                        """
                                        UPDATE seen_titles
                                        SET fetched_at = ?, expires_at = ?
                                        WHERE title_hash = ? AND source = ?
                                        """,
                                        (now_iso, expires_iso, normalized, item.source),
                                    )
                                seen_keys.add(key)

            return inserted_items
        except sqlite3.IntegrityError:
            # Штатный дедуп: конфликт UNIQUE — не ошибка, просто пропускаем батч.
            # M3: IntegrityError внутри транзакции → ROLLBACK, возвращать
            # inserted_items нельзя (pipeline считает их сохранёнными → дубль).
            logger.warning(
                "save_many(%d items): IntegrityError → ROLLBACK, "
                "%d вставленных потеряно", len(items), len(inserted_items)
            )
            return []
        except Exception as e:
            # M2: логируем на уровне error. Сюда попадают I/O, corruption, etc.
            logger.exception("Ошибка save_many(%d items): %s", len(items), e)
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
        limit: int = None,
    ) -> List[NewsItem]:
        """
        Возвращает items state='pending' старше max_age_seconds
        с retry_count < max_retries (не терминальные).

        M3: порог 60 с защищает «в работе» — pending младше порога
        пропускается и обрабатывается в штатном цикле pipeline.
        На следующем цикле (через >=60 с) они попадут сюда, если still pending.

        Элементы с retry_count >= max_retries попадают ВРУЧНУЮ в recover_stuck_pending,
        который снимает их сразу (без ожидания 60 с) — recover вызывается
        ДО get_stuck_pending на старте цикла.

        Args:
            limit: макс. число возвращаемых items (защита RAM при большом backlog).
                   None — без лимита.
        """
        if max_age_seconds is None:
            max_age_seconds = self.DEFAULT_RETRY_THRESHOLD_SECONDS
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_RETRIES
        conn = self._connect()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        ).isoformat()
        # M6: `<=` вместо `<` при max_age_seconds==0. При max_age_seconds=0
        # cutoff=now, state_at тоже now → `<` отсекает только-что созданные
        # pending-items (race по микросекундам в тестах). `<=` корректен:
        # state_at=now означает «awaiting обработки прямо сейчас» — такие items
        # принадлежат текущему retry-окну и должны быть возвращены.
        state_at_op = "<=" if max_age_seconds == 0 else "<"
        sql = f"""
            SELECT source, url, title, summary, published, tags, cluster_key, lang, fetched_at,
                   llm_score, llm_summary, llm_reason, llm_urgent, llm_title, retry_count
            FROM items
            WHERE state = 'pending'
              AND state_at {state_at_op} ?
              AND retry_count < ?
            ORDER BY id ASC
            """
        params: tuple = (cutoff, max_retries)
        if limit is not None:
            sql += "\n LIMIT ?"
            params = (cutoff, max_retries, limit)
        rows = conn.execute(sql, params).fetchall()
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
        в терминальный 'failed'.

        M3-фикс: при retry_count >= max_retries элемент ТЕРМИНАЛЬНЫЙ — его возраст
        не имеет значения. recover снимает такие элементы НЕЗАВИСИМО от возраста.
        Это предотвращает зависшие «мертвые» pending (например, при падении бота
        сразу после set_state(pending)), которые могли бы висеть часами.

        ⚠️ Возрастной фильтр НЕ применяется: pending с retry_count < max_retries
        НЕ трогаются здесь — их штатно подбирает get_stuck_pending() по порогу
        retry_threshold_sec и они уходят в retry (переводить их в 'failed'
        здесь означало бы ломать retry-механику).

        Args:
            max_retries: лимит попыток.
            retry_threshold_sec: DEPRECATED, не используется (оставлен для
                                 обратной совместимости вызовов из pipeline).
                                 Возрастную выборку делает get_stuck_pending().
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        total = 0
        try:
            with self._tx() as conn:
                # M3 (primary): терминальные (retry_count>=max) — снимаем БЕЗ возрастного
                # фильтра. recover вызывается на старте цикла ДО get_stuck_pending.
                # Это защищает от зависших «мёртвых» pending, которые не успели
                # состариться (младше 60 с), но retry_count уже исчерпан.
                cur = conn.execute(
                    """
                    UPDATE items
                    SET state = 'failed', state_at = ?
                    WHERE state = 'pending'
                      AND retry_count >= ?
                    """,
                    (now_iso, max_retries),
                )
                total += cur.rowcount
            if total:
                logger.warning(
                    "C2-recovery: %d зависших items (pending, retry_count>=%d) "
                    "переведены в 'failed'",
                    total, max_retries,
                )
            return total
        except Exception as e:
            logger.warning("recover_stuck_pending: %s", e)
            return 0

    def get_all_by_state(self, state: str, limit: Optional[int] = None) -> List[NewsItem]:
        conn = self._connect()
        sql = """
            SELECT source, url, title, summary, published, tags, cluster_key, lang, fetched_at, state_at,
                   llm_score, llm_summary, llm_reason, llm_urgent, llm_title
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

    def count_deferred_before(self, cutoff_iso: str) -> int:
        """
        Возвращает количество deferred-items с fetched_at < cutoff_iso.
        Лёгкий COUNT(*) — используется для логирования «дропнутых» при
        закрытии дайджест-окна.

        fetched_at всегда tz-aware ISO; строковое сравнение корректно
        для ISO+00:00.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM items WHERE state='deferred' AND fetched_at < ?",
            (cutoff_iso,),
        ).fetchone()
        return row[0] if row else 0

    def get_deferred_items(
        self, limit: Optional[int] = None,
        since_iso: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        published_since: Optional[str] = None,
    ) -> List[NewsItem]:
        """
        Возвращает items со state='deferred', отсортированные:
          llm_score DESC (лучшие первыми), затем fetched_at ASC.
        Лимит опционален (для дайджеста: top-N по score).

        Args:
            limit: если задано — LIMIT в SQL.
            since_iso: если задано — выборка ограничена fetched_at >= since_iso
                       (ISO-8601, например fromisoformat() или datetime.isoformat()).
            max_age_hours: если задано — fetched_at >= now - max_age_hours.
                           При задании ОБОИХ параметров (since_iso и max_age_hours)
                           берётся MAX(since_iso, now - max_age_hours) —
                           более поздняя граница. None означает «без ограничения».
            published_since: DEPRECATED. Фильтрация по published выполняется на
                              стороне pipeline (Python), т.к. формат published в БД
                              неоднородный (ISO / 'DD.MM.YYYY HH:MM') и SQLite
                              julianday() по нему ненадёжен. Параметр оставлен для
                              обратной совместимости и игнорируется.
        """
        conn = self._connect()
        params: list = []
        sql = """
            SELECT source, url, title, summary, published, tags, cluster_key, lang, fetched_at, state_at,
                   llm_score, llm_summary, llm_reason, llm_urgent, llm_title
            FROM items
            WHERE state = 'deferred'
        """
        effective_since = since_iso
        # Minor 1-фикс: max_age_hours НЕ перекрывает since_iso — берём MAX
        # обоих (более поздняя граница = более свежие items).
        # Например: since_iso=5ч назад, max_age_hours=24 → fetched_at>=5ч назад
        # (окно уже ограничивает сильнее). since_iso=1ч назад, max_age=24ч
        # → fetched_at>=1ч назад (max_age не ужесточает). Оба None → без фильтра.
        if max_age_hours is not None:
            max_age_iso = (
                datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            ).isoformat()
            if effective_since is None:
                effective_since = max_age_iso
            elif max_age_iso > effective_since:
                # max_age_iso более свежая (более поздняя) → ограничивает сильнее
                effective_since = max_age_iso
        if effective_since is not None:
            # fetched_at всегда tz-aware ISO (см. save/_rows_to_items),
            # строковое сравнение корректно (лексикографически для ISO+00:00);
            # julianday() блокировал индекс и был NULL-небезопасен.
            sql += " AND fetched_at >= ?"
            params.append(effective_since)
        # DEPRECATED: published_since фильтрация вынесена в pipeline (Python),
        # т.к. julianday() по published ненадёжен (неоднородный формат в БД).
        sql += " ORDER BY llm_score DESC, fetched_at ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
        return self._rows_to_items(rows)

    def get_skipped_items(
        self,
        since_iso: Optional[str] = None,
        min_score: int = 0,
        limit: int = 50,
    ) -> List[NewsItem]:
        """
        Возвращает skipped-items с llm_score >= min_score, fetched_at >= since_iso.
        Сортировка: llm_score DESC, fetched_at ASC.

        Используется для гарантии кино (добор из skipped с score >= 3).
        Категоризация по source/tags выполняется вызывающей стороной
        (pipeline._infer_category — категория в БД не хранится).

        NOTE: items с llm_score IS NULL не возвращаются (не скорились —
        добор бессмысленен).
        """
        conn = self._connect()
        params: list = []
        sql = """
            SELECT source, url, title, summary, published, tags, cluster_key, lang,
                   fetched_at, state_at, llm_score, llm_summary, llm_reason, llm_urgent, llm_title
            FROM items
            WHERE state = 'skipped'
              AND llm_score IS NOT NULL
              AND llm_score >= ?
        """
        params.append(int(min_score))
        if since_iso is not None:
            sql += " AND fetched_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY llm_score DESC, fetched_at ASC"
        sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, params).fetchall()
        return self._rows_to_items(rows)

    def set_digested(self, urls: List[str]) -> int:
        """
        Переводит URLs в терминальный state='digested'.
        Возвращает количество обновлённых строк.
        """
        return self.set_state(urls, "digested")

    def delete_items_before(
        self, cutoff_iso: str, retention_days: Optional[int] = None,
    ) -> int:
        """
        Удаляет items в терминальных состояниях (delivered/skipped/digested/
        failed) с fetched_at < cutoff_iso.
        'new' и 'pending' НЕ удаляются — они ещё не прошли скоринг и могут
        попасть в следующее окно.
        'deferred' НЕ удаляется здесь: deferred хранится до 3 дней
        (deferred_retention_days, v1.1.5) и очищается отдельным purge_items
        по state_at.

        M5-фикс: effective cutoff = более ранняя из (cutoff_iso, now - retention_days).
        Ничего новее retention_days границы не удаляется НИКОГДА, даже если
        дайджест пришёл раньше (например, запуск в 20:00 для окна ~12ч при
        retention_days=7: item от 08:05 живёт, несмотря на window_start~08:00).
        Консистентность с purge_items: оба метода уважают retention_days.
        Разница: delete_items_before использует fetched_at (когда item
        был впервые найден), purge_items использует state_at (когда item
        получил финальное состояние delivered/skipped/failed/digested).

        fetched_at всегда tz-aware ISO (см. save/_rows_to_items),
        строковое сравнение корректно (лексикографически для ISO+00:00);
        julianday() блокировал индекс и был NULL-небезопасен.

        Args:
            cutoff_iso: верхняя граница дайджест-окна (ISO-8601).
            retention_days: если задано — удалять не старше retention_days дней
                             (даже если cutoff_iso раньше). None = без ограничения
                             (обратная совместимость, deprecated: теперь None
                             означает DEFAULT_RETENTION_DAYS=1, как в purge_items).

        Возвращает количество удалённых строк items.
        """
        effective_cutoff = cutoff_iso
        # B3-фикс: None → DEFAULT_RETENTION_DAYS (консистентность с purge_items).
        # pipeline.py:2587 передаёт retention_days без дефолта (None → clamp
        # выключен). При отсутствии применяем дефолт 1, чтобы clamp заработал.
        resolved_retention = (
            retention_days if retention_days is not None
            else self.DEFAULT_RETENTION_DAYS
        )
        if resolved_retention is not None:
            retention_cutoff = (
                datetime.now(timezone.utc) - timedelta(days=resolved_retention)
            ).isoformat()
            # Более ранняя граница = менее агрессивная (меньше удалений).
            # min: если retention_days "моложе" window_start — используем
            # retention_days (не удаляем ничего новее retention-границы).
            if retention_cutoff < effective_cutoff:
                effective_cutoff = retention_cutoff

        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM items "
                "WHERE fetched_at < ? "
                "  AND state IN ('delivered','skipped','digested','failed')",
                (effective_cutoff,),
            )
            return cur.rowcount

    def get_oldest_unprocessed_fetched_at(self) -> Optional[str]:
        """
        Возвращает минимальный fetched_at среди items в состояниях
        'new'/'pending' (ещё не прошедших скоринг) или None, если таких нет.
        Нужен, чтобы не «потерять» их при закрытии окна дайджеста.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT MIN(fetched_at) FROM items WHERE state IN ('new','pending')",
        ).fetchone()
        return row[0] if row and row[0] is not None else None

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
                    logger.warning(
                        "_rows_to_items: fetched_at=%r не распознан как ISO, "
                        "fallback на datetime.min UTC. url=%s",
                        fetched_at_raw, row["url"][:80],
                    )
                    fetched_at = datetime.min.replace(tzinfo=timezone.utc)
            else:
                logger.warning(
                    "_rows_to_items: fetched_at NULL для url=%s, "
                    "fallback на datetime.min UTC",
                    row["url"][:80],
                )
                fetched_at = datetime.min.replace(tzinfo=timezone.utc)

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
            llm_title: Optional[str] = None
            for val_name, attr_name in (
                ("llm_score", "llm_score"),
                ("llm_summary", "llm_summary"),
                ("llm_reason", "llm_reason"),
                ("llm_urgent", "llm_urgent"),
                ("llm_title", "llm_title"),
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
                        elif val_name == "llm_title":
                            llm_title = str(val) if val else None
                except Exception:
                    pass

            # tags: может отсутствовать в старых записях (до добавления поля)
            try:
                tags_val = row["tags"]
            except Exception:
                tags_val = None

            # cluster_key: может отсутствовать в старых записях (до v12)
            cluster_key = None
            try:
                cluster_key = row["cluster_key"]
            except Exception:
                pass

            # lang: может отсутствовать в старых записях (до v13)
            lang_val = None
            try:
                lang_val = row["lang"]
            except Exception:
                pass

            # published_parsed: восстанавливаем из published (ISO) при чтении,
            # чтобы round-trip save→read сохранял данные.
            # published может быть в формате "DD.MM.YYYY HH:MM" или ISO,
            # поэтому try/except для каждого формата.
            published_val = row["published"]
            published_parsed_val = None
            if published_val:
                try:
                    published_parsed_val = datetime.fromisoformat(published_val)
                except Exception:
                    # Minor 2-фикс: парсим прод-формат "DD.MM.YYYY HH:MM UTC"
                    # (основной формат на проде, collector._format_pubdate).
                    if published_val.endswith(" UTC"):
                        base = published_val[:-4]
                        for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
                            try:
                                published_parsed_val = datetime.strptime(
                                    base, fmt,
                                ).replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                    if published_parsed_val is None:
                        # DD.MM.YYYY или DD.MM.YYYY HH:MM без суффикса
                        for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y"):
                            try:
                                published_parsed_val = datetime.strptime(
                                    published_val, fmt,
                                )
                                break
                            except ValueError:
                                continue

            ni = NewsItem(
                source=row["source"],
                url=row["url"],
                title=row["title"],
                summary=row["summary"],
                published=published_val or "—",
                tags=tags_val,
                published_parsed=published_parsed_val,
                cluster_key=cluster_key,
                lang=lang_val,
                fetched_at=fetched_at,
                llm_score=llm_score,
                llm_summary=llm_summary,
                llm_reason=llm_reason,
                llm_urgent=llm_urgent,
                llm_title=llm_title,
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
    ) -> List[str]:
        """
        Батчевое сохранение результатов LLM-оценки.
        records — список dict: {url, score, summary, reason, urgent?, cluster_key?, filtered_at?, title_en?}
        Если record содержит ключ 'urgent' (bool) — он пишется в llm_urgent (INTEGER 0/1).
        Если record содержит ключ 'cluster_key' (str) — он пишется в cluster_key.
        Если record содержит ключ 'title_en' (str, не-None) — он пишется в llm_title (v15).
        Если record НЕ содержит ключ 'title_en' или title_en is None — llm_title НЕ трогается
        (существующее значение в БД сохраняется). Это защищает retry-путь от затирания
        EN-заголовка при синтезе FilterResult без title_en.

        Per-record изоляция: каждый UPDATE выполняется в своей транзакции.
        Падение одного (DB locked, InterfaceError, …) НЕ затрагивает остальные.
        Упавший record логируется с url, его llm_score останется NULL,
        item останется в state=pending и подберётся recovery в следующем цикле.

        Возвращает список URL, для которых UPDATE прошёл успешно.
        При пустом входе возвращает пустой список.
        """
        if not records:
            return []
        now_iso = datetime.now(timezone.utc).isoformat()
        successful_urls: List[str] = []

        for rec in records:
            url = rec.get("url", "")
            cluster_key = rec.get("cluster_key")
            urgent = rec.get("urgent")
            title_en = rec.get("title_en")

            try:
                # Per-record transaction: каждый UPDATE изолирован.
                # Свой with self._tx() на каждый record.
                with self._tx() as conn:
                    if title_en is not None:
                        # title_en передан → пишем в llm_title
                        sets = "llm_score = ?, llm_summary = ?, llm_reason = ?, filtered_at = ?, llm_title = ?"
                        extra_sets = []
                        params_extra: list = []
                        if urgent is not None:
                            extra_sets.append("llm_urgent = ?")
                            params_extra.append(1 if urgent else 0)
                        if cluster_key is not None:
                            extra_sets.append("cluster_key = ?")
                            params_extra.append(cluster_key)
                        if extra_sets:
                            sets += ", " + ", ".join(extra_sets)

                        params = [
                            rec["score"], rec["summary"], rec["reason"],
                            rec.get("filtered_at") or now_iso, title_en,
                        ] + params_extra + [url]
                        cur = conn.execute(
                            f"UPDATE items SET {sets} WHERE url = ?",
                            params,
                        )
                    else:
                        # title_en отсутствует/None → НЕ пишем llm_title (preserve)
                        sets = "llm_score = ?, llm_summary = ?, llm_reason = ?, filtered_at = ?"
                        extra_sets = []
                        params_extra: list = []
                        if urgent is not None:
                            extra_sets.append("llm_urgent = ?")
                            params_extra.append(1 if urgent else 0)
                        if cluster_key is not None:
                            extra_sets.append("cluster_key = ?")
                            params_extra.append(cluster_key)
                        if extra_sets:
                            sets += ", " + ", ".join(extra_sets)

                        params = [
                            rec["score"], rec["summary"], rec["reason"],
                            rec.get("filtered_at") or now_iso,
                        ] + params_extra + [url]
                        cur = conn.execute(
                            f"UPDATE items SET {sets} WHERE url = ?",
                            params,
                        )
                    if cur.rowcount > 0:
                        successful_urls.append(url)
            except Exception as e:
                logger.error(
                    "update_llm_results_batch: UPDATE для %s упал: %s — "
                    "record пропущен, item останется pending",
                    url[:120], e,
                )
        return successful_urls

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
        """
        Пишет или обновляет ключ в meta. Полностью идемпотентна.

        Обёрнуто в _tx(): при ошибке commit соединение оставалось с открытой
        транзакцией, и все последующие BEGIN падали с «cannot start a
        transaction within a transaction» → записи отваливались до рестарта.
        """
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def _terminalize_stale_pending(
        self, pending_cutoff: str, now_iso: str, batch_size: Optional[int] = None,
    ) -> int:
        """
        Переводит застрявший pending (state_at < pending_cutoff) в 'failed'.

        Minor (волна 4): делает это ПОРЦИЯМИ по batch_size строк, каждая
        порция — отдельная короткая write-транзакция. Одним UPDATE на весь
        бэклог нельзя: десятки тысяч строк под WAL write-lock блокировали
        параллельных writers (poller) до «database is locked».

        Условие `state_at < pending_cutoff` после UPDATE само себя исключает
        (state_at = now_iso), поэтому цикл монотонно сходится; rowcount < batch
        означает, что хвост выбран, — выходим.
        """
        size = batch_size or self.PURGE_PENDING_BATCH_SIZE
        try:
            size = max(1, int(size))
        except (TypeError, ValueError):
            size = self.PURGE_PENDING_BATCH_SIZE
        total = 0
        while True:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    UPDATE items
                    SET state = 'failed', state_at = ?
                    WHERE rowid IN (
                        SELECT rowid FROM items
                        WHERE state = 'pending'
                          AND state_at < ?
                        LIMIT ?
                    )
                    """,
                    (now_iso, pending_cutoff, size),
                )
                updated = cur.rowcount
            total += updated
            if updated < size:
                break
        return total

    def purge_items(
        self,
        retention_days: int = None,
        feedback_retention_days: int = None,
        deferred_retention_days: int = None,
    ) -> int:
        """
        Удаляет items старше retention_days:
          - 'new': отдельное окно — не меньше DEFAULT_NEW_RETENTION_DAYS (7 дней),
            чтобы пауза или долгий скоринг-цикл не уничтожили ещё не оценённые
            новости, которые не попадали в дайджест.
          - 'deferred': старые отложенные, которые не попали ни в один дайджест
            (накапливались бесконечно при паузе или дайджест-окнах < max_items;
            digest берёт топ-30 по score, остаток копился вечно).
          - 'delivered', 'skipped', 'failed', 'digested': терминальные items
            старше retention_days (классический purge).
        'pending' — не удаляется (активный retry). Но pending, который не
        двигается (state_at не обновлялся) дольше PENDING_STALE_DAYS, —
        терминализируется в 'failed' (см. ниже), НЕ удаляется: удаление
        произойдёт обычным retention-окном уже для failed.

        Также вызывает prune_feedback_targets для очистки старых targets
        (feedback при этом СОХРАНЯЕТСЯ — обучающие данные).
        """
        retention_days = _coerce_days(
            retention_days, self.DEFAULT_RETENTION_DAYS, "purge_items.retention_days",
        )
        if retention_days == 0:
            logger.warning(
                "purge_items: retention_days=0 — все терминальные items "
                "будут удалены немедленно (осознанный путь? проверьте конфиг)",
            )
        deferred_retention_days = _coerce_days(
            deferred_retention_days,
            self.DEFAULT_DEFERRED_RETENTION_DAYS,
            "purge_items.deferred_retention_days",
        )
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        deferred_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=deferred_retention_days)
        ).isoformat()
        # 'new' живут минимум DEFAULT_NEW_RETENTION_DAYS дней, даже если
        # retention_days передан как 0 (например, из тестов).
        new_retention_days = max(retention_days, self.DEFAULT_NEW_RETENTION_DAYS)
        cutoff_new = (
            datetime.now(timezone.utc) - timedelta(days=new_retention_days)
        ).isoformat()
        # Застрявший pending: state_at не двигался PENDING_STALE_DAYS дней.
        pending_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.PENDING_STALE_DAYS)
        ).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()

        total_deleted = 0
        pending_failed = 0
        try:
            # Pending-терминализация (НЕ удаление): защита от роста бэклога,
            # когда retry_count не инкрементируется (ChatIdNotBoundError /
            # BotBlocked) — такие items не снимет recover_stuck_pending
            # (его фильтр retry_count >= max_retries) и не удалит purge.
            # state_at обновляется при каждом переходе в pending, поэтому
            # «state_at старше N дней» = item не двигается N дней.
            # Minor (волна 4): вынесено ИЗ общей _tx и выполняется батчами
            # (см. _terminalize_stale_pending) — иначе один UPDATE на весь
            # бэклог держал write-lock вместе с тремя DELETE.
            pending_failed = self._terminalize_stale_pending(
                pending_cutoff, now_iso,
            )

            with self._tx() as conn:
                # терминальные (retention_days)
                cur = conn.execute(
                    """
                    DELETE FROM items
                    WHERE state IN ('delivered', 'skipped', 'failed', 'digested')
                      AND state_at < ?
                    """,
                    (cutoff,),
                )
                terminal_del = cur.rowcount

                # new: отдельное окно (DEFAULT_NEW_RETENTION_DAYS, минимум 7 дней)
                cur_new = conn.execute(
                    """
                    DELETE FROM items
                    WHERE state = 'new'
                      AND state_at < ?
                    """,
                    (cutoff_new,),
                )
                new_del = cur_new.rowcount

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

                total_deleted = terminal_del + new_del + deferred_del
            if pending_failed:
                logger.warning(
                    "purge_items: %d застрявших pending (state_at старше %d дней) "
                    "переведены в 'failed' — retry-выборка их больше не подберёт",
                    pending_failed, self.PENDING_STALE_DAYS,
                )
            if total_deleted:
                parts = []
                if terminal_del:
                    parts.append(f"terminal={terminal_del}")
                if new_del:
                    parts.append(f"new={new_del}")
                if deferred_del:
                    parts.append(f"deferred={deferred_del}")
                logger.info("purge_items: удалено %s", " + ".join(parts))
        except Exception as e:
            logger.error("purge_items: ошибка очистки — %s", e)

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
            logger.error("purge_seen_titles: ошибка очистки — %s", e)
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

        Storage Minor 4: INSERT OR IGNORE + SELECT — таблица имеет
        UNIQUE(url, kind), поэтому голый INSERT падал IntegrityError
        при повторе (retry/повторная доставка). Поведение теперь
        идемпотентно: возвращается id существующей записи.

        Returns:
            id новой (или уже существующей для url+kind) записи feedback_targets.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO feedback_targets
                (url, source, title, summary, llm_score, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, source, title, summary, llm_score, kind, now_iso),
            )
            row = conn.execute(
                "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
                (url, kind),
            ).fetchone()
            return row[0]  # type: ignore

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
        F1/Minor#4: идемпотентное создание feedback_target.
        Minor#4-фикс: использует INSERT OR IGNORE + UNIQUE(url, kind) вместо
        SELECT→INSERT (устранён TOCTOU). В одной транзакции: гарантирует,
        что retry-вызовы возвращают тот же id без лишних INSERT.

        Args:
            item: NewsItem
            kind: 'push', 'fallback' или 'digest'

        Returns:
            id существующего или нового feedback_targets.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            # Minor#4: INSERT OR IGNORE — безопасно при конкурентном создании.
            conn.execute(
                """
                INSERT OR IGNORE INTO feedback_targets
                (url, source, title, summary, llm_score, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (item.url, item.source, item.title, item.summary,
                 item.llm_score, kind, now_iso),
            )
            row = conn.execute(
                "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
                (item.url, kind),
            ).fetchone()
            return row[0]  # type: ignore

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
        F1/Minor#4: идемпотентное создание feedback_target (plain-вариант без NewsItem).
        Minor#4-фикс: INSERT OR IGNORE + UNIQUE(url, kind), транзакция.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO feedback_targets
                (url, source, title, summary, llm_score, kind, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, source, title, summary, llm_score, kind, now_iso),
            )
            row = conn.execute(
                "SELECT id FROM feedback_targets WHERE url = ? AND kind = ?",
                (url, kind),
            ).fetchone()
            return row[0]  # type: ignore

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
            " ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC, "
            "ROWID DESC "
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

        Невалидный (не приводимый к int) older_than_days → 0 и warning,
        БЕЗ удаления: feedback_targets — данные для кнопок, сносить их
        из-за опечатки в конфиге нельзя.

        older_than_days <= 0 → 0 и warning, БЕЗ удаления. Раньше здесь стоял
        max(days, 0): -1 (опечатка `feedback.retention_days: -1`) давал
        cutoff = now и сносил ВСЕ targets — кнопки под старыми сообщениями
        переставали работать, что прямо противоречило докстрингу «невалидный
        → без удаления». 0 теперь тоже трактуется как «не чистить»:
        осмысленного «удалить всё» для кнопок не существует.
        """
        try:
            days = int(older_than_days)
        except (TypeError, ValueError):
            logger.warning(
                "prune_feedback_targets: невалидный older_than_days=%r — пропуск",
                older_than_days,
            )
            return 0
        if days <= 0:
            logger.warning(
                "prune_feedback_targets: older_than_days=%r (<=0) — пропуск, "
                "targets не удаляются",
                older_than_days,
            )
            return 0
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
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
                    deleted, days,
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
        limit: int = 0,
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
            limit: лимит возвращаемых записей; 0 (дефолт) — без лимита,
                выборка только по окну fetched_at (индекс idx_seen_fetched)
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

        limit_sql = ""
        if limit and limit > 0:
            limit_sql = " LIMIT ?"
            params.append(int(limit))

        # title_normalized может отсутствовать (старые записи до v9)
        rows = conn.execute(
            f"""
            SELECT title_hash,
                   COALESCE(title_normalized, title_hash) AS norm,
                   source,
                   fetched_at
            FROM seen_titles
            WHERE fetched_at > ?{source_filter}
            ORDER BY fetched_at DESC{limit_sql}
            """,
            params,
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

    def get_shown_for_window(
        self,
        cutoff_iso: str,
        limit: int = 200,
    ) -> List[dict]:
        """
        Возвращает новости, УЖЕ показанные (state IN ('digested', 'delivered'))
        за текущее дайджест-окно (fetched_at >= cutoff_iso).

        Используется как RECENT-контекст в LLM-дедупликации дайджеста
        (``_cluster_deferred_duplicates``), чтобы отсекать кросс-оконные дубли:
        новость, показанная утром (digested), не попадала в вечерний дайджест
        даже если вечером другой источник опубликовал перефразированную версию.

        Формат возврата — ``List[dict]`` с ключами ``title`` и ``source``:
        это exactly тот формат, который ожидает ``LLMFilter.detect_duplicates``
        для второго аргумента ``recent``.

        Лимит 200 — компромисс между полнотой RECENT-контекста и стоимостью
        токенов: LLM-дедуп шлёт все recent titles в каждый чанк, при ~200 items
        и ~100 символах на title → ~20K символов, что укладывается в бюджет
        MAX_DEDUP_CTX_CHARS (~12000) с запасом на kept/dup из предыдущих чанков.
        Реальное окно дайджеста (24ч) редко генерирует больше 100–150 уникальных
        digested/delivered items, 200 — консервативный верхний порог.

        NOTE: кандидаты текущего окна (``deferred``) НЕ попадают в выборку
        (state='deferred' ≠ 'digested'/'delivered'), поэтому инвариант
        «recent не содержит кандидатов» соблюдается автоматически.

        NOTE: ``llm_summary`` и ``url`` НЕ возвращаются — они не используются
        в RECENT-контексте LLM-дедупа (см. ``build_dedup_context_message``:
        только title+source, summary/url отброшены в MAJOR-фиксе ради
        символьного бюджета и устранения O(N²) роста).

        MAJOR-фикс (4d): граница — COALESCE(state_at, fetched_at), где
        state_at = момент доставки (set_digested/delivered). Для старых строк
        (state_at=NULL) fallback → fetched_at. Новость, пролежавшая в deferred
        >24ч и доставленная утром (state_at внутри окна), попадает в RECENT
        вечернего окна.

        Args:
            cutoff_iso: граница окна (ISO-8601, UTC). Все items с
                fetched_at >= cutoff_iso попадают в выборку.
                Для дайджеста это ``window_start_iso`` (начало текущего окна).
            limit: максимум возвращаемых записей (default 200).
                0 = без лимита (используйте осознанно).

        Returns:
            List[dict], каждый dict: ``{'title': str, 'source': str}``.
            Пустой список если ничего не найдено или при ошибке БД.
        """
        try:
            conn = self._connect()
            params: list = [cutoff_iso]
            limit_sql = ""
            if limit and limit > 0:
                limit_sql = " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(
                f"""
                SELECT title, source
                FROM items
                WHERE state IN ('digested', 'delivered')
                  AND COALESCE(state_at, fetched_at) >= ?
                ORDER BY COALESCE(state_at, fetched_at) DESC{limit_sql}
                """,
                params,
            ).fetchall()
            return [{"title": r[0], "source": r[1]} for r in rows]
        except Exception as e:
            logger.warning(
                "get_shown_for_window: ошибка при cutoff=%s: %s — "
                "возвращаем пустой список (fail-open)",
                cutoff_iso, e,
            )
            return []

    def save_skipped(self, item: NewsItem) -> bool:
        """
        Сохраняет item в items со state='skipped' (кросс-источниковый дубль).
        Это нужно чтобы:
          (a) URL-дедуп ловил его в следующих циклах
          (б) он не попал в пуш/дайджест

        Returns True если ВСТАВЛЕН (URL ещё не было), False если IGNORE.
        """
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        MAX_SUMMARY_CHARS = 6000
        summary = item.summary[:MAX_SUMMARY_CHARS]
        title = item.title[:MAX_TITLE_CHARS] if item.title else item.title
        url_norm = normalize_url(item.url)

        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, url_norm, title, summary, published, tags, cluster_key,
                     lang, fetched_at, state, state_at, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'skipped', ?, 0)
                    """,
                    (
                        item.source,
                        item.url,
                        url_norm,
                        title,
                        summary,
                        item.published,
                        getattr(item, "tags", None),
                        getattr(item, "cluster_key", None),
                        getattr(item, "lang", None),
                        now_iso,          # fetched_at
                        now_iso,          # state_at
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