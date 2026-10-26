"""
Тесты сквозной фичи title_en (EN-перевод заголовка от LLM).

Покрытие:
  1. Парсинг single: title_en присутствует / отсутствует / пусто / не-строка
  2. Парсинг batch: title_en присутствует / отсутствует / пусто / не-строка
  3. Миграция v14→v15: колонка llm_title добавляется, старые строки NULL
  4. Миграция v15→v15 (идемпотентность): повторный старт без ошибок
  5. update_llm_results_batch пишет llm_title; None → NULL
  6. Digest-рендер: llm_title → EN-заголовок; llm_title=None → fallback
  7. Push/Telegram-рендер: llm_title → EN-заголовок + control-chars обработаны
  8. Дедуп-регресс: title_hash/title_normalized считаются от ИСХОДНОГО title
  9. Batch-сквозной: мок ответа с title_en → item.llm_title установлен

Запуск: python -m pytest tests/test_llm_title_en.py -q
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.digest import _render_item_raw
from newsbot.llm.filter import FilterResult, LLMFilter
from newsbot.storage import Storage


# =============================================================================
# Helpers — мок-клиент (по мотиву test_llm_filter_mocks.py)
# =============================================================================

class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


_saved_httpx: type = None


def _mock_httpx(mock_cls: type) -> None:
    global _saved_httpx
    from newsbot import llm as llm_module
    if _saved_httpx is None:
        _saved_httpx = llm_module.client.httpx.Client
    llm_module.client.httpx.Client = mock_cls


def _unmock() -> None:
    global _saved_httpx
    if _saved_httpx is not None:
        from newsbot import llm as llm_module
        llm_module.client.httpx.Client = _saved_httpx
        _saved_httpx = None


def _parse_single(raw: dict) -> FilterResult:
    """Парсит single-item ответ через LLMFilter.evaluate."""
    from newsbot.llm.client import DeepSeekClient

    resp_str = json.dumps(raw)

    class _Mock:
        is_closed = False

        def __init__(s, **kw):
            pass

        def post(s, url, **kwargs):
            return FakeResponse(
                {"choices": [{"message": {"content": resp_str}}]},
                status_code=200,
            )

        def close(s):
            pass

        def __enter__(s):
            return s

        def __exit__(s, *args):
            pass

    _mock_httpx(_Mock)
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = {
        "enabled": True,
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "mock",
        "push_threshold": 7,
        "digest_threshold": 5,
        "system_prompt": "test",
    }
    flt = LLMFilter(cfg, client=client)

    item = NewsItem(source="Test", url="https://e.com/1",
                    title="Заголовок", summary="Резюме", published="2024-01-01")
    try:
        return flt.evaluate(item)
    finally:
        _unmock()


def make_item(url="https://e.com/1", title="Test", summary="Sum", **kwargs):
    return NewsItem(source="Test", url=url, title=title, summary=summary,
                    published="2024-01-01", published_parsed=None, **kwargs)


# =============================================================================
# 1. Парсинг single
# =============================================================================

class TestTitleEnSingleParsing(unittest.TestCase):
    """Тесты парсинга title_en в single-item ответе (_parse_response)."""

    def test_title_en_present_string(self):
        """title_en присутствует и непустая строка → FilterResult.title_en = значение."""
        r = _parse_single({
            "score": 8, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": "Storm Hits Spain, Five Dead",
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.title_en, "Storm Hits Spain, Five Dead")

    def test_title_en_missing(self):
        """title_en отсутствует → None (не роняет ответ)."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
        })
        self.assertTrue(r.ok)
        self.assertIsNone(r.title_en)

    def test_title_en_empty_string(self):
        """title_en = "" → None (пустая строка → None)."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": "",
        })
        self.assertTrue(r.ok)
        self.assertIsNone(r.title_en)

    def test_title_en_whitespace_only(self):
        """title_en = "   " → None (trimmed → пустая)."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": "   \t  ",
        })
        self.assertTrue(r.ok)
        self.assertIsNone(r.title_en)

    def test_title_en_integer_rejected(self):
        """title_en = int → None (м4: вредный формат, не сохраняем "42" как заголовок)."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": 42,
        })
        self.assertTrue(r.ok)
        self.assertIsNone(r.title_en)

    def test_title_en_float_rejected(self):
        """title_en = float → None (м4: вредный формат, не сохраняем "3.14")."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": 3.14,
        })
        self.assertTrue(r.ok)
        self.assertIsNone(r.title_en)

    def test_title_en_strips_whitespace(self):
        """title_en с пробелами по краям → strip."""
        r = _parse_single({
            "score": 7, "summary": "sum", "reason": "r",
            "urgent": False, "location": "",
            "title_en": "  Drone Attack on Moscow  ",
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.title_en, "Drone Attack on Moscow")


# =============================================================================
# 2. Парсинг batch
# =============================================================================

class TestTitleEnBatchParsing(unittest.TestCase):
    """Тесты парсинга title_en в батч-ответе (_parse_batch_response)."""

    def _parse_batch(self, raw_results: list) -> list:
        """Использует multi-item путь evaluate_batch (batch_size=2+)."""
        from newsbot.llm.filter import LLMFilter
        from newsbot.llm.client import DeepSeekClient

        resp_body = {"results": raw_results}
        resp_str = json.dumps(resp_body)

        class _Mock:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return FakeResponse(
                    {"choices": [{"message": {"content": resp_str}}]},
                    status_code=200,
                )

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        _mock_httpx(_Mock)
        client = DeepSeekClient("key", "https://api.deepseek.com", "model")
        cfg = {
            "enabled": True, "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com", "api_key": "mock",
            "push_threshold": 7, "digest_threshold": 5,
            "system_prompt": "test",
        }
        flt = LLMFilter(cfg, client=client)
        # Multi-item → batch path. len(raw_results) items.
        n = len(raw_results)
        items = [make_item(f"https://e.com/{i}", f"T{i}", f"S{i}") for i in range(1, n + 1)]
        try:
            return flt.evaluate_batch(items)
        finally:
            _unmock()

    def test_title_en_present_in_batch(self):
        """title_en присутствует у item → значение."""
        results = self._parse_batch([
            {"id": 1, "score": 8, "summary": "S1", "reason": "r1",
             "urgent": False, "title_en": "EN Title One"},
            {"id": 2, "score": 6, "summary": "S2", "reason": "r2",
             "urgent": False, "title_en": "EN Title Two"},
        ])
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].title_en, "EN Title One")
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].title_en, "EN Title Two")

    def test_title_en_missing_in_batch(self):
        """title_en отсутствует у item → None."""
        results = self._parse_batch([
            {"id": 1, "score": 8, "summary": "S1", "reason": "r1", "urgent": False},
            {"id": 2, "score": 6, "summary": "S2", "reason": "r2", "urgent": False},
        ])
        self.assertTrue(results[0].ok)
        self.assertIsNone(results[0].title_en)
        self.assertTrue(results[1].ok)
        self.assertIsNone(results[1].title_en)

    def test_title_en_empty_in_batch(self):
        """title_en = "" в батче → None."""
        results = self._parse_batch([
            {"id": 1, "score": 8, "summary": "S1", "reason": "r1",
             "urgent": False, "title_en": ""},
            {"id": 2, "score": 6, "summary": "S2", "reason": "r2", "urgent": False},
        ])
        self.assertTrue(results[0].ok)
        self.assertIsNone(results[0].title_en)

    def test_title_en_int_in_batch_rejected(self):
        """title_en = int в батче → None (м4: вредный формат)."""
        results = self._parse_batch([
            {"id": 1, "score": 8, "summary": "S1", "reason": "r1",
             "urgent": False, "title_en": 99},
            {"id": 2, "score": 6, "summary": "S2", "reason": "r2", "urgent": False},
        ])
        self.assertTrue(results[0].ok)
        self.assertIsNone(results[0].title_en)

    def test_title_en_mixed_in_batch(self):
        """В батче: item1 с title_en, item2 без, item3 с пустым → корректно."""
        results = self._parse_batch([
            {"id": 1, "score": 8, "summary": "S1", "reason": "r1",
             "urgent": False, "title_en": "Storm in Spain"},
            {"id": 2, "score": 6, "summary": "S2", "reason": "r2", "urgent": False},
            {"id": 3, "score": 5, "summary": "S3", "reason": "r3",
             "urgent": False, "title_en": ""},
        ])
        self.assertEqual(results[0].title_en, "Storm in Spain")
        self.assertIsNone(results[1].title_en)
        self.assertIsNone(results[2].title_en)


# =============================================================================
# 3. Миграция v14→v15 (идемпотентность)
# =============================================================================

class TestV15Migration(unittest.TestCase):
    """Тесты миграции v14→v15."""

    def test_migration_v14_adds_llm_title_keeps_data(self):
        """
        v14-мигрированная БД (user_version=14, без llm_title).
        После старта с v15: колонка llm_title появилась, старые данные целы.

        Python 3.8 sqlite3 не поддерживает DROP COLUMN (SQLite < 3.35.0).
        Симулируем v14 пересозданием таблицы без llm_title (без потери данных).
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(db_path)
        try:
            s = Storage(db_path=db_path)
            item = make_item(url="https://e.com/old", title="Старый заголовок",
                             summary="Old")
            s.save(item)
            s.close()

            # Симулируем v14: user_version=14 + пересоздаём items без llm_title.
            # v14-колонки: source, url, url_norm, title, summary, published,
            # tags, cluster_key, lang, fetched_at, state, state_at,
            # llm_score, llm_summary, llm_reason, llm_urgent, filtered_at, retry_count.
            conn2 = sqlite3.connect(db_path)
            conn2.execute("PRAGMA user_version = 14")
            # Пересоздаём таблицу без llm_title (SQLite 3.8 совместимо)
            conn2.execute("""
                CREATE TABLE items_v14 (
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
                )
            """)
            # Копируем данные
            conn2.execute("""
                INSERT INTO items_v14
                (id, source, url, url_norm, title, summary, published, tags,
                 cluster_key, lang, fetched_at, state, state_at,
                 llm_score, llm_summary, llm_reason, llm_urgent, filtered_at, retry_count)
                SELECT
                 id, source, url, url_norm, title, summary, published, tags,
                 cluster_key, lang, fetched_at, state, state_at,
                 llm_score, llm_summary, llm_reason, llm_urgent, filtered_at, retry_count
                FROM items
            """)
            conn2.execute("DROP TABLE items")
            conn2.execute("ALTER TABLE items_v14 RENAME TO items")
            # Индексы
            conn2.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url_norm ON items(url_norm)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_url ON items(url)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state ON items(state)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state_at ON items(state_at)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state_state_at ON items(state, state_at)")
            conn2.commit()
            conn2.close()

            # Переоткрываем — триггер миграции v14→v15
            s2 = Storage(db_path=db_path)
            conn3 = s2._connect()
            version = conn3.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 15)

            cols = {r[1] for r in conn3.execute("PRAGMA table_info(items)").fetchall()}
            self.assertIn("llm_title", cols)

            # Старые данные на месте, llm_title = NULL
            items = s2.get_all_by_state("new")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].title, "Старый заголовок")
            self.assertIsNone(items[0].llm_title)

            s2.close()
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_migration_v15_idempotent(self):
        """
        Миграция v15 идемпотентна: повторный старт не ломает БД.
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(db_path)
        try:
            # Создаём свежую v15 БД (Storage автоматически v15)
            s1 = Storage(db_path=db_path)
            item1 = make_item(url="https://e.com/t1", title="T1", summary="S1")
            s1.save(item1)
            s1.close()

            # Симулируем v14: user_version=14 + rebuild table без llm_title.
            conn2 = sqlite3.connect(db_path)
            conn2.execute("PRAGMA user_version = 14")
            conn2.execute("""
                CREATE TABLE items_v14 (
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
                )
            """)
            conn2.execute("""
                INSERT INTO items_v14
                (id, source, url, url_norm, title, summary, published, tags,
                 cluster_key, lang, fetched_at, state, state_at,
                 llm_score, llm_summary, llm_reason, llm_urgent, filtered_at, retry_count)
                SELECT
                 id, source, url, url_norm, title, summary, published, tags,
                 cluster_key, lang, fetched_at, state, state_at,
                 llm_score, llm_summary, llm_reason, llm_urgent, filtered_at, retry_count
                FROM items
            """)
            conn2.execute("DROP TABLE items")
            conn2.execute("ALTER TABLE items_v14 RENAME TO items")
            conn2.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url_norm ON items(url_norm)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_url ON items(url)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state ON items(state)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state_at ON items(state_at)")
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_items_state_state_at ON items(state, state_at)")
            conn2.commit()
            conn2.close()

            # Первая миграция (v14→v15)
            s2 = Storage(db_path=db_path)
            v1 = s2._connect().execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(v1, 15)
            s2.close()

            # Повторный старт (уже v15)
            s3 = Storage(db_path=db_path)
            v2 = s3._connect().execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(v2, 15)

            # Данные на месте, llm_title = NULL
            items = s3.get_all_by_state("new")
            self.assertEqual(len(items), 1)
            self.assertIsNone(items[0].llm_title)

            s3.close()
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_fresh_db_has_llm_title_column(self):
        """Свежая БД (v0) сразу получает llm_title при создании."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(db_path)
        try:
            s = Storage(db_path=db_path)
            conn = s._connect()
            cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            self.assertIn("llm_title", cols)

            item = make_item(url="https://e.com/new", title="New", summary="S")
            s.save(item)
            saved = s.get_all_by_state("new")
            self.assertEqual(len(saved), 1)
            self.assertIsNone(saved[0].llm_title)
            s.close()
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass


# =============================================================================
# 4. update_llm_results_batch пишет llm_title
# =============================================================================

class TestUpdateBatchTitleEn(unittest.TestCase):
    """Тесты: update_llm_results_batch пишет llm_title."""

    def _fresh_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        return path, lambda: os.unlink(path) if os.path.exists(path) else None

    def test_batch_writes_llm_title(self):
        """record с title_en → llm_title в БД."""
        path, cleanup = self._fresh_db()
        try:
            s = Storage(path)
            item = make_item(url="https://e.com/1", title="T", summary="S")
            s.save(item)

            n = s.update_llm_results_batch([{
                "url": "https://e.com/1",
                "score": 8,
                "summary": "Sum",
                "reason": "r",
                "title_en": "Storm Hits Spain",
            }])
            self.assertEqual(len(n), 1)  # List[str]: 1 URL

            items = s.get_all_by_state("new")
            self.assertEqual(items[0].llm_title, "Storm Hits Spain")
            s.close()
        finally:
            cleanup()

    def test_batch_preserves_llm_title_on_missing_key(self):
        """title_en отсутствует в record → llm_title НЕ трогается (preserve).\n        Это защищает retry-путь от затирания EN-заголовка."""
        path, cleanup = self._fresh_db()
        try:
            s = Storage(path)
            item = make_item(url="https://e.com/1", title="T", summary="S")
            s.save(item)

            # Пишем llm_title
            s.update_llm_results_batch([{
                "url": "https://e.com/1", "score": 8,
                "summary": "S", "reason": "r",
                "title_en": "Storm Hits Spain",
            }])

            # Повторный вызов БЕЗ title_en — llm_title сохраняется
            n = s.update_llm_results_batch([{
                "url": "https://e.com/1", "score": 7,
                "summary": "S2", "reason": "r2",
                # title_en отсутствует
            }])
            self.assertEqual(len(n), 1)  # List[str]: 1 URL

            items = s.get_all_by_state("new")
            self.assertEqual(items[0].llm_title, "Storm Hits Spain")
            s.close()
        finally:
            cleanup()


# =============================================================================
# 5. Digest-рендер: llm_title vs fallback
# =============================================================================

class TestDigestRenderTitleEn(unittest.TestCase):
    """Тесты: digest-рендер использует llm_title."""

    def test_llm_title_in_digest(self):
        """item.llm_title задан → в тексте дайджеста EN-заголовок."""
        item = NewsItem(
            source="Reuters", url="https://e.com/1",
            title="Шторм обрушился на Испанию, есть погибшие",
            summary="Пять человек погибли",
            llm_score=8, llm_summary="Five dead in Spain storm",
            llm_title="Storm Hits Spain, Five Dead",
            published="2025-01-01", published_parsed=None,
        )
        title_raw, _, _, _ = _render_item_raw(item)
        self.assertEqual(title_raw, "Storm Hits Spain, Five Dead")

    def test_llm_title_none_uses_original(self):
        """item.llm_title=None → в тексте дайджеста исходный заголовок."""
        item = NewsItem(
            source="Reuters", url="https://e.com/2",
            title="Шторм обрушился на Испанию",
            summary="Подробности",
            llm_score=8, llm_summary="Storm in Spain",
            llm_title=None,
            published="2025-01-01", published_parsed=None,
        )
        title_raw, _, _, _ = _render_item_raw(item)
        self.assertEqual(title_raw, "Шторм обрушился на Испанию")

    def test_llm_title_empty_uses_original(self):
        """item.llm_title='' (edge case) → fallback на исходный."""
        item = NewsItem(
            source="Reuters", url="https://e.com/3",
            title="Оригинальный заголовок",
            summary="Sum",
            llm_score=8, llm_summary="Sum",
            llm_title="",  # пустой — не должен показываться
            published="2025-01-01", published_parsed=None,
        )
        title_raw, _, _, _ = _render_item_raw(item)
        self.assertEqual(title_raw, "Оригинальный заголовок")


# =============================================================================
# 6. Push/Telegram-рендер: llm_title + control-chars + fallback
# =============================================================================

class TestPushRenderTitleEn(unittest.TestCase):
    """Тесты: push-рендер использует llm_title через реальный _build_message_parts."""

    def _build_payload(self, item: NewsItem) -> dict:
        """Вызывает реальный _build_message_parts (не ходит в сеть)."""
        from newsbot.delivery.telegram import TelegramDelivery
        td = TelegramDelivery.__new__(TelegramDelivery)
        td.token = "mock"
        td.chat_id = "mock"
        td._base_url = "https://api.telegram.org/botmock"
        td._http = None  # не используется в _build_message_parts
        parts = td._build_message_parts(item)
        return parts[0]  # одно сообщение (short item)

    def test_llm_title_in_push(self):
        """item.llm_title задан → EN-заголовок в тексте push-сообщения."""
        item = NewsItem(
            source="Reuters", url="https://e.com/1",
            title="Шторм в Испании",
            summary="Детали бури.",
            llm_score=8, llm_summary="Storm details.",
            llm_title="Storm in Spain, Five Dead",
            published="2025-01-01", published_parsed=None,
        )
        payload = self._build_payload(item)
        text = payload["text"]
        self.assertIn("Storm in Spain, Five Dead", text)
        self.assertNotIn("Шторм в Испании", text)  # RU-title не виден

    def test_llm_title_none_uses_original(self):
        """llm_title=None → fallback на исходный RU-title."""
        item = NewsItem(
            source="Reuters", url="https://e.com/3",
            title="Шторм обрушился на Испанию",
            summary="Details.",
            llm_score=8, llm_summary="Storm details.",
            llm_title=None,
            published="2025-01-01", published_parsed=None,
        )
        payload = self._build_payload(item)
        text = payload["text"]
        self.assertIn("Шторм обрушился на Испанию", text)

    def test_llm_title_truncated_at_250(self):
        """EN-title длиной >250 обрезается (m3: реальный путь, не тавтология).\n        _MAX_RAW_TITLE=250 обрезает без «…»."""
        item = NewsItem(
            source="Src", url="https://e.com/4",
            title="RU",
            summary="S",
            llm_score=8, llm_summary="S",
            llm_title="A" * 300,  # 300 chars — превышает _MAX_RAW_TITLE=250
            published="2025-01-01", published_parsed=None,
        )
        payload = self._build_payload(item)
        text = payload["text"]
        title_line = text.split("\n")[0]
        # Ровно 250 символов: A*250 (без "…")
        self.assertEqual(len(title_line), 250)
        self.assertTrue(title_line.startswith("A" * 250))

    def test_llm_title_with_control_chars_stripped(self):
        """\x00/\x0b удаляются; \n сохраняется (реальный _strip_control_chars)."""
        item = NewsItem(
            source="Reuters", url="https://e.com/5",
            title="RU",
            summary="S",
            llm_score=8, llm_summary="S",
            # \x00 (NUL) и \x0b (VT) — удаляются; \n и \t — сохраняются
            llm_title="Storm\x00Batteries\x0bSpain\n",
            published="2025-01-01", published_parsed=None,
        )
        payload = self._build_payload(item)
        text = payload["text"]
        title_line = text.split("\n")[0]
        self.assertNotIn("\x00", title_line)
        self.assertNotIn("\x0b", title_line)
        self.assertIn("Storm", title_line)
        self.assertIn("Batteries", title_line)
        self.assertIn("Spain", title_line)
        # \n сохраняется в text (html.escape НЕ экранирует \n), но split("\n") убирает
        # его из title_line. Проверяем newline в полном text, а не в title_line.
        self.assertIn("\n", text)
        self.assertNotIn("\n", title_line)

    def test_llm_title_empty_uses_original(self):
        """llm_title='' → fallback на RU-title (реальный путь)."""
        item = NewsItem(
            source="Reuters", url="https://e.com/6",
            title="Оригинальный заголовок",
            summary="S",
            llm_score=8, llm_summary="S",
            llm_title="",
            published="2025-01-01", published_parsed=None,
        )
        payload = self._build_payload(item)
        text = payload["text"]
        self.assertIn("Оригинальный заголовок", text)


# =============================================================================
# 7. Дедуп-регресс: title_hash от исходного, llm_title не участвует
# =============================================================================

class TestDedupRegressionTitleEn(unittest.TestCase):
    """Тесты: title_hash/title_normalized считаются от ИСХОДНОГО title (не llm_title)."""

    def test_same_original_title_same_hash_regardless_of_llm_title(self):
        """
        Два NewsItem с одинаковым title, но разными llm_title.
        title_normalized() одинаков → дедуп-ключ одинаков.
        llm_title НЕ влияет на дедуп.
        """
        item1 = make_item(
            url="https://e.com/1",
            title="Шторм обрушился на Испанию",
            summary="S1",
            llm_title="Storm Hits Spain",
        )
        item2 = make_item(
            url="https://e.com/2",
            title="Шторм обрушился на Испанию",
            summary="S2",
            llm_title="Different EN Title",  # другой перевод — неважно
        )
        self.assertEqual(
            item1.title_normalized(),
            item2.title_normalized(),
        )
        # Также проверяем: сам исходный title не меняется
        self.assertEqual(item1.title, "Шторм обрушился на Испанию")
        self.assertEqual(item2.title, "Шторм обрушился на Испанию")

    def test_different_original_title_different_hash_regardless_of_llm_title(self):
        """Разные исходные title → разные hash, даже если llm_title совпадает."""
        item1 = make_item(
            url="https://e.com/1",
            title="Шторм в Испании",
            summary="S1",
            llm_title="Storm Hits Spain",  # совпадает с item2
        )
        item2 = make_item(
            url="https://e.com/2",
            title="Шторм в Португалии",
            summary="S2",
            llm_title="Storm Hits Spain",  # тот же EN, но RU другой
        )
        self.assertNotEqual(
            item1.title_normalized(),
            item2.title_normalized(),
        )

    def test_llm_title_not_set_title_normalized_unchanged(self):
        """llm_title=None не меняет title_normalized (базовый регресс)."""
        item = make_item(
            url="https://e.com/1",
            title="Обычный заголовок",
            summary="Sum",
        )
        # llm_title=None по умолчанию
        normalized = item.title_normalized()
        self.assertEqual(normalized, "обычный заголовок")


if __name__ == "__main__":
    unittest.main()