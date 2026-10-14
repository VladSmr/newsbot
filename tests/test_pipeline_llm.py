"""
Тесты интеграции LLM-фильтра в pipeline (этап 2b-2a).
Без сети — fake-фильтр, мок delivery.

Запуск: python -m pytest tests -q
        python tests/test_pipeline_llm.py  (standalone)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest import mock

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore

# проектный корень → пакет
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.llm.filter import FilterResult, LLMFilter, route
from newsbot.llm.client import BalanceInfo
from newsbot.storage import Storage
from newsbot.pipeline import NewsPipeline


# ─────────────────────────────────────────────────────────────────
# Fake-фильтр: подменяет LLMFilter.evaluate()
# ─────────────────────────────────────────────────────────────────

class FakeLLMFilter:
    """
    Подставляется в pipeline.llm_filter.
    rules: score → FilterResult
      8+  → ok=True, push
      5-6 → ok=True, digest
      3-4 → ok=True, skip
      -1  → ok=False (fallback)
    """
    def __init__(self, score_map: dict):
        """
        score_map: {url_substring: score}
        score=-1 → ok=False (fallback).
        """
        self.score_map = score_map
        self.push_threshold = 7
        self.digest_threshold = 5

    def evaluate(self, item: NewsItem, feedback_context=None) -> FilterResult:
        for pattern, score in self.score_map.items():
            if pattern in item.url or pattern in item.title:
                if score == -1:
                    return FilterResult(ok=False)
                return FilterResult(
                    ok=True,
                    score=score,
                    summary=f"LLM-выжимка для: {item.title[:40]}",
                    reason=f"тест score={score}",
                )
        # по умолчанию — fallback
        return FilterResult(ok=False)


class MockDelivery:
    """Мок доставки: принимает всё, возвращает все URL."""
    def __init__(self, fail_on_urls: List[str] = None):
        self.sent: List[str] = []
        self.fail_on_urls = set(fail_on_urls or [])

    def send_batch(self, items: List[NewsItem]) -> List[str]:
        result = []
        for item in items:
            if item.url in self.fail_on_urls:
                continue
            result.append(item.url)
            self.sent.append(item.url)
        return result

    def send_text(self, text: str) -> bool:
        return True

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def make_item(url="https://example.com/1", title="Тест", summary="Краткое"):
    return NewsItem(
        source="Test",
        url=url,
        title=title,
        summary=summary,
        published="2024-01-01",
        published_parsed=None,
    )


def _cleanup_db(path: str) -> None:
    """Windows-safe cleanup: закрывает соединение и удаляет DB + WAL."""
    import time
    # Пробуем удалить с задержкой (WAL checkpoint на закрытии)
    for attempt in range(3):
        try:
            Path(path).unlink(missing_ok=True)
            Path(path + "-wal").unlink(missing_ok=True)
            Path(path + "-shm").unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.1)
            else:
                # последняя попытка — игнорируем
                pass


def fresh_db():
    """Возвращает путь к временной БД и cleanup-функцию."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    Path(path).unlink()  # SQLite сам создаст
    return path, lambda: _cleanup_db(path)


# ─────────────────────────────────────────────────────────────────
# Тесты: Storage — update_llm_result / update_llm_results_batch
# ─────────────────────────────────────────────────────────────────

class TestStorageLLM:
    def test_update_llm_result_writes_fields(self):
        path, cleanup = fresh_db()
        try:
            s = Storage(path)

            # сохраняем item без llm-полей
            item = make_item(url="https://e.com/t1", title="Н", summary="С")
            s.save(item)
            s.set_state(["https://e.com/t1"], "pending")

            # обновляем
            ok = s.update_llm_result(
                url="https://e.com/t1",
                score=8,
                summary="Выжимка LLM",
                reason="тест",
            )
            assert ok is True

            # читаем обратно
            items = s.get_all_by_state("pending")
            assert len(items) == 1
            assert items[0].llm_score == 8
            assert items[0].llm_summary == "Выжимка LLM"
            assert items[0].llm_reason == "тест"
            s.close()
        finally:
            cleanup()

    def test_update_llm_results_batch(self):
        path, cleanup = fresh_db()
        try:
            s = Storage(path)

            for i in range(3):
                item = make_item(url=f"https://e.com/b{i}", title=f"Т{i}", summary=f"С{i}")
                s.save(item)

            records = [
                {"url": "https://e.com/b0", "score": 9, "summary": "S0", "reason": "R0"},
                {"url": "https://e.com/b1", "score": 6, "summary": "S1", "reason": "R1"},
                {"url": "https://e.com/b2", "score": 3, "summary": "S2", "reason": "R2"},
            ]
            n = s.update_llm_results_batch(records)
            assert n == 3

            items = s.get_all_by_state("new")
            items_map = {i.url: i for i in items}

            assert items_map["https://e.com/b0"].llm_score == 9
            assert items_map["https://e.com/b1"].llm_score == 6
            assert items_map["https://e.com/b2"].llm_score == 3
            s.close()
        finally:
            cleanup()

    def test_migration_v3_to_v4(self):
        """Проверяем: v3-мигрированная БД получает v4-колонки без потери данных."""
        path, cleanup = fresh_db()
        try:
            # вручную создаём схему v3 (до llm-колонок)
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA user_version = 3")
            conn.execute("""
                CREATE TABLE items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    published TEXT,
                    fetched_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'new',
                    state_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE TABLE seen_titles (id INTEGER PRIMARY KEY)")
            conn.execute("""
                INSERT INTO items (source, url, title, summary, published, fetched_at, state, state_at)
                VALUES ('OldSrc', 'https://old.com/1', 'OldTitle', 'OldSummary', '2024-01-01', '2024-01-01T00:00:00', 'delivered', '2024-01-01T00:00:00')
            """)
            conn.commit()
            conn.close()

            # открываем через Storage — должен мигрировать до последней (v6)
            s = Storage(path)

            # проверяем версию
            conn2 = sqlite3.connect(path)
            ver = conn2.execute("PRAGMA user_version").fetchone()[0]
            conn2.close()
            assert ver == 8, f"ожидалась v8, получена v{ver}"

            # старые данные на месте
            items = s.get_all_by_state("delivered")
            assert len(items) == 1
            assert items[0].title == "OldTitle"
            assert items[0].llm_score is None
            assert items[0].llm_summary is None
            s.close()
        finally:
            cleanup()

    def test_purge_keeps_deferred(self):
        """
        К5: deferred НЕ удаляется purge (дайджест ещё не реализован,
        данные сохраняем). delivered/skipped/failed — удаляются.
        """
        path, cleanup = fresh_db()
        try:
            s = Storage(path)

            # deferred item — должен ОСТАТЬСЯ
            item_def = make_item(url="https://e.com/deferred", title="D", summary="S")
            s.save(item_def)
            s.set_state(["https://e.com/deferred"], "deferred")

            # delivered item — должен УДАЛИТЬСЯ
            item_del = make_item(url="https://e.com/delivered", title="X", summary="Y")
            s.save(item_del)
            s.set_state(["https://e.com/delivered"], "delivered")

            # skipped item — должен УДАЛИТЬСЯ
            item_skip = make_item(url="https://e.com/skipped", title="Z", summary="W")
            s.save(item_skip)
            s.set_state(["https://e.com/skipped"], "skipped")

            # failed item — должен УДАЛИТЬСЯ
            item_fail = make_item(url="https://e.com/failed", title="F", summary="G")
            s.save(item_fail)
            s.set_state(["https://e.com/failed"], "failed")

            deleted = s.purge_items(retention_days=0)
            assert deleted == 3, f"ожидалось 3 удалённых (delivered/skipped/failed), получено {deleted}"

            # deferred — ОСТАЛСЯ
            remaining = s.get_all_by_state("deferred")
            assert len(remaining) == 1, f"deferred должен остаться, осталось {len(remaining)}"
            assert remaining[0].url == "https://e.com/deferred"

            # остальные — удалены
            assert len(s.get_all_by_state("delivered")) == 0
            assert len(s.get_all_by_state("skipped")) == 0
            assert len(s.get_all_by_state("failed")) == 0
            s.close()
        finally:
            cleanup()


# ─────────────────────────────────────────────────────────────────
# Тесты: Pipeline — маршрутизация через LLM
# ─────────────────────────────────────────────────────────────────

class TestPipelineLLMRouting:
    def _make_config(self, llm_enabled=True, api_key=""):
        return {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": llm_enabled,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "api_key": api_key,
                "base_url": "https://api.deepseek.com",
                "temperature": 0,
                "timeout_sec": 60,
                "max_retries": 3,
                "push_threshold": 7,
                "digest_threshold": 5,
                "system_prompt": "ты фильтр",
            },
        }

    def _run_pipeline(self, items, fake_filter=None, mock_delivery=None):
        """Создаёт pipeline, подменяет llm_filter и delivery, запускает _process_batch."""
        cfg = self._make_config(llm_enabled=True, api_key="fake-key")
        path, cleanup = fresh_db()
        cfg["database"]["path"] = path

        p = NewsPipeline.__new__(NewsPipeline)
        p.config = cfg
        p.sources = []
        p.storage = Storage(path)
        p.delivery = mock_delivery or MockDelivery()
        p.llm_filter = fake_filter
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._feedback_context = None

        # Items должны быть СНАЧАЛА сохранены в БД (как в реальном pipeline),
        # иначе set_state (UPDATE) не найдёт строк.
        for item in items:
            p.storage.save(item)

        # вызываем _process_batch
        p._process_batch(items)

        # закрыть соединение до cleanup (важно для Windows/WAL)
        p.storage.close()

        return p, cleanup

    def test_score_8_goes_to_delivery(self):
        """Score 8 → route=push → delivered."""
        items = [make_item(url="https://push.com/8", title="Push", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"push.com/8": 8})
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == ["https://push.com/8"]
        # не skipped, не deferred
        pending = p.storage.get_all_by_state("pending")
        assert len(pending) == 0

        # llm_score сохранён
        delivered = p.storage.get_all_by_state("delivered")
        assert len(delivered) == 1
        assert delivered[0].llm_score == 8
        cleanup()

    def test_score_6_goes_to_deferred(self):
        """Score 6 → route=digest → deferred (НЕ delivered)."""
        items = [make_item(url="https://dig.com/6", title="Digest", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"dig.com/6": 6})
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == []
        deferred = p.storage.get_all_by_state("deferred")
        assert len(deferred) == 1
        assert deferred[0].llm_score == 6
        cleanup()

    def test_score_3_goes_to_skipped(self):
        """Score 3 → route=skip → skipped (НЕ delivered)."""
        items = [make_item(url="https://skip.com/3", title="Skip", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"skip.com/3": 3})
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == []
        skipped = p.storage.get_all_by_state("skipped")
        assert len(skipped) == 1
        cleanup()

    def test_fallback_ok_false_delivers_without_filter(self):
        """Fallback (ok=False) → доставляется непрофильтрованным."""
        items = [make_item(url="https://fall.com", title="Fallback", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"fall.com": -1})  # -1 = ok=False
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == ["https://fall.com"]
        delivered = p.storage.get_all_by_state("delivered")
        assert len(delivered) == 1
        cleanup()

    def test_no_llm_filter_delivers_all(self):
        """Без llm_filter — всё доставляется (fallback)."""
        items = [
            make_item(url="https://a.com/1", title="A", summary="S"),
            make_item(url="https://a.com/2", title="B", summary="S"),
        ]
        mock_del = MockDelivery()
        p, cleanup = self._run_pipeline(items, fake_filter=None, mock_delivery=mock_del)

        assert len(mock_del.sent) == 2
        cleanup()

    def test_mixed_routes(self):
        """Батч из 3 items: push + digest + skip."""
        items = [
            make_item(url="https://x.com/push", title="Push", summary="S"),
            make_item(url="https://x.com/dig", title="Digest", summary="S"),
            make_item(url="https://x.com/skip", title="Skip", summary="S"),
        ]
        score_map = {
            "push": 8,
            "dig": 6,
            "skip": 3,
        }
        mock_del = MockDelivery()
        fake = FakeLLMFilter(score_map)
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == ["https://x.com/push"]
        assert p.storage.get_all_by_state("delivered")[-1].llm_score == 8
        assert len(p.storage.get_all_by_state("deferred")) == 1
        assert len(p.storage.get_all_by_state("skipped")) == 1
        cleanup()

    def test_delivery_partial_failure(self):
        """При частичном сбое доставки: успешные → delivered, остальные → pending."""
        items = [
            make_item(url="https://ok.com", title="OK", summary="S"),
            make_item(url="https://fail.com", title="Fail", summary="S"),
        ]
        mock_del = MockDelivery(fail_on_urls=["https://fail.com"])
        fake = FakeLLMFilter({})  # fallback на всё → оба должны были идти
        # переключим на fake с ok=False, тогда оба → delivered если бы не mock_del
        # Лучше: fake с ok=False на оба, но delivery отказывает на один
        fake2 = FakeLLMFilter({"ok.com": 9, "fail.com": 9})  # оба push
        p, cleanup = self._run_pipeline(items, fake_filter=fake2, mock_delivery=mock_del)

        # delivered только тот, что не в fail_on_urls
        delivered = p.storage.get_all_by_state("delivered")
        assert [i.url for i in delivered] == ["https://ok.com"]
        pending = p.storage.get_all_by_state("pending")
        assert [i.url for i in pending] == ["https://fail.com"]
        cleanup()

    def test_retry_uses_existing_llm_score(self):
        """При retry: если llm_score уже есть — НЕ переоценивать."""
        path, cleanup = fresh_db()
        s = Storage(path)

        # saved item с уже известным llm_score
        item = make_item(url="https://retry.com", title="Retry", summary="S")
        s.save(item)
        s.set_state(["https://retry.com"], "pending")
        s.update_llm_result(
            url="https://retry.com",
            score=9,
            summary="Сохранённая выжимка",
            reason="reason",
        )

        # получаем через get_stuck_pending
        stuck = s.get_stuck_pending(max_age_seconds=0)
        assert len(stuck) == 1
        assert stuck[0].llm_score == 9
        assert stuck[0].llm_summary == "Сохранённая выжимка"

        # fake-фильтр должен НЕ быть вызван
        class CountingFilter:
            calls = 0
            push_threshold = 7
            digest_threshold = 5

            def evaluate(self, item):
                CountingFilter.calls += 1
                return FilterResult(ok=True, score=10, summary="NEW", reason="new")

        p = NewsPipeline.__new__(NewsPipeline)
        p.config = self._make_config()
        p.sources = []
        p.storage = s
        p.delivery = MockDelivery()
        p.llm_filter = CountingFilter()
        p._max_per_cycle = 150
        p._llm_calls_left = 150

        p._process_batch(stuck)

        assert CountingFilter.calls == 0, "Не должен вызывать evaluate при существующем llm_score"
        delivered = s.get_all_by_state("delivered")
        assert len(delivered) == 1
        # выжимка из БД, не из нового evaluate
        assert delivered[0].llm_summary == "Сохранённая выжимка"
        s.close()
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Тесты: Delivery использует llm_summary
# ─────────────────────────────────────────────────────────────────

class TestDeliveryLLMSummary:
    def test_telegram_uses_llm_summary(self):
        from newsbot.delivery.telegram import TelegramDelivery
        from unittest import mock

        item = make_item()
        item.llm_summary = "LLM-выжимка новости"

        with mock.patch.object(TelegramDelivery, "_api_call") as mock_call:
            mock_call.return_value = {"result": {"message_id": 1}}
            td = TelegramDelivery.__new__(TelegramDelivery)
            td.token = "x"
            td.chat_id = "1"
            td._base_url = "https://api.telegram.org/botx"
            td._http = mock.MagicMock()

            td.send_batch([item])

            # тело сообщения содержит llm_summary
            call_args = mock_call.call_args
            payload = call_args[1].get("json") or call_args[0][1]
            assert "LLM-выжимка" in payload["text"]

    def test_telegram_falls_back_to_raw_summary(self):
        from newsbot.delivery.telegram import TelegramDelivery
        from unittest import mock

        item = make_item(summary="Сырой summary")
        item.llm_summary = None  # нет LLM-выжимки

        with mock.patch.object(TelegramDelivery, "_api_call") as mock_call:
            mock_call.return_value = {"result": {"message_id": 1}}
            td = TelegramDelivery.__new__(TelegramDelivery)
            td.token = "x"
            td.chat_id = "1"
            td._base_url = "https://api.telegram.org/botx"
            td._http = mock.MagicMock()

            td.send_batch([item])

            call_args = mock_call.call_args
            payload = call_args[1].get("json") or call_args[0][1]
            assert "Сырой summary" in payload["text"]

    def test_console_uses_llm_summary(self, capsys):
        from newsbot.delivery.console import ConsoleDelivery

        item = make_item(summary="Сырой текст")
        item.llm_summary = "LLM-выжимка"

        deliv = ConsoleDelivery()
        deliv.send_batch([item])

        out = capsys.readouterr().out
        assert "LLM-выжимка" in out
        assert "Сырой текст" not in out  # приоритет — llm_summary


# ─────────────────────────────────────────────────────────────────
# Тесты: баланс-алерт
# ─────────────────────────────────────────────────────────────────

class _MockDeliveryWithSendText:
    """Mock-доставка с отслеживанием send_text."""

    def __init__(self):
        self.sent_texts: List[str] = []
        self.send_batch_urls: List[str] = []

    def send_batch(self, items: List[NewsItem]) -> List[str]:
        urls = [i.url for i in items]
        self.send_batch_urls.extend(urls)
        return urls

    def send_text(self, text: str) -> bool:
        self.sent_texts.append(text)
        return True

    def close(self) -> None:
        pass


class _MockFilterWithBalance:
    """Fake-фильтр с подконтрольным балансом."""

    def __init__(self, balance: BalanceInfo):
        self._balance = balance
        self.push_threshold = 7
        self.digest_threshold = 5
        self.enabled = True

    def evaluate(self, item: NewsItem) -> FilterResult:
        return FilterResult(ok=True, score=8, summary="ok", reason="ok")

    def get_balance(self) -> BalanceInfo:
        return self._balance

    @property
    def client(self):
        class _DummyClient:
            def close(self):
                pass
        return _DummyClient()


class TestBalanceAlert:
    def _make_config(self):
        return {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": True,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "api_key": "fake-key",
                "base_url": "https://api.deepseek.com",
                "temperature": 0,
                "timeout_sec": 60,
                "max_retries": 3,
                "push_threshold": 7,
                "digest_threshold": 5,
                "system_prompt": "ты фильтр",
                "balance": {
                    "enabled": True,
                    "threshold": 2.0,
                    "check_interval_hours": 24,
                },
                "max_per_cycle": 150,
            },
        }

    def test_migration_v4_to_v5_meta_table(self):
        """Таблица meta создаётся при миграции v4→v5."""
        path, cleanup = fresh_db()
        try:
            # вручную создаём v4 (без meta)
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA user_version = 4")
            conn.execute("""
                CREATE TABLE items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    published TEXT,
                    fetched_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'new',
                    state_at TEXT NOT NULL,
                    llm_score INTEGER,
                    llm_summary TEXT,
                    llm_reason TEXT,
                    filtered_at TEXT
                )
            """)
            conn.execute("CREATE TABLE seen_titles (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()

            # открываем через Storage — должен мигрировать до последней (v6)
            s = Storage(path)

            # meta-таблица должна существовать
            conn2 = sqlite3.connect(path)
            tables = [r[0] for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn2.close()
            assert "meta" in tables, f"meta не найдена, таблицы: {tables}"

            ver = s.get_meta("__test_key__")  # ещё не должно быть
            assert ver is None

            s.set_meta("__test_key__", "test_value")
            assert s.get_meta("__test_key__") == "test_value"

            # user_version = 6
            conn3 = sqlite3.connect(path)
            v = conn3.execute("PRAGMA user_version").fetchone()[0]
            conn3.close()
            assert v == 8, f"ожидалась v8, получена v{v}"

            s.close()
        finally:
            cleanup()

    def test_meta_get_set(self):
        """get_meta/set_meta работают корректно."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            assert s.get_meta("nonexistent") is None
            assert s.get_meta("nonexistent", "default") == "default"

            s.set_meta("key1", "value1")
            assert s.get_meta("key1") == "value1"

            s.set_meta("key1", "value2")  # обновление
            assert s.get_meta("key1") == "value2"

            s.set_meta("key2", "val2")
            assert s.get_meta("key1") == "value2"
            assert s.get_meta("key2") == "val2"
            s.close()
        finally:
            cleanup()

    def test_low_balance_triggers_alert_once(self):
        """Баланс ниже порога → алерт отправлен один раз."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            mock_del = _MockDeliveryWithSendText()

            # подконтрольный фильтр: баланс 1.5 CNY (< threshold=2.0)
            mock_filter = _MockFilterWithBalance(
                BalanceInfo(is_available=True, currency="CNY", total_balance=1.5)
            )

            cfg = self._make_config()
            cfg["database"]["path"] = path
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = mock_filter
            p._max_per_cycle = 150
            p._llm_calls_left = 150

            # первый вызов: алерт должен уйти
            p._maybe_check_balance()

            assert len(mock_del.sent_texts) == 1
            assert "1.5" in mock_del.sent_texts[0]
            assert "CNY" in mock_del.sent_texts[0]
            assert "порог 2.0" in mock_del.sent_texts[0]
            assert "platform.deepseek.com" in mock_del.sent_texts[0]

            # meta-ключи установлены
            assert s.get_meta("llm_balance_last_check") is not None
            assert s.get_meta("llm_balance_last_alert") is not None

            s.close()
        finally:
            cleanup()

    def test_balance_above_threshold_no_alert(self):
        """Баланс выше порога → алерта нет."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            mock_del = _MockDeliveryWithSendText()

            # баланс 5.0 CNY (>= threshold=2.0)
            mock_filter = _MockFilterWithBalance(
                BalanceInfo(is_available=True, currency="CNY", total_balance=5.0)
            )

            cfg = self._make_config()
            cfg["database"]["path"] = path
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = mock_filter
            p._max_per_cycle = 150
            p._llm_calls_left = 150

            p._maybe_check_balance()

            assert mock_del.sent_texts == []
            s.close()
        finally:
            cleanup()

    def test_get_balance_error_no_crash(self):
        """Ошибка get_balance() — не падает."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            mock_del = _MockDeliveryWithSendText()

            class _BrokenFilter:
                push_threshold = 7
                digest_threshold = 5
                enabled = True

                def evaluate(self, item):
                    return FilterResult(ok=True, score=8, summary="ok", reason="ok")

                def get_balance(self):
                    raise RuntimeError("сеть недоступна")

                @property
                def client(self):
                    class _DummyClient:
                        def close(self):
                            pass
                    return _DummyClient()

            cfg = self._make_config()
            cfg["database"]["path"] = path
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = _BrokenFilter()
            p._max_per_cycle = 150
            p._llm_calls_left = 150

            # не падает
            p._maybe_check_balance()

            assert mock_del.sent_texts == []
            s.close()
        finally:
            cleanup()

    def test_cooldown_prevents_repeated_alerts(self):
        """Повторный вызов в пределах интервала → алерта нет (cooldown)."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            mock_del = _MockDeliveryWithSendText()

            # баланс 1.5 CNY (< threshold)
            mock_filter = _MockFilterWithBalance(
                BalanceInfo(is_available=True, currency="CNY", total_balance=1.5)
            )

            cfg = self._make_config()
            cfg["database"]["path"] = path
            cfg["llm"]["balance"]["check_interval_hours"] = 24

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = mock_filter
            p._max_per_cycle = 150
            p._llm_calls_left = 150

            # первый вызов: алерт отправлен
            p._maybe_check_balance()
            assert len(mock_del.sent_texts) == 1

            # имитируем что last_check уже недавно
            from datetime import datetime, timezone
            recent = (datetime.now(timezone.utc)).isoformat()
            s.set_meta("llm_balance_last_check", recent)
            s.set_meta("llm_balance_last_alert", recent)

            # второй вызов: не должен отправлять алерт (cooldown)
            p._maybe_check_balance()
            assert len(mock_del.sent_texts) == 1, "cooldown не сработал — отправлен лишний алерт"

            s.close()
        finally:
            cleanup()

    def test_cooldown_works_when_last_check_expired_but_last_alert_fresh(self):
        """
        Cooldown работает независимо от last_check.
        Сценарий: last_check просрочен → проверка идёт, но last_alert свежий →
        повторного алерта нет (cooldown по last_alert, а не только по last_check).
        """
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            mock_del = _MockDeliveryWithSendText()

            mock_filter = _MockFilterWithBalance(
                BalanceInfo(is_available=True, currency="CNY", total_balance=1.5)
            )

            cfg = self._make_config()
            cfg["database"]["path"] = path
            cfg["llm"]["balance"]["check_interval_hours"] = 24

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = mock_filter
            p._max_per_cycle = 150
            p._llm_calls_left = 150

            # last_check=None → «не проверяли» → проверка идёт
            # last_alert=recent → cooldown блокирует алерт
            from datetime import datetime, timedelta, timezone
            recent = (datetime.now(timezone.utc)).isoformat()
            very_old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
            s.set_meta("llm_balance_last_check", very_old)  # просрочен → проверка идёт
            s.set_meta("llm_balance_last_alert", recent)   # свежий → cooldown блокирует

            p._maybe_check_balance()

            # проверка прошла (last_check просрочен), но cooldown по last_alert → нет алерта
            assert mock_del.sent_texts == [], \
                f"cooldown по last_alert не сработал: {mock_del.sent_texts}"

            s.close()
        finally:
            cleanup()


# ─────────────────────────────────────────────────────────────────
# Main runner (standalone)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # storage
        (lambda: TestStorageLLM().test_update_llm_result_writes_fields(), "update_llm_result_writes_fields"),
        (lambda: TestStorageLLM().test_update_llm_results_batch(), "update_llm_results_batch"),
        (lambda: TestStorageLLM().test_migration_v3_to_v4(), "migration_v3_to_v4"),
        (lambda: TestStorageLLM().test_purge_keeps_deferred(), "purge_keeps_deferred"),
        # routing
        (lambda: TestPipelineLLMRouting().test_score_8_goes_to_delivery(), "score_8_goes_to_delivery"),
        (lambda: TestPipelineLLMRouting().test_score_6_goes_to_deferred(), "score_6_goes_to_deferred"),
        (lambda: TestPipelineLLMRouting().test_score_3_goes_to_skipped(), "score_3_goes_to_skipped"),
        (lambda: TestPipelineLLMRouting().test_fallback_ok_false_delivers_without_filter(), "fallback_ok_false_delivers_without_filter"),
        (lambda: TestPipelineLLMRouting().test_no_llm_filter_delivers_all(), "no_llm_filter_delivers_all"),
        (lambda: TestPipelineLLMRouting().test_mixed_routes(), "mixed_routes"),
        (lambda: TestPipelineLLMRouting().test_delivery_partial_failure(), "delivery_partial_failure"),
        (lambda: TestPipelineLLMRouting().test_retry_uses_existing_llm_score(), "retry_uses_existing_llm_score"),
        # delivery
        (lambda: TestDeliveryLLMSummary().test_telegram_uses_llm_summary(), "telegram_uses_llm_summary"),
        (lambda: TestDeliveryLLMSummary().test_telegram_falls_back_to_raw_summary(), "telegram_falls_back_to_raw_summary"),
        (lambda: TestDeliveryLLMSummary().test_console_uses_llm_summary(), "console_uses_llm_summary"),
        # balance alert
        (lambda: TestBalanceAlert().test_low_balance_triggers_alert_once(), "low_balance_triggers_alert_once"),
        (lambda: TestBalanceAlert().test_balance_above_threshold_no_alert(), "balance_above_threshold_no_alert"),
        (lambda: TestBalanceAlert().test_get_balance_error_no_crash(), "get_balance_error_no_crash"),
        (lambda: TestBalanceAlert().test_cooldown_prevents_repeated_alerts(), "cooldown_prevents_repeated_alerts"),
        (lambda: TestBalanceAlert().test_cooldown_works_when_last_check_expired_but_last_alert_fresh(), "cooldown_last_check_expired_last_alert_fresh"),
        (lambda: TestBalanceAlert().test_migration_v4_to_v5_meta_table(), "migration_v4_to_v5_meta_table"),
        (lambda: TestBalanceAlert().test_meta_get_set(), "meta_get_set"),
    ]

    passed = 0
    failed = 0
    for fn, name in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    total = passed + failed
    print(f"Result: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} failed")
        sys.exit(1)
    else:
        print(" -- all green!")
        sys.exit(0)