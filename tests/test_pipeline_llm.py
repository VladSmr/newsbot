"""
Тесты интеграции LLM-фильтра в pipeline (этап 2b-2a).
Без сети — fake-фильтр, мок delivery.

Запуск: python -m pytest tests -q
        python tests/test_pipeline_llm.py  (standalone)
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
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
from newsbot.llm.client import BalanceInfo, UsageStats
from newsbot.storage import Storage
from newsbot.pipeline import NewsPipeline, build_cluster_key


# ─────────────────────────────────────────────────────────────────
# Fake-фильтр: подменяет LLMFilter.evaluate()
# ─────────────────────────────────────────────────────────────────

class FakeLLMFilter:
    """
    Подставляется в pipeline.llm_filter.
    rules: score → FilterResult (и optional urgent)
      8+  → ok=True, push (urgent=False по умолчанию)
      5-6 → ok=True, digest
      3-4 → ok=True, skip
      -1  → ok=False (fallback)
    urgent-override: score_map values могут быть dict {score, urgent} или int score
    """
    def __init__(self, score_map: dict):
        """
        score_map: {url_substring: score} или {url_substring: {"score": N, "urgent": bool}}.
        score=-1 → ok=False (fallback).
        """
        self.score_map = score_map
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = False  # новый атрибут

    def evaluate(self, item: NewsItem, feedback_context=None) -> FilterResult:
        for pattern, score in self.score_map.items():
            if pattern in item.url or pattern in item.title:
                if isinstance(score, dict):
                    s = score.get("score", -1)
                    urgent = bool(score.get("urgent", False))
                else:
                    s = score
                    urgent = False
                if s == -1:
                    return FilterResult(ok=False)
                return FilterResult(
                    ok=True,
                    score=s,
                    summary=f"LLM-выжимка для: {item.title[:40]}",
                    reason=f"тест score={s}",
                    urgent=urgent,
                )
        # по умолчанию — fallback
        return FilterResult(ok=False)

    def evaluate_batch(self, items, feedback_context=None):
        """Batch API: делегирует в evaluate() для каждого item."""
        return [self.evaluate(item, feedback_context) for item in items]


class MockDelivery:
    """Мок доставки: принимает всё, возвращает (successful_urls, [])."""
    def __init__(self, fail_on_urls: List[str] = None):
        self.sent: List[str] = []
        self.fail_on_urls = set(fail_on_urls or [])

    def send_batch(self, items: List[NewsItem]) -> tuple:
        result = []
        for item in items:
            if item.url in self.fail_on_urls:
                continue
            result.append(item.url)
            self.sent.append(item.url)
        return (result, [])

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
            assert ver == Storage.SCHEMA_VERSION, f"ожидалась v{Storage.SCHEMA_VERSION}, получена v{ver}"

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
                "model": "deepseek-flash",
                "api_key": api_key,
                "base_url": "https://api.deepseek.com",
                "temperature": 0,
                "timeout_sec": 60,
                "max_retries": 3,
                "push_threshold": 7,
                "digest_threshold": 5,
                "push_urgent_only": False,
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
            # Присваиваем cluster_key, чтобы избежать коллизий при score propagation.
            # Без cluster_key rebuild_cluster_key("") генерирует один nostem-hash для
            # всех items без старого ключа → все три попадают в один кластер и все
            # получают max(score)=8 → все push вместо push+dig+skip.
            if not getattr(item, "cluster_key", None):
                item.cluster_key = build_cluster_key(
                    location="",
                    stem=None,
                    published=item.published,
                    title=item.title,
                )
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
            push_urgent_only = False

            def evaluate(self, item):
                CountingFilter.calls += 1
                return FilterResult(ok=True, score=10, summary="NEW", reason="new")

            def evaluate_batch(self, items, feedback_context=None):
                for item in items:
                    CountingFilter.calls += 1
                return [FilterResult(ok=True, score=10, summary="NEW", reason="new")
                        for _ in items]

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
# Тесты: маршрутизация с флагом urgent (push_urgent_only=True)
# ─────────────────────────────────────────────────────────────────

class TestPipelineUrgentRouting:
    """
    Тесты маршрутизации с учётом флага urgent и параметра push_urgent_only.
    FakeLLMFilter возвращает dict {score, urgent} для точного контроля.
    """

    def _run_pipeline(self, items, fake_filter=None, mock_delivery=None):
        """Аналогично _run_pipeline из TestPipelineLLMRouting."""
        cfg = {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                "api_key": "fake-key", "base_url": "https://api.deepseek.com",
                "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                "push_threshold": 7, "digest_threshold": 5,
                "push_urgent_only": False,  # управляется через filter.push_urgent_only
                "system_prompt": "ты фильтр",
            },
        }
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

        for item in items:
            p.storage.save(item)

        p._process_batch(items)
        p.storage.close()

        return p, cleanup

    def test_urgent_high_score_delivered(self):
        """urgent=True + score>=7 → delivered (push)."""
        items = [make_item(url="https://urg.com/9", title="Urgent", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"urg.com/9": {"score": 9, "urgent": True}})
        fake.push_urgent_only = True
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == ["https://urg.com/9"]
        delivered = p.storage.get_all_by_state("delivered")
        assert len(delivered) == 1
        cleanup()

    def test_not_urgent_high_score_not_delivered(self):
        """Не-urgent + score>=7 + push_urgent_only=True → deferred (digest), НЕ delivered."""
        items = [make_item(url="https://nonurg.com/9", title="NotUrgent", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"nonurg.com/9": {"score": 9, "urgent": False}})
        fake.push_urgent_only = True
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        # не delivered (не push)
        assert mock_del.sent == []
        # → deferred (digest)
        deferred = p.storage.get_all_by_state("deferred")
        assert len(deferred) == 1
        cleanup()

    def test_urgent_low_score_deferred(self):
        """urgent=True + score<7 → deferred (digest), НЕ delivered."""
        items = [make_item(url="https://urglow.com/6", title="UrgLow", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"urglow.com/6": {"score": 6, "urgent": True}})
        fake.push_urgent_only = True
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == []
        deferred = p.storage.get_all_by_state("deferred")
        assert len(deferred) == 1
        cleanup()

    def test_not_urgent_low_score_skipped(self):
        """Не-urgent + score<5 → skipped."""
        items = [make_item(url="https://skip.com/3", title="Skip", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"skip.com/3": {"score": 3, "urgent": False}})
        fake.push_urgent_only = True
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        assert mock_del.sent == []
        skipped = p.storage.get_all_by_state("skipped")
        assert len(skipped) == 1
        cleanup()

    def test_legacy_mode_ignores_urgent(self):
        """push_urgent_only=False → score>=7 → push даже без urgent."""
        items = [make_item(url="https://legacy.com/8", title="Legacy", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"legacy.com/8": {"score": 8, "urgent": False}})
        fake.push_urgent_only = False  # legacy
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        # push (legacy: только score важен)
        assert mock_del.sent == ["https://legacy.com/8"]
        delivered = p.storage.get_all_by_state("delivered")
        assert len(delivered) == 1
        cleanup()


# ─────────────────────────────────────────────────────────────────
# C1-тесты: retry сохраняет llm_urgent (v10 persistence)
# ─────────────────────────────────────────────────────────────────

class TestRetryUrgentPersistence:
    """C1-фикс: retry восстанавливает urgent из llm_urgent (персистентное поле)."""

    def _make_config(self):
        return {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                "api_key": "fake-key", "base_url": "https://api.deepseek.com",
                "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                "push_threshold": 7, "digest_threshold": 5,
                "push_urgent_only": True,  # urgent-only mode
                "system_prompt": "ты фильтр",
            },
        }

    def test_retry_urgent_true_pushed(self):
        """C1: retry item с llm_score + llm_urgent=1 + push_urgent_only=True → push (delivered)."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            item = make_item(url="https://c1.com/1", title="C1", summary="S")
            s.save(item)
            s.set_state(["https://c1.com/1"], "pending")
            # Записываем llm_urgent=1 (INTEGER 1 в БД)
            s.update_llm_results_batch([{
                "url": "https://c1.com/1",
                "score": 9,
                "summary": "Срочная новость",
                "reason": "urgent=1",
                "urgent": True,
            }])

            # Имитируем retry: получаем item с llm_urgent из БД
            stuck = s.get_stuck_pending(max_age_seconds=0)
            assert len(stuck) == 1
            assert stuck[0].llm_score == 9
            assert stuck[0].llm_urgent is True, \
                f"llm_urgent должен быть True, получено: {stuck[0].llm_urgent}"

            mock_del = MockDelivery()
            # Fake-фильтр: проверяем что НЕ вызывается
            class NeverCalledFilter:
                calls = 0
                push_threshold = 7
                digest_threshold = 5
                push_urgent_only = True

                def evaluate(self, item):
                    NeverCalledFilter.calls += 1
                    return FilterResult(ok=True, score=10, summary="NEW", reason="new")

            def evaluate_batch(self, items, feedback_context=None):
                for item in items:
                    NeverCalledFilter.calls += 1
                return [FilterResult(ok=True, score=10, summary="NEW", reason="new")
                        for _ in items]

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = self._make_config()
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = NeverCalledFilter()
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._feedback_context = None

            p._process_batch(stuck)

            # LLM НЕ вызван (retry использует сохранённый score)
            assert NeverCalledFilter.calls == 0, "LLM не должен вызываться при retry"
            # Item доставлен (urgent=True, score>=7, push_urgent_only=True → push)
            assert mock_del.sent == ["https://c1.com/1"], \
                f"expected push, sent={mock_del.sent}"
            s.close()
        finally:
            cleanup()

    def test_retry_urgent_false_deferred(self):
        """C1: retry item с llm_score + llm_urgent=0 + push_urgent_only=True → deferred (digest)."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            item = make_item(url="https://c1.com/2", title="C1b", summary="S")
            s.save(item)
            s.set_state(["https://c1.com/2"], "pending")
            # llm_urgent=0 — НЕ срочный, даже с высоким score
            s.update_llm_results_batch([{
                "url": "https://c1.com/2",
                "score": 8,
                "summary": "Не срочная",
                "reason": "urgent=0",
                "urgent": False,
            }])

            stuck = s.get_stuck_pending(max_age_seconds=0)
            assert len(stuck) == 1
            assert stuck[0].llm_urgent is False, \
                f"llm_urgent должен быть False, получено: {stuck[0].llm_urgent}"

            mock_del = MockDelivery()

            class NeverCalledFilter:
                calls = 0
                push_threshold = 7
                digest_threshold = 5
                push_urgent_only = True

                def evaluate(self, item):
                    NeverCalledFilter.calls += 1
                    return FilterResult(ok=True, score=10, summary="NEW", reason="new")

                def evaluate_batch(self, items, feedback_context=None):
                    for item in items:
                        NeverCalledFilter.calls += 1
                    return [FilterResult(ok=True, score=10, summary="NEW", reason="new")
                            for _ in items]

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = self._make_config()
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = NeverCalledFilter()
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._feedback_context = None

            p._process_batch(stuck)

            # LLM НЕ вызван
            assert NeverCalledFilter.calls == 0
            # Не доставлен (urgent=False → digest/deferred)
            assert mock_del.sent == [], f"expected deferred, sent={mock_del.sent}"
            deferred = s.get_all_by_state("deferred")
            assert len(deferred) == 1, f"expected deferred state, got: {[i.state for i in s.get_all_by_state('pending')]}"
            s.close()
        finally:
            cleanup()

    def test_retry_legacy_no_llm_urgent(self):
        """C1: legacy-строка без llm_urgent (NULL) → urgent=False на retry (безопасный дефолт)."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            item = make_item(url="https://c1.com/3", title="C1c", summary="S")
            s.save(item)
            s.set_state(["https://c1.com/3"], "pending")
            # Записываем БЕЗ ключа urgent (legacy-вызов update_llm_results_batch без urgent)
            s.update_llm_results_batch([{
                "url": "https://c1.com/3",
                "score": 9,
                "summary": "Без llm_urgent",
                "reason": "legacy",
                # urgent отсутствует — старый вызов
            }])

            # Прямое чтение из БД: llm_urgent должен быть NULL
            # NOTE: НЕ закрываем conn — get_stuck_pending ниже работает с тем же соединением
            conn = s._connect()
            row = conn.execute("SELECT llm_urgent FROM items WHERE url=?", ("https://c1.com/3",)).fetchone()
            assert row[0] is None, "llm_urgent должен быть NULL при отсутствии ключа"

            stuck = s.get_stuck_pending(max_age_seconds=0)
            assert len(stuck) == 1
            # llm_urgent=None → False на retry
            assert stuck[0].llm_urgent is None, \
                f"llm_urgent должен быть None (не прочитан из БД), получено: {stuck[0].llm_urgent}"

            mock_del = MockDelivery()

            class NeverCalledFilter:
                calls = 0
                push_threshold = 7
                digest_threshold = 5
                push_urgent_only = True

                def evaluate(self, item):
                    NeverCalledFilter.calls += 1
                    return FilterResult(ok=True, score=10, summary="NEW", reason="new")

                def evaluate_batch(self, items, feedback_context=None):
                    for item in items:
                        NeverCalledFilter.calls += 1
                    return [FilterResult(ok=True, score=10, summary="NEW", reason="new")
                            for _ in items]

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = self._make_config()
            p.sources = []
            p.storage = s
            p.delivery = mock_del
            p.llm_filter = NeverCalledFilter()
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._feedback_context = None

            p._process_batch(stuck)

            # LLM НЕ вызван
            assert NeverCalledFilter.calls == 0
            # Не доставлен: llm_urgent=None → False на retry → deferred
            assert mock_del.sent == [], \
                f"legacy-строка с llm_urgent=NULL → deferred, sent={mock_del.sent}"
            deferred = s.get_all_by_state("deferred")
            assert len(deferred) == 1
            s.close()
        finally:
            cleanup()


# ─────────────────────────────────────────────────────────────────
# M1-тест: warning при отсутствии ключа push_urgent_only
# ─────────────────────────────────────────────────────────────────

class TestM1PushUrgentOnlyWarning:
    """M1-фикс: отсутствие llm.push_urgent_only в конфиге → warning."""

    def test_missing_push_urgent_only_key_logs_warning(self, caplog):
        """Ключ push_urgent_only отсутствует → warning при инициализации фильтра."""
        import logging
        from newsbot.llm.filter import LLMFilter

        cfg = {
            "enabled": True,
            "model": "deepseek-flash",
            "push_threshold": 7,
            "digest_threshold": 5,
            # push_urgent_only ЯВНО отсутствует
        }
        with caplog.at_level(logging.WARNING, logger="newsbot.llm.filter"):
            flt = LLMFilter(cfg)
            assert flt.push_urgent_only is True, "дефолт остаётся True"

        # Warning залогирован
        assert any(
            "push_urgent_only" in r.message and "отсутствует" in r.message
            for r in caplog.records
        ), f"Expected warning about missing push_urgent_only. Records: {[r.message for r in caplog.records]}"

    def test_present_push_urgent_only_no_warning(self, caplog):
        """Ключ push_urgent_only присутствует → warning НЕ логируется."""
        import logging
        from newsbot.llm.filter import LLMFilter

        cfg = {
            "enabled": True,
            "model": "deepseek-flash",
            "push_threshold": 7,
            "digest_threshold": 5,
            "push_urgent_only": True,
        }
        with caplog.at_level(logging.WARNING, logger="newsbot.llm.filter"):
            flt = LLMFilter(cfg)

        warnings = [r for r in caplog.records if "push_urgent_only" in r.message and "отсутствует" in r.message]
        assert len(warnings) == 0, f"Не должно быть warning при наличии ключа: {warnings}"


# ─────────────────────────────────────────────────────────────────
# M2-тесты: fallback → deferred при push_urgent_only=True
# ─────────────────────────────────────────────────────────────────

class TestM2FallbackToDigest:
    """M2-фикс: ok=False (fallback) при push_urgent_only=True → deferred (digest)."""

    def _run_pipeline(self, items, fake_filter=None, mock_delivery=None, push_urgent_only=True):
        cfg = {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                "api_key": "fake-key", "base_url": "https://api.deepseek.com",
                "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                "push_threshold": 7, "digest_threshold": 5,
                "push_urgent_only": push_urgent_only,
                "system_prompt": "ты фильтр",
            },
        }
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

        for item in items:
            p.storage.save(item)

        p._process_batch(items)
        p.storage.close()

        return p, cleanup

    def test_fallback_urgent_only_deferred(self):
        """M2: ok=False + push_urgent_only=True → deferred (digest), НЕ delivered."""
        items = [make_item(url="https://m2.com/1", title="M2", summary="S")]
        mock_del = MockDelivery()
        # -1 → ok=False (fallback)
        fake = FakeLLMFilter({"m2.com/1": -1})
        fake.push_urgent_only = True
        p, cleanup = self._run_pipeline(items, fake_filter=fake, mock_delivery=mock_del)

        # НЕ доставлен (push_urgent_only=True, ok=False → deferred)
        assert mock_del.sent == [], f"expected no push, sent={mock_del.sent}"
        # → deferred
        deferred = p.storage.get_all_by_state("deferred")
        assert len(deferred) == 1, f"expected deferred, got: {[(i.url, i.state) for i in p.storage.get_all_by_state('pending')]}"
        cleanup()

    def test_fallback_legacy_mode_delivers(self):
        """M2: ok=False + push_urgent_only=False → delivered (legacy-поведение)."""
        items = [make_item(url="https://m2.com/2", title="M2b", summary="S")]
        mock_del = MockDelivery()
        fake = FakeLLMFilter({"m2.com/2": -1})
        fake.push_urgent_only = False
        p, cleanup = self._run_pipeline(
            items, fake_filter=fake, mock_delivery=mock_del, push_urgent_only=False,
        )

        # Fallback доставляет при push_urgent_only=False (legacy)
        assert mock_del.sent == ["https://m2.com/2"], \
            f"expected push (legacy), sent={mock_del.sent}"
        delivered = p.storage.get_all_by_state("delivered")
        assert len(delivered) == 1
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Digest-лимит: проверяем что в дайджест уходит ≤max_items
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# v10 post-check: defensive ADD COLUMN llm_urgent при user_version=10 без колонки
# ─────────────────────────────────────────────────────────────────

class TestV10PostCheck:
    """v10 post-check: при user_version=10 без колонки llm_urgent — добавляется идемпотентно."""

    def test_v10_user_version_without_llm_urgent_column_adds_it(self, tmp_path):
        """
        Edge case: БД с user_version=10, но без колонки llm_urgent.
        При открытии Storage колонка добавляется post-check'ом,
        SELECT по llm_urgent не падает.
        """
        import sqlite3
        db_path = tmp_path / "v10nocol.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version = 10")
        conn.execute("""
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                source TEXT, url TEXT UNIQUE, title TEXT, summary TEXT,
                published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
                llm_score INTEGER, llm_summary TEXT, llm_reason TEXT,
                filtered_at TEXT, retry_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE seen_titles (
                id INTEGER PRIMARY KEY, title_hash TEXT, source TEXT,
                fetched_at TEXT, expires_at TEXT, title_normalized TEXT
            )
        """)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()

        # Storage открывает БД — post-check добавляет llm_urgent
        s = Storage(db_path)

        conn2 = sqlite3.connect(str(db_path))
        conn2.row_factory = sqlite3.Row
        cols = {r["name"] for r in conn2.execute("PRAGMA table_info(items)").fetchall()}
        conn2.close()

        assert "llm_urgent" in cols, f"llm_urgent должна быть добавлена post-check: {cols}"

        # SELECT по llm_urgent не падает (проверяем indirectly: сохранение item)
        item = make_item(url="https://v10post.com/1", title="V10Post", summary="S")
        s.save(item)
        n = s.update_llm_results_batch([{
            "url": "https://v10post.com/1",
            "score": 9, "summary": "Test", "reason": "ok", "urgent": True,
        }])
        assert n == 1
        items = s.get_all_by_state("new")
        assert items[0].llm_urgent is True
        s.close()


# ─────────────────────────────────────────────────────────────────
# 2c-фикс: candidates remainder → retry_count инкремент + терминализация
# ─────────────────────────────────────────────────────────────────

class TestCandidatesRetryAccounting:
    """2c-фикс: сбой доставки кандидатов инкрементирует retry_count и терминализирует при лимите."""

    def test_candidates_remainder_returned(self):
        """_process_batch возвращает remainder кандидатов (не доставленные)."""
        from newsbot.pipeline import NewsPipeline
        from newsbot.storage import Storage

        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            cfg = {
                "sources": [],
                "database": {"path": path, "max_retries": 8},
                "telegram": {"token": "", "chat_id": ""},
                "llm": {
                    "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                    "api_key": "fake", "base_url": "https://x.com",
                    "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                    "push_threshold": 7, "digest_threshold": 5,
                    "push_urgent_only": False, "system_prompt": "",
                },
            }

            class MockFilter:
                push_threshold = 7
                digest_threshold = 5
                push_urgent_only = False

                def evaluate_batch(self, items, feedback_context=None):
                    from newsbot.llm.filter import FilterResult
                    return [FilterResult(ok=True, score=9, summary="S", reason="r")
                            for _ in items]

            class PartialDelivery:
                """Доставляет только первые 2 из 3 (new tuple contract)."""
                def send_batch(self, items):
                    return ([items[0].url, items[1].url], [])

                def send_text(self, text): return True
                def close(self): pass

            items = [make_item(url=f"https://cand.com/{i}", title=f"C{i}", summary="S")
                     for i in range(3)]
            for item in items:
                s.save(item)

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = PartialDelivery()
            p.llm_filter = MockFilter()
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._feedback_context = None

            # _process_batch возвращает remainder: item 2 не доставлен
            remainder = p._process_batch(items, budget_override=150)
            s.close()

            # remainder содержит не доставленные
            assert "https://cand.com/2" in remainder, \
                f"expected cand.com/2 in remainder, got {remainder}"
            # Доставленные не в remainder
            assert "https://cand.com/0" not in remainder
            assert "https://cand.com/1" not in remainder
        finally:
            cleanup()

    def test_candidates_remainder_triggers_increment_in_run_cycle(self):
        """
        Major #3 (p.3): budget-excess items возвращаются в state='new'.
        Budget-независимый TG-сбой → failed_urls → retry_count → 'failed'.
        Budget-cut items НЕ попадают в failed_urls (не дошли до delivery).

        Проверяем НОВОЕ корректное поведение (p.3):
        — Budget-cut → 'new' (retry в след.цикле, retry_count сохраняется)
        — TG-сбой (send_batch < to_deliver) → в failed_urls → retry_count
        """
        import unittest.mock as mock
        import sqlite3

        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            cfg = {
                "sources": [],
                "database": {"path": path, "max_retries": 3},
                "telegram": {"token": "", "chat_id": ""},
                "llm": {
                    "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                    "api_key": "fake", "base_url": "https://x.com",
                    "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                    "push_threshold": 7, "digest_threshold": 5,
                    "push_urgent_only": False, "system_prompt": "",
                },
            }

            class MockFilter:
                push_threshold = 7
                digest_threshold = 5
                push_urgent_only = False

                def evaluate_batch(self, items, feedback_context=None):
                    from newsbot.llm.filter import FilterResult
                    return [FilterResult(ok=True, score=9, summary="S", reason="r")
                            for _ in items]

            class PartialDelivery:
                """Доставляет только первые 2 из 3 (new tuple contract)."""
                def send_batch(self, items):
                    return ([items[0].url, items[1].url], [])

                def send_text(self, text): return True
                def close(self): pass

            # Создаём items в Storage
            items = [make_item(url=f"https://cand2.com/{i}", title=f"C{i}", summary="S")
                     for i in range(3)]
            for item in items:
                s.save(item)

            with mock.patch.object(s, "get_all_by_state", return_value=[]), \
                 mock.patch.object(s, "get_stuck_pending", return_value=[]), \
                 mock.patch.object(s, "recover_stuck_pending", return_value=0), \
                 mock.patch.object(s, "update_llm_results_batch"), \
                 mock.patch.object(s, "get_or_create_feedback_target_from_item", return_value=1), \
                 mock.patch.object(s, "get_meta", return_value=None), \
                 mock.patch.object(s, "set_meta"), \
                 mock.patch.object(s, "purge_items"), \
                 mock.patch.object(s, "purge_seen_titles", return_value=0):
                p = NewsPipeline.__new__(NewsPipeline)
                p.config = cfg
                p.sources = []
                p.storage = s
                p.delivery = PartialDelivery()
                p.llm_filter = MockFilter()
                p._max_per_cycle = 150
                p._llm_calls_left = 150
                p._max_retries = 3
                p._feedback_context = None
                p._is_paused = lambda: False

                # budget=2 → items[0,1] обработаны, items[2] срезан бюджетом
                # set_state НЕ мокаем — хотим реальные state-переходы в БД
                remainder = p._process_batch(items, budget_override=2)

                # p.3: items[2] срезан бюджетом → НЕ в failed_urls
                # (он не дошёл до delivery, не было TG-сбоя)
                assert "https://cand2.com/2" not in remainder, (
                    "Budget-cut item[2] НЕ должен быть в failed_urls (p.3 fix)"
                )

                # items[2] → state='new' (budget cut, retry в след.цикле)
                conn = sqlite3.connect(path)
                state = dict(conn.execute(
                    "SELECT url, state FROM items"
                ).fetchall())
                rc = dict(conn.execute(
                    "SELECT url, retry_count FROM items"
                ).fetchall())
                conn.close()

                # Delivered items — delivered
                assert state.get("https://cand2.com/0") == "delivered", \
                    f"item0 должен быть delivered, got {state.get('https://cand2.com/0')}"
                assert state.get("https://cand2.com/1") == "delivered", \
                    f"item1 должен быть delivered, got {state.get('https://cand2.com/1')}"
                # Budget-cut item → 'new' (p.3 fix: возвращается, не зависает pending)
                assert state.get("https://cand2.com/2") == "new", \
                    f"item2 должен быть в state='new' (budget cut), got {state.get('https://cand2.com/2')}"
                # retry_count сохранён (не сброшен при возврате в 'new')
                assert rc.get("https://cand2.com/2") == 0, \
                    f"item2 retry_count должен быть 0, got {rc.get('https://cand2.com/2')}"

            s.close()
        finally:
            cleanup()


# ─────────────────────────────────────────────────────────────────
# Digest-лимит: проверяем что в дайджест уходит ≤max_items
# ─────────────────────────────────────────────────────────────────

class TestDigestLimit:
    """m1: digest.max_items=30 → в дайджест уходит ≤30 новостей (топ по score)."""

    def test_deferred_top_by_score_capped(self):
        """40 deferred-items с разными score → в дайджест уходит ≤30 (топ-30)."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            # Создаём 40 deferred-items: score от 5 до 8
            for i in range(40):
                item = make_item(
                    url=f"https://dig.com/{i}",
                    title=f"Dig{i}",
                    summary=f"S{i}",
                )
                item.llm_score = 5 + (i % 4)  # score 5, 6, 7, 8
                s.save(item)
                s.set_state([f"https://dig.com/{i}"], "deferred")

            # get_deferred_items без лимита
            all_deferred = s.get_deferred_items()
            assert len(all_deferred) == 40

            # get_deferred_items с лимитом 30
            limited = s.get_deferred_items(limit=30)
            assert len(limited) == 30, f"expected 30, got {len(limited)}"

            # Первые 30 — топ по score DESC, fetched_at ASC
            for i in range(1, len(limited)):
                prev = limited[i - 1]
                cur = limited[i]
                # score DESC — если score равны, fetched_at ASC
                if prev.llm_score == cur.llm_score:
                    assert prev.fetched_at <= cur.fetched_at, \
                        f"При равных score fetched_at должен расти: {prev.fetched_at} vs {cur.fetched_at}"
                else:
                    assert prev.llm_score >= cur.llm_score, \
                        f"score должен убывать: {prev.llm_score} vs {cur.llm_score}"

            s.close()
        finally:
            cleanup()


# ─────────────────────────────────────────────────────────────────
# Storage: llm_urgent записывается и читается
# ─────────────────────────────────────────────────────────────────

class TestStorageUrgent:
    """v10: update_llm_results_batch с urgent → llm_urgent INTEGER 0/1."""

    def test_update_llm_results_batch_with_urgent(self):
        """urgent=True → llm_urgent=1; urgent=False → llm_urgent=0."""
        path, cleanup = fresh_db()
        try:
            s = Storage(path)
            for i in range(4):
                item = make_item(url=f"https://u.com/{i}", title=f"U{i}", summary=f"S{i}")
                s.save(item)

            records = [
                {"url": "https://u.com/0", "score": 8, "summary": "S0", "reason": "R0", "urgent": True},
                {"url": "https://u.com/1", "score": 7, "summary": "S1", "reason": "R1", "urgent": False},
                {"url": "https://u.com/2", "score": 6, "summary": "S2", "reason": "R2"},  # без urgent
                {"url": "https://u.com/3", "score": 5, "summary": "S3", "reason": "R3", "urgent": True},
            ]
            n = s.update_llm_results_batch(records)
            assert n == 4

            # Читаем обратно
            items = {i.url: i for i in s.get_all_by_state("new")}
            assert items["https://u.com/0"].llm_urgent is True
            assert items["https://u.com/1"].llm_urgent is False
            assert items["https://u.com/2"].llm_urgent is None  # без ключа → NULL
            assert items["https://u.com/3"].llm_urgent is True

            s.close()
        finally:
            cleanup()

    def test_migration_v9_adds_llm_urgent(self, tmp_path):
        """v9-БД получает колонку llm_urgent при миграции v9→v10."""
        import sqlite3
        db_path = tmp_path / "v9.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version = 9")
        conn.execute("""
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                source TEXT, url TEXT, title TEXT, summary TEXT,
                published TEXT, fetched_at TEXT, state TEXT, state_at TEXT,
                llm_score INTEGER, llm_summary TEXT, llm_reason TEXT,
                filtered_at TEXT, retry_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE seen_titles (
                id INTEGER PRIMARY KEY,
                title_hash TEXT, source TEXT, fetched_at TEXT,
                expires_at TEXT, title_normalized TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)
        """)
        conn.execute("""
            CREATE TABLE feedback_targets (
                id INTEGER PRIMARY KEY,
                url TEXT, source TEXT, title TEXT, summary TEXT,
                llm_score INTEGER, kind TEXT, chat_id TEXT,
                message_id INTEGER, created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY,
                target_id INTEGER, url TEXT, source TEXT, title TEXT,
                summary TEXT, llm_score INTEGER,
                verdict TEXT NOT NULL CHECK(verdict IN ('like','dislike')),
                chat_id TEXT, message_id INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Миграция
        s = Storage(db_path)
        assert sqlite3.connect(str(db_path)).execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION

        # Колонка llm_urgent добавлена
        conn2 = sqlite3.connect(str(db_path))
        conn2.row_factory = sqlite3.Row
        cols = {r["name"] for r in conn2.execute("PRAGMA table_info(items)").fetchall()}
        conn2.close()
        assert "llm_urgent" in cols, f"llm_urgent должна быть добавлена: {cols}"

        # INSERT + UPDATE с urgent
        item = make_item(url="https://v10.com/1", title="V10", summary="S")
        s.save(item)
        n = s.update_llm_results_batch([{
            "url": "https://v10.com/1",
            "score": 9,
            "summary": "Срочная",
            "reason": "test",
            "urgent": True,
        }])
        assert n == 1

        items = s.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].llm_urgent is True
        s.close()

    def test_fresh_db_has_llm_urgent(self, tmp_path):
        """Свежая БД (v10) сразу имеет колонку llm_urgent."""
        s = Storage(tmp_path / "fresh.db")
        conn = sqlite3.connect(str(s.db_path))
        conn.row_factory = sqlite3.Row
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        conn.close()
        assert "llm_urgent" in cols
        assert sqlite3.connect(str(s.db_path)).execute("PRAGMA user_version").fetchone()[0] == Storage.SCHEMA_VERSION
        s.close()


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

    def send_batch(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        urls = [i.url for i in items]
        self.send_batch_urls.extend(urls)
        return (urls, [])

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
        self.push_urgent_only = False
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
                "model": "deepseek-flash",
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

            # user_version = Storage.SCHEMA_VERSION
            conn3 = sqlite3.connect(path)
            v = conn3.execute("PRAGMA user_version").fetchone()[0]
            conn3.close()
            assert v == Storage.SCHEMA_VERSION, f"ожидалась v{Storage.SCHEMA_VERSION}, получена v{v}"

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
                push_urgent_only = False
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
# Тесты: UsageStats — накопление и логирование
# ─────────────────────────────────────────────────────────────────

class TestUsageStatsAccumulation:
    """UsageStats: накопление между батчами, сброс, логирование."""

    def test_accumulate_across_multiple_responses(self):
        """Несколько add_response() → все поля суммируются."""
        stats = UsageStats(model="deepseek-flash")

        # батч 1: два вызова
        stats.add_response({
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "total_tokens": 600,
                "prompt_cache_hit_tokens": 400,
                "prompt_cache_miss_tokens": 100,
            }
        })
        stats.add_response({
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": 150,
                "total_tokens": 850,
                "prompt_cache_hit_tokens": 600,
                "prompt_cache_miss_tokens": 100,
            }
        })

        assert stats.calls == 2
        assert stats.prompt_tokens == 1200
        assert stats.completion_tokens == 250
        assert stats.total_tokens == 1450
        assert stats.prompt_cache_hit_tokens == 1000
        assert stats.prompt_cache_miss_tokens == 200

    def test_accumulate_openai_cached_tokens_format(self):
        """Формат OpenAI: prompt_tokens_details.cached_tokens → корректные cached/miss."""
        stats = UsageStats(model="deepseek-flash")

        stats.add_response({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {
                    "cached_tokens": 800,
                },
            }
        })

        assert stats.calls == 1
        assert stats.prompt_tokens == 1000
        assert stats.prompt_cache_hit_tokens == 800
        assert stats.prompt_cache_miss_tokens == 200  # 1000 - 800

    def test_accumulate_deepseek_top_level_format(self):
        """Формат DeepSeek top-level: cached_hit + cached_miss → берём напрямую."""
        stats = UsageStats(model="deepseek-flash")

        stats.add_response({
            "usage": {
                "prompt_tokens": 1100,
                "completion_tokens": 200,
                "total_tokens": 1300,
                "prompt_cache_hit_tokens": 900,
                "prompt_cache_miss_tokens": 200,
            }
        })

        assert stats.calls == 1
        assert stats.prompt_cache_hit_tokens == 900
        assert stats.prompt_cache_miss_tokens == 200

    def test_reset_clears_counters_preserves_model(self):
        """reset() обнуляет всё кроме model."""
        stats = UsageStats(model="deepseek-flash")
        stats.calls = 5
        stats.prompt_tokens = 5000
        stats.prompt_cache_hit_tokens = 4000
        stats.prompt_cache_miss_tokens = 1000
        stats.completion_tokens = 1000
        stats.total_tokens = 6000

        stats.reset()

        assert stats.model == "deepseek-flash"
        assert stats.calls == 0
        assert stats.prompt_tokens == 0
        assert stats.prompt_cache_hit_tokens == 0
        assert stats.prompt_cache_miss_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0

    def test_log_cycle_not_called_when_no_calls(self, caplog=None):
        """log_cycle() при calls=0 не логирует (безопасный early return)."""
        stats = UsageStats(model="deepseek-flash")
        # вызов с caplog для проверки отсутствия логирования
        import logging
        if caplog is not None:
            with caplog.at_level(logging.INFO, logger="newsbot.llm.client"):
                stats.log_cycle()
            assert len(caplog.records) == 0
        else:
            # standalone runner: просто вызываем без ошибок
            stats.log_cycle()


# ─────────────────────────────────────────────────────────────────
# Тесты: off-peak гейт в pipeline
# ─────────────────────────────────────────────────────────────────

class TestOffPeakGate:
    """
    Интеграция off-peak гейта в run_cycle_with_delivery.

    Гейт расположен в run_cycle_with_delivery() ДО вызовов _process_batch().
    При off_peak_only=True + пик → early return ДО любого LLM-вызова.

    Паттерн тестирования:
      — подменяем is_llm_allowed_now (peak/off-peak контроль)
      — подменяем collector._client.close() (не падает при пустых sources)
      — мокаем _process_batch (проверяем что НЕ вызван при пике)
      — проверяем что items остаются new/pending
    """

    def _make_cfg(self, off_peak_only: bool):
        return {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "llm": {
                "enabled": True, "provider": "deepseek", "model": "deepseek-flash",
                "api_key": "fake-key", "base_url": "https://api.deepseek.com",
                "temperature": 0, "timeout_sec": 60, "max_retries": 3,
                "push_threshold": 7, "digest_threshold": 5, "system_prompt": "ты фильтр",
                "balance": {"enabled": False, "threshold": 2.0, "check_interval_hours": 24},
                "max_per_cycle": 150, "off_peak_only": off_peak_only,
            },
            "digest": {"enabled": False},
            "feedback": {"enabled": False},
        }

    def _run_cycle_with_gate(self, p, is_llm_allowed_now_value):
        """
        Запускает run_cycle_with_delivery с замоканным off-peak гейтом.
        NOTE: FeedCollector не мокаем — он требует реальный httpx.
        Но гейт срабатывает ДО создания FeedCollector, и при sources=[] —
        httpx.Client.close() не падает (создаётся но не используется).
        """
        from unittest import mock
        with mock.patch("newsbot.pipeline.is_llm_allowed_now", return_value=is_llm_allowed_now_value):
            p.run_cycle_with_delivery()

    def test_offpeak_true_peak_blocks_llm(self):
        """
        off_peak_only=True + пик (is_llm_allowed_now=False) →
        _process_batch НЕ вызван, items остаются new.
        """
        from newsbot.pipeline import NewsPipeline

        cfg = self._make_cfg(off_peak_only=True)
        path, cleanup = fresh_db()
        cfg["database"]["path"] = path
        try:
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = Storage(path)
            p.delivery = MockDelivery()
            p.llm_filter = FakeLLMFilter({"x": 9})
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._off_peak_only = True
            p._max_retries = 8
            p._feedback_context = None

            item = make_item(url="https://x.com/1", title="A", summary="S")
            p.storage.save(item)

            # мокаем _process_batch → проверяем что не вызван
            from unittest import mock
            with mock.patch.object(p, "_process_batch") as mock_batch:
                self._run_cycle_with_gate(p, is_llm_allowed_now_value=False)
                mock_batch.assert_not_called()

            # items остаются new (не прошли через _process_batch)
            new_items = p.storage.get_all_by_state("new")
            assert len(new_items) == 1, f"ожидалось 1 new-item, получено {len(new_items)}"

            p.storage.close()
        finally:
            cleanup()

    def test_offpeak_true_offpeak_calls_llm(self):
        """
        off_peak_only=True + off-peak (is_llm_allowed_now=True) →
        _process_batch вызван, items доставлены.
        """
        from newsbot.pipeline import NewsPipeline
        from unittest import mock

        cfg = self._make_cfg(off_peak_only=True)
        path, cleanup = fresh_db()
        cfg["database"]["path"] = path
        try:
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = Storage(path)
            p.delivery = MockDelivery()
            p.llm_filter = FakeLLMFilter({"x.com": 9})
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._off_peak_only = True
            p._max_retries = 8
            p._feedback_context = None

            item = make_item(url="https://x.com/1", title="A", summary="S")
            p.storage.save(item)

            with mock.patch.object(p, "_process_batch") as mock_batch:
                self._run_cycle_with_gate(p, is_llm_allowed_now_value=True)
                mock_batch.assert_called_once()

            p.storage.close()
        finally:
            cleanup()

    def test_offpeak_false_allows_llm_in_peak(self):
        """
        off_peak_only=False → _process_batch вызван ВСЕГДА, даже когда
        is_llm_allowed_now=False (пик игнорируется).
        """
        from newsbot.pipeline import NewsPipeline
        from unittest import mock

        cfg = self._make_cfg(off_peak_only=False)
        path, cleanup = fresh_db()
        cfg["database"]["path"] = path
        try:
            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = Storage(path)
            p.delivery = MockDelivery()
            p.llm_filter = FakeLLMFilter({"x.com": 9})
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._off_peak_only = False  # дефолт: вызовы всегда разрешены
            p._max_retries = 8
            p._feedback_context = None

            item = make_item(url="https://x.com/1", title="A", summary="S")
            p.storage.save(item)

            with mock.patch.object(p, "_process_batch") as mock_batch:
                # даже если is_llm_allowed_now возвращает False
                self._run_cycle_with_gate(p, is_llm_allowed_now_value=False)
                mock_batch.assert_called_once()

            p.storage.close()
        finally:
            cleanup()

    def test_offpeak_gate_preserves_retry_queue(self):
        """
        off_peak_only=True + пик: pending items (retry) остаются pending,
        retry-очередь не обрабатывается.
        """
        from newsbot.pipeline import NewsPipeline
        from unittest import mock

        cfg = self._make_cfg(off_peak_only=True)
        path, cleanup = fresh_db()
        cfg["database"]["path"] = path
        try:
            s = Storage(path)
            item = make_item(url="https://retry.com/1", title="Retry", summary="S")
            s.save(item)
            s.set_state(["https://retry.com/1"], "pending")
            item.llm_score = None

            p = NewsPipeline.__new__(NewsPipeline)
            p.config = cfg
            p.sources = []
            p.storage = s
            p.delivery = MockDelivery()
            p.llm_filter = FakeLLMFilter({})
            p._max_per_cycle = 150
            p._llm_calls_left = 150
            p._off_peak_only = True
            p._max_retries = 8
            p._feedback_context = None

            with mock.patch.object(p, "_process_batch") as mock_batch:
                self._run_cycle_with_gate(p, is_llm_allowed_now_value=False)
                mock_batch.assert_not_called()

            # retry item остался pending
            pending = p.storage.get_all_by_state("pending")
            assert len(pending) == 1, "retry item должен остаться pending"

            p.storage.close()
        finally:
            cleanup()

    def test_offpeak_string_parsing_false(self):
        """off_peak_only='false' (str) → отключено (не bool("false")==True)."""
        from newsbot.pipeline import NewsPipeline, _parse_bool

        assert _parse_bool(False) is False
        assert _parse_bool(True) is True
        assert _parse_bool("false") is False, "str 'false' должен быть False"
        assert _parse_bool("False") is False
        assert _parse_bool("FALSE") is False
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("1") is True
        assert _parse_bool("0") is False
        assert _parse_bool("yes") is True
        assert _parse_bool("no") is False
        assert _parse_bool("on") is True
        assert _parse_bool("off") is False
        assert _parse_bool("") is False
        assert _parse_bool(None) is False
        assert _parse_bool(123) is True   # truthy int → True (int check before fallback)

    def test_offpeak_gate_pending_counter_in_log(self, tmp_path, caplog):
        """
        off_peak_only=True + пик: в логе фигурирует реальное число
        waiting-items (new + pending), а не заниженное.
        total_waiting = count_by_state("new") + pending_count.
        """
        from newsbot.pipeline import NewsPipeline
        from unittest import mock

        db_path = str(tmp_path / "nb.db")
        cfg = self._make_cfg(off_peak_only=True)
        cfg["database"]["path"] = db_path

        s = Storage(db_path)
        # один new-item (candidates)
        item_new = make_item(url="https://new.com/1", title="New", summary="S")
        s.save(item_new)
        # один pending-item (retry)
        item_pending = make_item(url="https://pending.com/1", title="Pending", summary="S")
        s.save(item_pending)
        s.set_state(["https://pending.com/1"], "pending")

        p = NewsPipeline.__new__(NewsPipeline)
        p.config = cfg
        p.sources = []
        p.storage = s
        p.delivery = MockDelivery()
        p.llm_filter = FakeLLMFilter({})
        p._max_per_cycle = 150
        p._llm_calls_left = 150
        p._off_peak_only = True
        p._max_retries = 8
        p._feedback_context = None

        with caplog.at_level("INFO"):
            with mock.patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
                p.run_cycle_with_delivery()

        # лог содержит реальное число waiting items: 1 new + 1 pending = 2
        assert any(
            "2 items ждут off-peak" in r.message for r in caplog.records
        ), f"Ожидался счётчик 2 в логе. Записи: {[r.message for r in caplog.records]}"

        s.close()


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
        # usage stats
        (lambda: TestUsageStatsAccumulation().test_accumulate_across_multiple_responses(), "usage_accumulate_multiple_responses"),
        (lambda: TestUsageStatsAccumulation().test_accumulate_openai_cached_tokens_format(), "usage_accumulate_openai_cached"),
        (lambda: TestUsageStatsAccumulation().test_accumulate_deepseek_top_level_format(), "usage_accumulate_deepseek_top_level"),
        (lambda: TestUsageStatsAccumulation().test_reset_clears_counters_preserves_model(), "usage_reset_clears_counters"),
        (lambda: TestUsageStatsAccumulation().test_log_cycle_not_called_when_no_calls(), "usage_log_cycle_no_calls"),
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