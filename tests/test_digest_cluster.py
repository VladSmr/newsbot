"""
Тесты финальной LLM-кластеризации дублей в дайджесте (Этап 5, Фаза 2).

Покрывает:
  - _cluster_deferred_duplicates: все edge-cases (unit, без БД)
  - run_digest(force=True): фильтрация дублей из текста (интеграция, tmp БД)
  - run_digest: терминализация дублей только при полном успехе
  - run_digest: при sent_count==0 дубли НЕ помечаются digested

Запуск: python -m pytest tests/test_digest_cluster.py -v
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.pipeline import NewsPipeline
from newsbot.storage import Storage


# =============================================================================
# helpers
# =============================================================================

def _mk_item(
    source="TestSrc",
    title="Test Title",
    url=None,
    summary="Test summary.",
    llm_score=6,
    **kwargs,
) -> NewsItem:
    # url — явный параметр; если передан, используется как есть (не хэшируется).
    if url is None:
        url = f"https://example.com/{hash(title) & 0xFFFF:05d}"
    return NewsItem(
        source=source, url=url, title=title, summary=summary,
        published="2025-01-01 12:00", published_parsed=None,
        llm_score=llm_score, **kwargs,
    )


@contextmanager
def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _insert_deferred_item(db_path, title, url, published, fetched_at, score=6):
    """
    Вставляет deferred-item напрямую в БД с заданным fetched_at.
    Storage.save() игнорирует item.fetched_at (всегда ставит NOW),
    поэтому нужен прямой INSERT для контроля времени.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
        """,
        ("TestSrc", url, title, "Test summary", published, fetched_at, fetched_at, score),
    )
    conn.commit()
    conn.close()


def _mk_pipeline_unit(llm_filter=None, digest_cfg=None, off_peak_only=False):
    """
    Создаёт объект Pipeline через __new__ (БЕЗ Storage, без реальной БД).
    Подменяет llm_filter и _off_peak_only.
    """
    cfg = {
        "sources": [],
        "database": {"path": "data/nb.db"},
        "telegram": {"token": "", "chat_id": ""},
        "digest": digest_cfg or {"cluster_duplicates": True},
        "llm": {"enabled": True, "api_key": "fake", "off_peak_only": off_peak_only},
    }
    with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
        with patch.object(NewsPipeline, "_build_llm_filter", return_value=llm_filter):
            p = NewsPipeline(cfg)
            p.llm_filter = llm_filter
            return p


def _mk_pipeline_integration(db_path, digest_cfg=None, llm_filter=None):
    """Создаёт Pipeline с временной БД."""
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
            "cluster_duplicates": True,
        },
        "llm": {"enabled": False},
    }
    with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
        with patch.object(NewsPipeline, "_build_llm_filter", return_value=llm_filter):
            p = NewsPipeline(cfg)
            # Закрываем первое соединение перед подменой
            p.storage.close()
            p.storage = Storage(db_path)
            if llm_filter is not None:
                p.llm_filter = llm_filter
            return p


# =============================================================================
# _cluster_deferred_duplicates — unit (без БД, через __new__)
# =============================================================================

class TestClusterUnit(unittest.TestCase):
    """Тесты _cluster_deferred_duplicates через __new__ (без Storage)."""

    def test_detects_duplicates_and_returns_urls(self):
        """3 items, detect_duplicates → {1,2} (0-based) ⇒ возвращает {url1, url2}."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1, 2}

        items = [
            _mk_item(title="A", url="https://ex.com/u0"),
            _mk_item(title="B", url="https://ex.com/u1"),
            _mk_item(title="C", url="https://ex.com/u2"),
        ]

        p = _mk_pipeline_unit(llm_filter=mock_filter)
        result = p._cluster_deferred_duplicates(items)

        self.assertEqual(result, {"https://ex.com/u1", "https://ex.com/u2"})
        self.assertNotIn("https://ex.com/u0", result, "Лидер не должен быть в дублях")

    def test_empty_list_returns_empty(self):
        """Пустой список → вызов LLM не делается, пустое множество."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result = p._cluster_deferred_duplicates([])
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_single_item_returns_empty(self):
        """1 item → вызов LLM не делается."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result = p._cluster_deferred_duplicates([_mk_item(title="Only", url="https://ex.com/o1")])
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_llm_filter_none_returns_empty(self):
        """llm_filter is None → безопасный пропуск."""
        p = _mk_pipeline_unit(llm_filter=None)

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/a0"),
            _mk_item(title="B", url="https://ex.com/a1"),
        ])
        self.assertEqual(result, set())

    def test_detect_duplicates_raises_returns_empty(self):
        """detect_duplicates бросает исключение → безопасный пропуск."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.side_effect = RuntimeError("network error")

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/b0"),
            _mk_item(title="B", url="https://ex.com/b1"),
        ])
        self.assertEqual(result, set())

    def test_detect_duplicates_returns_non_set_returns_empty(self):
        """detect_duplicates вернул не (set|frozenset) → безопасный пропуск."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = MagicMock()  # не set

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/c0"),
            _mk_item(title="B", url="https://ex.com/c1"),
        ])
        self.assertEqual(result, set())

    def test_detect_duplicates_returns_frozenset_accepted(self):
        """detect_duplicates вернул frozenset → принимается (set | frozenset)."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = frozenset({1})

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/d0"),
            _mk_item(title="B", url="https://ex.com/d1"),
        ])
        self.assertEqual(result, {"https://ex.com/d1"})

    def test_cluster_duplicates_false_bool_skips_llm(self):
        """cluster_duplicates=false (bool) → вызов LLM не делается."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(llm_filter=mock_filter, digest_cfg={"cluster_duplicates": False})

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/e0"),
            _mk_item(title="B", url="https://ex.com/e1"),
        ])
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_cluster_duplicates_false_string_skips_llm(self):
        """cluster_duplicates="false" (строка) → вызов LLM не делается."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(
            llm_filter=mock_filter,
            digest_cfg={"cluster_duplicates": "false"},
        )

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/f0"),
            _mk_item(title="B", url="https://ex.com/f1"),
        ])
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_cluster_duplicates_default_true(self):
        """По умолчанию cluster_duplicates=True (без ключа в конфиге)."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        cfg = {
            "sources": [],
            "database": {"path": "data/nb.db"},
            "telegram": {"token": "", "chat_id": ""},
            "digest": {},  # нет ключа cluster_duplicates
            "llm": {"enabled": True, "api_key": "fake"},
        }
        with patch.object(NewsPipeline, "_build_delivery", return_value=MagicMock()):
            with patch.object(NewsPipeline, "_build_llm_filter", return_value=mock_filter):
                p = NewsPipeline(cfg)
                p.llm_filter = mock_filter

        result = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/g0"),
            _mk_item(title="B", url="https://ex.com/g1"),
        ])
        mock_filter.detect_duplicates.assert_called_once()
        self.assertEqual(result, {"https://ex.com/g1"})

    def test_off_peak_skips_llm(self):
        """off_peak_only=True + в пик → вызов LLM не делается."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
            p = _mk_pipeline_unit(llm_filter=mock_filter, off_peak_only=True)

            result = p._cluster_deferred_duplicates([
                _mk_item(title="A", url="https://ex.com/h0"),
                _mk_item(title="B", url="https://ex.com/h1"),
            ])

        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_off_peak_allows_llm_in_off_peak(self):
        """off_peak_only=True + вне пика → вызов LLM делается."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=True):
            p = _mk_pipeline_unit(llm_filter=mock_filter, off_peak_only=True)

            result = p._cluster_deferred_duplicates([
                _mk_item(title="A", url="https://ex.com/j0"),
                _mk_item(title="B", url="https://ex.com/j1"),
            ])

        mock_filter.detect_duplicates.assert_called_once()
        self.assertEqual(result, {"https://ex.com/j1"})

    def test_out_of_range_indices_ignored(self):
        """Индексы за пределами списка игнорируются."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1, 999, -1}

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        items = [_mk_item(title=str(i), url=f"https://ex.com/k{i}") for i in range(3)]
        result = p._cluster_deferred_duplicates(items)

        self.assertEqual(result, {"https://ex.com/k1"})

    def test_string_indices_ignored(self):
        """Не-числовые индексы (строки, None) игнорируются."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1, "2", None}

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        items = [_mk_item(title=str(i), url=f"https://ex.com/l{i}") for i in range(5)]
        result = p._cluster_deferred_duplicates(items)

        self.assertEqual(result, {"https://ex.com/l1"})


# =============================================================================
# run_digest — интеграция (временная БД)
# =============================================================================

class TestRunDigestClusterIntegration(unittest.TestCase):
    """Интеграционные тесты: фильтрация + терминализация дублей в run_digest."""

    def test_duplicates_filtered_and_digested_on_full_success(self):
        """
        Все items отправлены (sent_count == total_to_send):
        лидер → отправлен, дубли → отфильтрованы и не отправлены.
        После успешного дайджеста ВСЁ окно удалено из БД.
        """
        with _tmp_db() as db_path:
            # Инициализируем схему
            s = Storage(db_path)
            s.close()

            # Вставляем напрямую с явным fetched_at в ПРОШЛОМ.
            # Storage.save() ставит NOW = digest_started_iso (обе обрезаны до секунды)
            # → fetched_at < cutoff = False → items не удалятся.
            leader_fetched = "2025-01-01T12:00:00+00:00"
            _insert_deferred_item(
                db_path,
                title="Лидер", url="https://ex.com/leader",
                published="2025-01-01 12:00",
                fetched_at=leader_fetched,
                score=6,
            )
            for i in range(1, 3):
                _insert_deferred_item(
                    db_path,
                    title=f"Дубль{i}", url=f"https://ex.com/dup{i}",
                    published="2025-01-01 12:00",
                    fetched_at=leader_fetched,
                    score=6,
                )

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = {1, 2}

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 1)
            # Проверяем отправку: лидер отправлен, дубли отфильтрованы
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("Лидер", send_args)
            self.assertNotIn("Дубль1", send_args)
            self.assertNotIn("Дубль2", send_args)

            # После успешного дайджеста ВСЁ окно удалено из БД
            # (лидер отправлен, дубли терминализированы — всё удалено delete_items_before)
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertNotIn("https://ex.com/leader", all_urls)
            self.assertNotIn("https://ex.com/dup1", all_urls)
            self.assertNotIn("https://ex.com/dup2", all_urls)
            p.close()

    def test_duplicates_not_digested_when_send_total_failure(self):
        """
        sent_count==0 (доставка полностью провалилась):
        ни лидер, ни дубли НЕ помечаются digested.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            leader = _mk_item(title="Лидер", url="https://ex.com/l", llm_score=6)
            dup = _mk_item(title="Дубль", url="https://ex.com/d", llm_score=6)
            s.save(leader)
            s.save(dup)
            s.set_state(["https://ex.com/l"], "deferred")
            s.set_state(["https://ex.com/d"], "deferred")
            s.close()

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = {1}

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = False
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 0)
            s = p.storage
            digested_urls = {i.url for i in s.get_all_by_state("digested")}
            self.assertNotIn("https://ex.com/l", digested_urls)
            self.assertNotIn("https://ex.com/d", digested_urls)
            deferred_urls = {i.url for i in s.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/l", deferred_urls)
            self.assertIn("https://ex.com/d", deferred_urls)
            s.close()
            p.close()

    def test_duplicates_not_digested_on_partial_failure(self):
        """
        Частичный сбой (sent_count > 0, но < total_to_send).
        4 deferred-items, НЕ фильтруются (detect_duplicates → {}).
        msg1 (2 items) → OK, msg2 (2 items) → FAIL.
        sent_count=2 < total_to_send=4 → дубли НЕ терминализируются.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            for i, url in enumerate([
                "https://ex.com/l1", "https://ex.com/l2",
                "https://ex.com/l3", "https://ex.com/l4",
            ]):
                item = _mk_item(title=f"Item{i+1}", url=url, llm_score=6)
                s.save(item)
                s.set_state([url], "deferred")
            s.close()

            mock_filter = MagicMock()
            # Ни одного дубля — все 4 items идут в digest.
            # (Чтобы проверить частичный сбой без влияния фильтрации.)
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 2,
                    "cluster_duplicates": True,
                },
                llm_filter=mock_filter,
            )
            delivery = MagicMock()
            call_count = [0]

            def fake_send(text):
                call_count[0] += 1
                # msg1 (2 items) → OK, msg2 (2 items) → FAIL
                return call_count[0] == 1

            delivery.send_text.side_effect = fake_send
            p.delivery = delivery

            count = p.run_digest(force=True)

            # sent_count=2 (msg1 отправлен), total_to_send=4 (4 items в groups)
            self.assertEqual(count, 2)
            s = p.storage
            digested_urls = {i.url for i in s.get_all_by_state("digested")}
            deferred_urls = {i.url for i in s.get_all_by_state("deferred")}
            # 2 items из msg1 помечены digested
            self.assertEqual(len(digested_urls), 2, digested_urls)
            # 2 items из msg2 НЕ помечены (частичный сбой: sent_count=2 < total_to_send=4)
            for url in ["https://ex.com/l3", "https://ex.com/l4"]:
                self.assertNotIn(url, digested_urls)
            # Оба отложенных item'а (msg2) остались deferred
            for url in ["https://ex.com/l3", "https://ex.com/l4"]:
                self.assertIn(url, deferred_urls)
            s.close()
            p.close()

    def test_set_digested_error_logs_warning_does_not_crash(self):
        """
        При ошибке set_digested (pipeline-вызов для дублей) — warning,
        run_digest НЕ падает. Лидер отправлен, дубли остаются deferred
        (best-effort: fallback set_state убран).
        """
        import sqlite3
        with _tmp_db() as db_path:
            # Инициализируем схему
            s = Storage(db_path)
            s.close()

            # Вставляем напрямую: явный fetched_at в ПРОШЛОМ гарантирует
            # удаление при delete_items_before (fetched_at < digest_started_iso).
            # Storage.save() ставит NOW, что совпадает с digest_started_iso
            # (обе обрезаны до секунды) → fetched_at < cutoff = False → не удалится.
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
                """,
                ("TestSrc", "https://ex.com/lead2", "Лидер",
                 "Test summary", "2025-01-01 12:00",
                 "2025-01-01T12:00:00+00:00",
                 "2025-01-01T12:00:00+00:00", 6),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
                """,
                ("TestSrc", "https://ex.com/d2", "Дубль",
                 "Test summary", "2025-01-01 12:00",
                 "2025-01-01T12:00:00+00:00",
                 "2025-01-01T12:00:00+00:00", 6),
            )
            conn.commit()
            conn.close()

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = {1}  # dup = дубль

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Патчим set_digested: первый вызов (digest, для лидера) — реальный write,
            # второй вызов (pipeline, для дублей) — бросает.
            call_count = [0]
            orig_set_digested = p.storage.set_digested

            def tracking_set_digested(urls):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Первый вызов: digest для лидера — реальный write
                    return orig_set_digested(urls)
                else:
                    # Второй вызов: pipeline для дублей — FAIL
                    raise RuntimeError("DB error")

            with patch.object(p.storage, "set_digested", side_effect=tracking_set_digested):
                count = p.run_digest(force=True)

            # run_digest не упал
            self.assertEqual(count, 1)
            # Лидер отправлен (дубли отфильтрованы из groups — они не попали в send_text)
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("Лидер", send_args)
            self.assertNotIn("Дубль", send_args)

            # Переоткрываем соединение чтобы увидеть закоммиченные данные
            p.storage.close()
            s = Storage(db_path)
            # lead2: selected -> digested -> УДАЛЁН delete_items_before.
            # d2: NOT selected (дубль, detect_duplicates исключило) -> остаётся deferred.
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in s.get_all_by_state(state))
            self.assertNotIn("https://ex.com/lead2", all_urls,
                "lead2 (selected -> digested) удалён delete_items_before")
            deferred_urls = {i.url for i in s.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/d2", deferred_urls,
                "d2 (NOT selected) остаётся deferred (delete_items_before не трогает deferred)")
            s.close()

    def test_no_duplicates_means_all_sent(self):
        """detect_duplicates пустое → все items отправлены, ни один не помечен twice."""
        with _tmp_db() as db_path:
            # Инициализируем схему
            s = Storage(db_path)
            s.close()

            # Вставляем напрямую с явным fetched_at в ПРОШЛОМ.
            # Storage.save() ставит NOW, что совпадает с digest_started_iso
            # (обе обрезаны до секунды) → fetched_at < cutoff = False.
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Item{i}", url=f"https://ex.com/i{i}",
                    published="2025-01-01 12:00",
                    fetched_at="2025-01-01T12:00:00+00:00",
                    score=6,
                )

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 3)
            # Проверяем что отправлены все 3 items
            for i in range(3):
                self.assertIn(f"Item{i}", delivery.send_text.call_args[0][0])
            # После успешного дайджеста ВСЁ окно удалено из БД
            s = p.storage
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in s.get_all_by_state(state))
            for i in range(3):
                self.assertNotIn(f"https://ex.com/i{i}", all_urls)
            s.close()
            p.close()


if __name__ == "__main__":
    unittest.main()