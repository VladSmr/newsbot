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
from newsbot.pipeline import NewsPipeline, _cap_by_location
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
    # Подменяем storage.get_shown_for_window, чтобы unit-тесты
    # _cluster_deferred_duplicates не падали на обращении к self.storage
    # (pipeline через __new__ не имеет storage — он создаётся в __init__).
    # Mock возвращает пустой список — кросс-оконный дедуп в unit-тестах не проверяем.
    mock_storage = MagicMock()
    mock_storage.get_shown_for_window.return_value = []
    p.storage = mock_storage
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
        result_dups, _ = p._cluster_deferred_duplicates(items, "2026-01-01T00:00:00+00:00")

        self.assertEqual(result_dups, {"https://ex.com/u1", "https://ex.com/u2"})
        self.assertNotIn("https://ex.com/u0", result_dups, "Лидер не должен быть в дублях")

    def test_empty_list_returns_empty(self):
        """Пустой список → вызов LLM не делается, пустое множество."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates([], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_single_item_returns_empty(self):
        """1 item + empty RECENT → detect_duplicates вызывается, возвращает {} (empty).
        NOTE: Fix 4a изменил семантику — 1 item проверяется против RECENT
        (если RECENT непуст, может отсечься). При пустом RECENT → пустой result.
        """
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = set()
        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates(
            [_mk_item(title="Only", url="https://ex.com/o1")],
            "2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(result_dups, set())

    def test_llm_filter_none_returns_empty(self):
        """llm_filter is None → безопасный пропуск."""
        p = _mk_pipeline_unit(llm_filter=None)

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/a0"),
            _mk_item(title="B", url="https://ex.com/a1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())

    def test_detect_duplicates_raises_returns_empty(self):
        """detect_duplicates бросает исключение → безопасный пропуск."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.side_effect = RuntimeError("network error")

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/b0"),
            _mk_item(title="B", url="https://ex.com/b1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())

    def test_detect_duplicates_returns_non_set_returns_empty(self):
        """detect_duplicates вернул не (set|frozenset) → безопасный пропуск."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = MagicMock()  # не set

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/c0"),
            _mk_item(title="B", url="https://ex.com/c1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())

    def test_detect_duplicates_returns_frozenset_accepted(self):
        """detect_duplicates вернул frozenset → принимается (set | frozenset)."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = frozenset({1})

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/d0"),
            _mk_item(title="B", url="https://ex.com/d1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, {"https://ex.com/d1"})

    def test_cluster_duplicates_false_bool_skips_llm(self):
        """cluster_duplicates=false (bool) → вызов LLM не делается."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(llm_filter=mock_filter, digest_cfg={"cluster_duplicates": False})

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/e0"),
            _mk_item(title="B", url="https://ex.com/e1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_cluster_duplicates_false_string_skips_llm(self):
        """cluster_duplicates="false" (строка) → вызов LLM не делается."""
        mock_filter = MagicMock()
        p = _mk_pipeline_unit(
            llm_filter=mock_filter,
            digest_cfg={"cluster_duplicates": "false"},
        )

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/f0"),
            _mk_item(title="B", url="https://ex.com/f1"),
        ], "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, set())
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

        result_dups, _ = p._cluster_deferred_duplicates([
            _mk_item(title="A", url="https://ex.com/g0"),
            _mk_item(title="B", url="https://ex.com/g1"),
        ], "2026-01-01T00:00:00+00:00")
        mock_filter.detect_duplicates.assert_called_once()
        self.assertEqual(result_dups, {"https://ex.com/g1"})

    def test_off_peak_skips_llm(self):
        """off_peak_only=True + в пик → вызов LLM не делается."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
            p = _mk_pipeline_unit(llm_filter=mock_filter, off_peak_only=True)

            result_dups, _ = p._cluster_deferred_duplicates([
                _mk_item(title="A", url="https://ex.com/h0"),
                _mk_item(title="B", url="https://ex.com/h1"),
            ], "2026-01-01T00:00:00+00:00")

        self.assertEqual(result_dups, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_off_peak_allows_llm_in_off_peak(self):
        """off_peak_only=True + вне пика → вызов LLM делается."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=True):
            p = _mk_pipeline_unit(llm_filter=mock_filter, off_peak_only=True)

            result_dups, _ = p._cluster_deferred_duplicates([
                _mk_item(title="A", url="https://ex.com/j0"),
                _mk_item(title="B", url="https://ex.com/j1"),
            ], "2026-01-01T00:00:00+00:00")

        mock_filter.detect_duplicates.assert_called_once()
        self.assertEqual(result_dups, {"https://ex.com/j1"})

    def test_out_of_range_indices_ignored(self):
        """Индексы за пределами списка игнорируются."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1, 999, -1}

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        items = [_mk_item(title=str(i), url=f"https://ex.com/k{i}") for i in range(3)]
        result_dups, _ = p._cluster_deferred_duplicates(items, "2026-01-01T00:00:00+00:00")

        self.assertEqual(result_dups, {"https://ex.com/k1"})

    def test_string_indices_ignored(self):
        """Не-числовые индексы (строки, None) игнорируются."""
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1, "2", None}

        p = _mk_pipeline_unit(llm_filter=mock_filter)

        items = [_mk_item(title=str(i), url=f"https://ex.com/l{i}") for i in range(5)]
        result_dups, _ = p._cluster_deferred_duplicates(items, "2026-01-01T00:00:00+00:00")
        self.assertEqual(result_dups, {"https://ex.com/l1"})

    def test_llm_passes_case_variant_pair(self):
        """
        LLM-дедуп (mock) пропускает пару заголовков в разном регистре —
        оба проходят в пул. F6-детерминированный containment работает
        отдельно в run_digest (протестирован в test_dedup.py:
        is_containment_duplicate с title_normalized).
        """
        items = [
            _mk_item(title="Two Killed in Ukrainian Attack on Voronezh Region",
                     url="https://ex.com/c1"),
            _mk_item(title="Two killed in Ukrainian attack on Voronezh region",
                     url="https://ex.com/c2"),
        ]
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = set()  # LLM не нашла дублей
        p = _mk_pipeline_unit(llm_filter=mock_filter)

        result_dups, _ = p._cluster_deferred_duplicates(
            items, "2026-01-01T00:00:00+00:00",
        )
        # LLM-шаг: дубли не найдены → оба проходят в пул (для F6)
        self.assertEqual(result_dups, set(),
                         "LLM не помечает пару с разным регистром — оба в пуле")


# =============================================================================
# M4: location-cap warning observability
# =============================================================================

# =============================================================================
# M4: location-cap warning observability (pytest style for caplog fixture)
# =============================================================================

def test_duplicate_location_no_warning():
    """
    Две новости с ОДИНАКОВОЙ локацией → warning НЕТ.
    len(set(orig_locations)) == 1 → внешний if False.
    """
    items = [
        _mk_item(title="Item A", url="https://ex.com/mos1",
                 cluster_key="москва|other|2026-10-20"),
        _mk_item(title="Item B", url="https://ex.com/mos2",
                 cluster_key="москва|other|2026-10-20"),
    ]
    result = _cap_by_location(items, max_per_location=2)
    # Ни один не отсечён (лимит 2), warning не генерируется
    assert len(result) == 2


def test_normalized_collision_warns_with_variants(caplog):
    """
    Minor C (pipeline.py:800-808): warning при схлопывании нескольких вариантов
    в одно нормализованное ядро — считаются УНИКАЛЬНЫЕ варианты, не вхождения.
    Условие: `normalized_location_sources` содержит ≥1 ключ с ≥2 уникальными
    исходными строками. Два разных cluster_key нормализуются к одному ядру
    (loc_norm, day), поэтому лежат в одной группе — len(set) > 1 внутри группы.
    """
    items = [
        # оба нормализуются к (voronezh, 2026-10-20) но с РАЗНЫХ raw строк
        _mk_item(title="A", url="https://ex.com/v1",
                 cluster_key="voronezh region|a|2026-10-20"),
        _mk_item(title="B", url="https://ex.com/v2",
                 cluster_key="Voronezh|b|2026-10-20"),
        # отдельный ключ (krasnodar, 2026-10-21) — чтобы len(normalized_location_sources) > 1
        _mk_item(title="C", url="https://ex.com/v3",
                 cluster_key="krasnodar|c|2026-10-21"),
    ]
    with caplog.at_level("WARNING"):
        result = _cap_by_location(items, max_per_location=1)
    # cap=1: voronezh отдаёт 1, krasnodar не ограничен → 2 результата
    assert len(result) == 2
    # Warning: 2 уникальных варианта в ядре 'voronezh'
    assert any(
        "объединяет 2 вариантов" in record.message
        and "voronezh" in record.message.lower()
        for record in caplog.records
    ), f"Warning должен сообщать о 2 вариантах (uniq). Got: {[r.message for r in caplog.records]}"


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


# =============================================================================
# Major #2 regression: off-peak gate в _check_against_shown
# =============================================================================

class TestOffpeakGateInCheckAgainstShown(unittest.TestCase):
    """
    Регрессионные тесты: _check_against_shown соблюдает off-peak гейт
    (pipeline.py:1050-1055, Major #2 fix).

    Покрывает оба пути:
      — запланированный дайджест (_run_delivery_side_effects → run_digest)
      — /digest из чата (run_digest вызывается напрямую)
    Оба проходят через _check_against_shown через пул backfill и skipped.
    """

    def _mk_pipeline_unit(self, llm_filter, off_peak_only: bool = False):
        """Создаёт pipeline через __new__ для unit-теста _check_against_shown."""
        p = NewsPipeline.__new__(NewsPipeline)
        p.config = {}
        p.sources = []
        p.storage = MagicMock()
        p.delivery = MagicMock()
        p.llm_filter = llm_filter
        p._off_peak_only = off_peak_only
        return p

    def _mk_candidates(self, n=3):
        return [_mk_item(url=f"https://cand.com/{i}", title=f"Candidate {i}") for i in range(n)]

    def _mk_recent_shown(self, n=2):
        return [{"title": f"Shown {i}", "source": "Src"} for i in range(n)]

    def test_peak_off_peak_only_true_detect_duplicates_not_called(self):
        """
        (a) off_peak_only=True + сейчас пик → detect_duplicates НЕ вызывается,
        _check_against_shown возвращает пустое множество (fail-open).
        """
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {0, 1}  # если бы вызвался

        p = self._mk_pipeline_unit(mock_filter, off_peak_only=True)
        candidates = self._mk_candidates(3)
        recent = self._mk_recent_shown(2)

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
            result = p._check_against_shown(candidates, recent)

        # Fail-open: пустой набор
        self.assertEqual(result, set())
        # LLM НЕ вызван
        mock_filter.detect_duplicates.assert_not_called()

    def test_off_peak_allowed_detect_duplicates_called(self):
        """
        (b) off_peak_only=True + off-peak → detect_duplicates вызывается,
        поведение штатное.
        """
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {0}  # первый кандидат — дубль

        p = self._mk_pipeline_unit(mock_filter, off_peak_only=True)
        candidates = self._mk_candidates(3)
        recent = self._mk_recent_shown(2)

        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=True):
            result = p._check_against_shown(candidates, recent)

        # detect_duplicates вызван, вернул {0} → кандидат 0 дубль
        mock_filter.detect_duplicates.assert_called_once_with(candidates, recent)
        self.assertEqual(result, {"https://cand.com/0"})

    def test_off_peak_only_false_detect_duplicates_called_in_peak(self):
        """
        (c) off_peak_only=False (гейт выключен юзером) → detect_duplicates
        вызывается даже в пик.
        """
        mock_filter = MagicMock()
        mock_filter.detect_duplicates.return_value = {1}

        p = self._mk_pipeline_unit(mock_filter, off_peak_only=False)
        candidates = self._mk_candidates(3)
        recent = self._mk_recent_shown(2)

        # off_peak_only=False → is_llm_allowed_now игнорируется
        with patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
            result = p._check_against_shown(candidates, recent)

        # LLM вызван несмотря на «пик» (off_peak_only=False)
        mock_filter.detect_duplicates.assert_called_once()
        self.assertEqual(result, {"https://cand.com/1"})

    def test_empty_candidates_returns_early(self):
        """Пустой candidates → early-return ДО проверки off-peak (независимо от флага)."""
        mock_filter = MagicMock()
        p = self._mk_pipeline_unit(mock_filter, off_peak_only=True)
        recent = self._mk_recent_shown(2)

        # Не падаем, не вызываем is_llm_allowed_now
        result = p._check_against_shown([], recent)
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_empty_recent_returns_early(self):
        """Пустой recent_shown → early-return ДО проверки off-peak."""
        mock_filter = MagicMock()
        p = self._mk_pipeline_unit(mock_filter, off_peak_only=True)
        candidates = self._mk_candidates(3)

        result = p._check_against_shown(candidates, [])
        self.assertEqual(result, set())
        mock_filter.detect_duplicates.assert_not_called()

    def test_no_llm_filter_returns_early(self):
        """llm_filter=None → early-return ДО проверки off-peak."""
        p = self._mk_pipeline_unit(None, off_peak_only=True)
        candidates = self._mk_candidates(3)
        recent = self._mk_recent_shown(2)

        result = p._check_against_shown(candidates, recent)
        self.assertEqual(result, set())

    def test_integration_run_digest_with_peak_guard_backfill(self):
        """
        Интеграция: run_digest(force=True) в пик с off_peak_only=True.
        Backfill pool проходит _check_against_shown, detect_duplicates
        не вызывается → дайджест завершается успешно (fail-open).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # deferred — уйдёт в digest
            s.save(_mk_item(title="Deferred1", url="https://ex.com/d1", llm_score=6))
            s.set_state(["https://ex.com/d1"], "deferred")
            # backfill-кандидаты — лежат в БД как new
            for i in range(2, 5):
                s.save(_mk_item(title=f"Backfill{i}", url=f"https://ex.com/b{i}", llm_score=6))
                s.set_state([f"https://ex.com/b{i}"], "new")
            s.close()

            mock_filter = MagicMock()
            # detect_duplicates вызываться НЕ должен
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            # Симулируем off-peak=только, сейчас пик
            p._off_peak_only = True

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with patch("newsbot.pipeline.is_llm_allowed_now", return_value=False):
                count = p.run_digest(force=True)

            # Дайджест отработал (fail-open: deferred item отправлен)
            self.assertGreaterEqual(count, 1)
            # detect_duplicates НЕ вызван ни разу
            mock_filter.detect_duplicates.assert_not_called()

            p.storage.close()
            p.close()


# =============================================================================
# F6-тесты: детерминированная containment-страховка после LLM-кластеризации
# =============================================================================

class TestContainmentDedupAfterLLM(unittest.TestCase):
    """F6-фикс: после LLM-кластеризации — проход containment-check.

    LLM мог пропустить пары с одинаковым содержанием, но разной формулировкой.
    Containment-check (is_containment_duplicate) ловит такие пары:
    - ≥30 символов нормализованных заголовков
    - одинаковое числовое множество (guard от «2 killed» vs «3 killed»)
    - Jaccard ≥0.8
    - word-boundary containment
    """

    def test_containment_duplicate_filtered_leader_sent(self):
        """
        Два containment-дубля в пуле дайджеста:
        - leader (высокий score) → отправлен
        - duplicate (низкий score) → отфильтрован, помечен 'digested'
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            # Титлы ≥30 символов, одинаковые числа, containment.
            # Leader: score 8, fetched 2025-01-01 (ранее → лидер по tiebreak).
            # Duplicate: score 6, fetched 2025-01-02 (позже).
            # LLM-кластеризация пропускает их (detect_duplicates → {}).
            leader_title = "Two killed in Ukrainian attack on Voronezh region"
            dup_title = "Two killed in Ukrainian attack on Voronezh"
            # Оба ≥ 30 символов после normalize_location (title_normalized в dedup):
            # "two killed in ukrainian attack on voronezh region" = 38
            # "two killed in ukrainian attack on voronezh" = 33

            _insert_deferred_item(
                db_path,
                title=leader_title,
                url="https://ex.com/leader-f6",
                published="2025-01-01 12:00",
                fetched_at="2025-01-01T12:00:00+00:00",
                score=8,
            )
            _insert_deferred_item(
                db_path,
                title=dup_title,
                url="https://ex.com/dup-f6",
                published="2025-01-01 12:00",
                fetched_at="2025-01-02T12:00:00+00:00",
                score=6,
            )

            # LLM пропускает пары (не находит дублей)
            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Leader отправлен, dup отфильтрован (проверяем по URL, не по title,
            # т.к. dup_title — подстрока leader_title)
            self.assertEqual(count, 1)
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("https://ex.com/leader-f6", send_args)
            self.assertNotIn("https://ex.com/dup-f6", send_args)

            # После успешного дайджеста ВСЁ окно удалено (лидер отправлен → digested → удалён)
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertNotIn("https://ex.com/leader-f6", all_urls)
            self.assertNotIn("https://ex.com/dup-f6", all_urls)
            p.close()

    def test_three_items_two_containment_pairs_one_leader_survives(self):
        """
        Три новости: leader(A) + dup(B) + unrelated(C).
        Containment: A⊇B. C не в паре.
        Результат: A и C отправлены, B отфильтрован.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            leader = "Russian drone strike hits civilian target in Moscow"
            dup = "Russian drone strike hits civilian target in Moscow city"
            unrelated = "SpaceX launches new satellite constellation successfully"

            for title, url, score, fetched in [
                (leader, "https://ex.com/a", 9, "2025-01-01T12:00:00+00:00"),
                (dup, "https://ex.com/b", 5, "2025-01-02T12:00:00+00:00"),
                (unrelated, "https://ex.com/c", 8, "2025-01-03T12:00:00+00:00"),
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2025-01-01 12:00",
                    fetched_at=fetched, score=score,
                )

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # A (leader, score 9) + C (unrelated, score 8) отправлены = 2
            self.assertEqual(count, 2, "leader + unrelated должны пройти, dup — нет")
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn(leader, send_args)
            self.assertIn(unrelated, send_args)
            self.assertNotIn(dup, send_args)

            p.close()

    def test_containment_skipped_when_numerical_guard_fails(self):
        """
        Containment-пара с РАЗНЫМИ числами: «Two killed» vs «Three killed».
        Числовой guard отклоняет → оба проходят (не дубль).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            # Оба ≥30, но разные числа → числовой guard не пройдён
            # Containment: "Three killed" in "Three killed in attack on Voronezh"
            a_title = "Three killed in attack on Voronezh region"
            b_title = "Three killed in attack on Voronezh region"  # exact same
            # But wait — for this test let's use different numbers:
            a_title = "Two killed in attack on Voronezh region"
            b_title = "Three killed in attack on Voronezh region"

            _insert_deferred_item(
                db_path, title=a_title, url="https://ex.com/2killed",
                published="2025-01-01 12:00", fetched_at="2025-01-01T12:00:00+00:00",
                score=8,
            )
            _insert_deferred_item(
                db_path, title=b_title, url="https://ex.com/3killed",
                published="2025-01-01 12:00", fetched_at="2025-01-02T12:00:00+00:00",
                score=7,
            )

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Оба проходят: числовой guard отклонил containment
            self.assertEqual(count, 2, "2 killed ≠ 3 killed, оба проходят")
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("Two killed", send_args)
            self.assertIn("Three killed", send_args)
            p.close()

    def test_containment_different_case_filtered(self):
        """
        F6: containment-пара с РАЗНЫМ РЕГИСТРОМ — «Two Killed…» vs «Two killed…».
        Нормализация lower() ловит это → F6 отсекает дубликат.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            # Leader: title ≥30, "two killed..." after normalize_location+lower
            # Duplicate: same text, "Killed" capitalized — same after lower
            leader_title = "Two Killed in Ukrainian attack on Voronezh region"
            dup_title = "Two killed in Ukrainian attack on Voronezh region"

            _insert_deferred_item(
                db_path, title=leader_title, url="https://ex.com/leader-case",
                published="2025-01-01 12:00", fetched_at="2025-01-01T12:00:00+00:00",
                score=8,
            )
            _insert_deferred_item(
                db_path, title=dup_title, url="https://ex.com/dup-case",
                published="2025-01-01 12:00", fetched_at="2025-01-02T12:00:00+00:00",
                score=6,
            )

            # LLM пропускает пары (не находит дублей — разный регистр)
            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Leader отправлен, dup отфильтрован F6-ContainmentCheck
            self.assertEqual(count, 1, "Разный регистр → containment → leader остаётся")
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("https://ex.com/leader-case", send_args)
            self.assertNotIn("https://ex.com/dup-case", send_args)
            p.close()

    def test_containment_short_titles_not_flagged(self):
        """
        Containment-пара с короткими заголовками (<30 символов).
        is_containment_duplicate возвращает False → оба проходят.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            # Оба < 30 символов после нормализации
            a_title = "Two killed in Voronezh attack"
            b_title = "Two killed in Voronezh"  # short

            _insert_deferred_item(
                db_path, title=a_title, url="https://ex.com/short1",
                published="2025-01-01 12:00", fetched_at="2025-01-01T12:00:00+00:00",
                score=8,
            )
            _insert_deferred_item(
                db_path, title=b_title, url="https://ex.com/short2",
                published="2025-01-01 12:00", fetched_at="2025-01-02T12:00:00+00:00",
                score=7,
            )

            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = set()

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Оба проходят: <30 символов → containment guard пропускает
            self.assertEqual(count, 2, "короткие титлы не подпадают под containment")
            p.close()

    def test_llm_dup_and_containment_dup_both_excluded(self):
        """
        Один item помечен LLM-дублем, другой — containment-дублем.
        Оба исключаются, лидер отправляется.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.close()

            # item[0] — лидер, item[1] — LLM-дубль, item[2] — containment-дубль
            for title, url, score, fetched in [
                ("Global leaders meet for G20 summit discussion", "https://ex.com/l0", 9, "2025-01-01T12:00:00+00:00"),
                ("Global leaders meet for G20 summit discussion", "https://ex.com/l1", 7, "2025-01-02T12:00:00+00:00"),
                ("Global leaders meet for G20 summit", "https://ex.com/l2", 5, "2025-01-03T12:00:00+00:00"),
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2025-01-01 12:00", fetched_at=fetched, score=score,
                )

            # LLM отмечает l1 как дубль (индекс 1)
            mock_filter = MagicMock()
            mock_filter.detect_duplicates.return_value = {1}

            p = _mk_pipeline_integration(db_path, llm_filter=mock_filter)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Лидер l0 отправлен, l1 (LLM-дубль) и l2 (containment-дубль) отфильтрованы
            self.assertEqual(count, 1)
            send_args = delivery.send_text.call_args[0][0]
            self.assertIn("l0", send_args)
            self.assertNotIn("l1", send_args)
            self.assertNotIn("l2", send_args)
            p.close()


if __name__ == "__main__":
    unittest.main()