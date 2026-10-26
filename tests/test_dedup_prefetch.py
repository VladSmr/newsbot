"""
Регрессионные тесты для Major#10-фикса: prefetch-оптимизация _domain_recent_duplicate.

Проверяют, что:
  1. Prefetch-кеш используется (filter_new вызывает get_recent_for_dedup 1 раз).
  2. Результаты дедупа идентичны оригинальной логике на эталонных кейсах.
  3. Fail-open сохраняется (ошибка → не дубль).
  4. Старая логика (без кэша) и новая дают одинаковый результат.

Запускать: python -m pytest tests/test_dedup_prefetch.py -v
"""
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys as _sys
_root = Path(__file__).parent.parent
_sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.dedup import filter_new, _domain_recent_duplicate
from newsbot.storage import Storage


def _mk_item(
    source="TestSource",
    title="Test Title",
    url=None,
    summary="Test summary text",
    published="01.01.2025 12:00 UTC",
    cluster_key=None,
    **kwargs,
) -> NewsItem:
    if url is None:
        url = f"https://example.com/{hash(title) + id(title)}"
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary=summary,
        published=published,
        published_parsed=None,
        cluster_key=cluster_key,
        **kwargs,
    )


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _cleanup_db(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        time.sleep(0.2)
        try:
            os.unlink(path)
        except (PermissionError, FileNotFoundError):
            pass


# =============================================================================
# Эталонные кейсы дедупа: результат должен быть идентичен старой логике
# =============================================================================

class TestPrefetchDedupSemantics(unittest.TestCase):
    """Семантика дедупа с prefetch должна быть идентична оригиналу."""

    def test_same_cluster_key_in_window_is_duplicate(self):
        """
        Оригинал: item с тем же cluster_key (суффикс) в 2-часовом окне — дубль.
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Вставляем item с cluster_key
            existing = _mk_item(
                title="Fed cuts interest rates by 25 bps",
                source="Reuters",
                url="https://reuters.com/fed-cuts",
                cluster_key="other|abc12345|2026-10-21",
            )
            st.save(existing)

            # Кандидат с тем же суффиксом (другой URL)
            candidate = _mk_item(
                title="Fed cuts interest rates by 25 bps",
                source="Bloomberg",
                url="https://bloomberg.com/fed-cuts-25",
                cluster_key="moscow|abc12345|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 1,
                "Тот же cluster_key (суффикс) → дубль")
            self.assertEqual(len(new_items), 0,
                "Дубль не в new_items")
            st.close()
        finally:
            _cleanup_db(path)

    def test_different_cluster_key_not_duplicate(self):
        """
        Оригинал: разные cluster_key (суффикс) — НЕ дубль.
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Fed cuts rates",
                source="Reuters",
                url="https://reuters.com/fed",
                cluster_key="other|abc11111|2026-10-21",
            )
            st.save(existing)

            # Другой cluster_key
            candidate = _mk_item(
                title="ECB raises rates",
                source="Bloomberg",
                url="https://bloomberg.com/ecb",
                cluster_key="other|def22222|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "Разные cluster_key → НЕ дубль")
            self.assertEqual(len(new_items), 1, "Кандидат проходит")
            st.close()
        finally:
            _cleanup_db(path)

    def test_number_guard_same_numbers_duplicate(self):
        """
        Оригинал: тот же cluster_key, одинаковые числа в title → дубль.
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Bitcoin reaches 70000 dollars",
                source="CoinDesk",
                url="https://coindesk.com/btc-70k",
                cluster_key="other|stem999|2026-10-21",
            )
            st.save(existing)

            candidate = _mk_item(
                title="Bitcoin reaches 70000 dollars",
                source="CoinTelegraph",
                url="https://cointelegraph.com/btc",
                cluster_key="moscow|stem999|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 1,
                "Тот же cluster_key + те же числа → дубль")
            st.close()
        finally:
            _cleanup_db(path)

    def test_number_guard_different_numbers_not_duplicate(self):
        """
        Оригинал: тот же cluster_key, разные числа → НЕ дубль (числовой guard).
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Bitcoin reaches 70000 dollars",
                source="CoinDesk",
                url="https://coindesk.com/btc-70k",
                cluster_key="other|stem123|2026-10-21",
            )
            st.save(existing)

            # Разные числа
            candidate = _mk_item(
                title="Bitcoin reaches 80000 dollars",
                source="CoinTelegraph",
                url="https://cointelegraph.com/btc-80k",
                cluster_key="moscow|stem123|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                "Тот же cluster_key, РАЗНЫЕ числа → НЕ дубль")
            self.assertEqual(len(new_items), 1,
                "Кандидат с другими числами проходит через числовой guard")
            st.close()
        finally:
            _cleanup_db(path)

    def test_outside_window_not_duplicate(self):
        """
        Оригинал: item за пределами 2-часового окна — НЕ дубль.
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Fed rate news",
                source="Reuters",
                url="https://reuters.com/fed-old",
                cluster_key="other|oldstem|2026-10-21",
            )
            st.save(existing)

            # Подвигаем fetched_at на 3 часа назад (за пределами окна)
            conn = sqlite3.connect(path)
            old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
            conn.execute(
                "UPDATE items SET fetched_at = ? WHERE url = ?",
                (old_time, existing.url),
            )
            conn.commit()
            conn.close()

            candidate = _mk_item(
                title="Fed rate decision",
                source="Bloomberg",
                url="https://bloomberg.com/fed-new",
                cluster_key="other|oldstem|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                "Item за пределами 2-часового окна → НЕ дубль")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)

    def test_old_format_cluster_key_exact_match(self):
        """
        Оригинал: старый cluster_key без '|' — точное совпадение.
        Prefetch: тот же результат.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Old Title",
                source="Reuters",
                url="https://reuters.com/old",
                cluster_key="cb_rate_2024",  # старый формат
            )
            st.save(existing)

            candidate = _mk_item(
                title="Old Title",
                source="Bloomberg",
                url="https://bloomberg.com/old",
                cluster_key="cb_rate_2024",  # то же, без |
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 1, "Старый формат: точное совпадение → дубль")
            st.close()
        finally:
            _cleanup_db(path)

    def test_no_cluster_key_passes_through(self):
        """Без cluster_key — guard пропускает item дальше."""
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Вставляем item с cluster_key
            existing = _mk_item(
                title="Some news",
                source="Reuters",
                url="https://reuters.com/some",
                cluster_key="other|xyz999|2026-10-21",
            )
            st.save(existing)

            # Кандидат БЕЗ cluster_key
            candidate = _mk_item(
                title="Some news",
                source="Bloomberg",
                url="https://bloomberg.com/some",
                cluster_key=None,  # нет cluster_key
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "Без cluster_key → НЕ дубль")
            self.assertEqual(len(new_items), 1)
            st.close()
        finally:
            _cleanup_db(path)


class TestPrefetchCacheUsed(unittest.TestCase):
    """Prefetch-кеш: filter_new вызывает get_recent_for_dedup ровно 1 раз."""

    def test_get_recent_for_dedup_called_once(self):
        """
        filter_new с N items (у которых есть cluster_key) вызывает
        get_recent_for_dedup ровно 1 раз, а не N раз.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Вставляем items с cluster_key
            for i in range(10):
                item = _mk_item(
                    title=f"Existing News {i}",
                    source="Reuters",
                    url=f"https://reuters.com/ex{i}",
                    cluster_key=f"other|existing{i}|2026-10-21",
                )
                st.save(item)

            # Кандидаты
            candidates = [
                _mk_item(
                    title=f"Candidate {i}",
                    source="Bloomberg",
                    url=f"https://bloomberg.com/cand{i}",
                    cluster_key=f"other|existing{i}|2026-10-21",  # совпадает
                )
                for i in range(10)
            ]

            # Мокаем get_recent_for_dedup
            original = getattr(st, "get_recent_for_dedup", None)
            call_count = [0]

            def counting_wrapper(*args, **kwargs):
                call_count[0] += 1
                if original:
                    return original(*args, **kwargs)
                return []

            st.get_recent_for_dedup = counting_wrapper

            new_items, total, dup = filter_new(candidates, st, dedup_config=cfg)

            self.assertEqual(call_count[0], 1,
                f"get_recent_for_dedup должна вызываться ровно 1 раз, "
                f"вызвана {call_count[0]} раз. "
                "Prefetch-оптимизация сломана — каждая итерация делает свой запрос!")

            st.close()
        finally:
            _cleanup_db(path)


class TestPrefetchFailOpen(unittest.TestCase):
    """Fail-open: prefetch=None → guard пропускает item (не падает)."""

    def test_prefetch_none_fails_open(self):
        """
        При prefetch=None (ошибка получения) _domain_recent_duplicate
        возвращает False (не дубль), сохраняя fail-open инвариант.
        """
        item = _mk_item(
            title="Fed news",
            source="Reuters",
            url="https://reuters.com/fed",
            cluster_key="other|test123|2026-10-21",
        )

        # prefetch=None → fail-open
        result = _domain_recent_duplicate(item, None)

        self.assertFalse(result,
            "prefetch=None → fail-open: не дубль")

    def test_filter_new_with_broken_prefetch_continues(self):
        """
        filter_new продолжает работу при ошибке prefetch (не падает).
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item = _mk_item(
                title="Test item",
                source="A",
                url="https://a.com/test",
                cluster_key="other|testkey|2026-10-21",
            )
            st.save(item)

            # Мокаем get_recent_for_dedup так, чтобы он упал
            def bad_get(*args, **kwargs):
                raise RuntimeError("DB is corrupt")

            st.get_recent_for_dedup = bad_get

            # Не должно упасть
            try:
                new_items, total, dup = filter_new(
                    [_mk_item(
                        title="Another",
                        source="B",
                        url="https://b.com/another",
                        cluster_key="other|testkey|2026-10-21",
                    )],
                    st, dedup_config=cfg,
                )
            except Exception as e:
                self.fail(f"filter_new не должен падать при ошибке prefetch: {e}")

            st.close()
        finally:
            _cleanup_db(path)


class TestPrefetchNoFalseNegatives(unittest.TestCase):
    """Prefetch не должен пропускать реальные дубли (false negatives)."""

    def test_multiple_candidates_with_same_cluster_key(self):
        """
        Несколько кандидатов с одинаковым cluster_key — все отклоняются.
        Без чисел в заголовках числовой guard пропускает (оба Counter() пустые).
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Существующий item — без чисел в заголовке
            existing = _mk_item(
                title="EUR/USD analysis and forecast",
                source="Reuters",
                url="https://reuters.com/eurusd",
                cluster_key="other|eurusd09|2026-10-21",
            )
            st.save(existing)

            # 5 кандидатов с тем же cluster_key, без чисел в заголовках.
            # NOTE: итерационные номера НЕ в заголовке (числовой guard!).
            titles = [
                "EUR/USD market update and financial analysis",
                "EUR/USD latest news and trading review",
                "EUR/USD trading signals and forecasts",
                "EUR/USD currency pair assessment",
                "EUR/USD weekly outlook and strategy",
            ]
            candidates = [
                _mk_item(
                    title=titles[i],
                    source=f"Source{i}",
                    url=f"https://s{i}.com/eur",
                    cluster_key=f"moscow|eurusd09|2026-10-21",  # суффикс совпадает
                )
                for i in range(5)
            ]

            new_items, total, dup = filter_new(candidates, st, dedup_config=cfg)

            # Все 5 — дубли (суффикс совпадает, числа оба пустые → совпадают)
            self.assertEqual(dup, 5,
                f"Все 5 кандидатов с тем же cluster_key должны быть дублями, "
                f"dup={dup}")
            self.assertEqual(len(new_items), 0,
                f"new_items должен быть пустым, got {len(new_items)}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_mixed_candidates(self):
        """
        3 дубликата (тот же cluster_key, без чисел) + 2 новых (другой cluster_key).
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Без чисел: числовой guard пропускает (оба пустые Counter)
            existing = _mk_item(
                title="Gold market analysis and forecast",
                source="Reuters",
                url="https://reuters.com/gold",
                cluster_key="other|gold2500|2026-10-21",
            )
            st.save(existing)

            candidates = [
                # Дубли (тот же cluster_key, без чисел)
                _mk_item(
                    title="Gold price update and summary one",
                    source="Bloomberg",
                    url="https://bloomberg.com/gold1",
                    cluster_key="moscow|gold2500|2026-10-21",
                ),
                _mk_item(
                    title="Gold price update and summary two",
                    source="CNBC",
                    url="https://cnbc.com/gold2",
                    cluster_key="other|gold2500|2026-10-21",
                ),
                _mk_item(
                    title="Gold market analysis and overview",
                    source="WSJ",
                    url="https://wsj.com/gold3",
                    cluster_key="moscow|gold2500|2026-10-21",
                ),
                # Новые (другой cluster_key)
                _mk_item(
                    title="Silver rises sharply in Asian markets",
                    source="Reuters",
                    url="https://reuters.com/silver",
                    cluster_key="other|silver30|2026-10-21",
                ),
                _mk_item(
                    title="Oil price update and market impact",
                    source="Bloomberg",
                    url="https://bloomberg.com/oil",
                    cluster_key="other|oil80|2026-10-21",
                ),
            ]

            new_items, total, dup = filter_new(candidates, st, dedup_config=cfg)

            self.assertEqual(dup, 3,
                f"3 дубликата ожидались, dup={dup}")
            self.assertEqual(len(new_items), 2,
                f"2 новых ожидались, new_items={len(new_items)}")
            st.close()
        finally:
            _cleanup_db(path)


class TestPrefetchMultipleCandidatesSameSuffix(unittest.TestCase):
    """
    Prefetch хранит list[candidates]: guard считает дублем при совпадении
    ХОТЯ БЫ С ОДНИМ из кандидатов (OR semantics).
    """

    def test_same_suffix_different_numbers_match_one_is_duplicate(self):
        """
        Два DB-item с одинаковым суффиксом, но разными числами в title.
        Новый item совпадает по числам с ПЕРВЫМ кандидатом → дубль.
        Prefetch хранит list, guard итерирует по всем и находит совпадение.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Item A: cluster_key совпадает, числа 80000
            item_a = _mk_item(
                title="Bitcoin reaches 80000 dollars today",
                source="CoinDesk",
                url="https://coindesk.com/btc-80k",
                cluster_key="other|stem123|2026-10-21",
            )
            st.save(item_a)

            # Item B: тот же суффикс, числа 70000 (сохранён позже — rowid больше)
            item_b = _mk_item(
                title="Bitcoin reaches 70000 dollars today",
                source="CoinTelegraph",
                url="https://cointelegraph.com/btc-70k",
                cluster_key="moscow|stem123|2026-10-21",
            )
            st.save(item_b)

            # Кандидат: те же числа что у item_a (80000) → совпадает с одним
            # из кандидатов → дубль. Раньше (single-entry prefetch, last-write-wins)
            # мог остаться только item_b → числовой guard видел 70000 → не дубль.
            candidate = _mk_item(
                title="Bitcoin reaches 80000 dollars update",
                source="Bloomberg",
                url="https://bloomberg.com/btc-80k-v2",
                cluster_key="other|stem123|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 1,
                "Совпадение с ОДНИМ из нескольких DB-candidates → дубль (OR semantics)")
            self.assertEqual(len(new_items), 0,
                "Кандидат, совпавший с одним из DB-items, не попадает в new_items")
            st.close()
        finally:
            _cleanup_db(path)

    def test_same_suffix_different_numbers_no_match_not_duplicate(self):
        """
        Два DB-item с одинаковым суффиксом, разные числа. Новый item с ЧУЖИМИ
        числами — не совпадает ни с одним кандидатом → НЕ дубль.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item_a = _mk_item(
                title="Bitcoin reaches 70000 dollars",
                source="CoinDesk",
                url="https://coindesk.com/btc-70k",
                cluster_key="other|stem456|2026-10-21",
            )
            st.save(item_a)

            item_b = _mk_item(
                title="Bitcoin reaches 80000 dollars",
                source="CoinTelegraph",
                url="https://cointelegraph.com/btc-80k",
                cluster_key="moscow|stem456|2026-10-21",
            )
            st.save(item_b)

            # Кандидат: числа 90000 — не совпадают ни с 70000, ни с 80000
            candidate = _mk_item(
                title="Bitcoin reaches 90000 dollars",
                source="Bloomberg",
                url="https://bloomberg.com/btc-90k",
                cluster_key="other|stem456|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                "Числа не совпадают ни с одним из DB-candidates → НЕ дубль")
            self.assertEqual(len(new_items), 1,
                "Кандидат с новыми числами проходит числовой guard")
            st.close()
        finally:
            _cleanup_db(path)

    def test_different_suffixes_all_pass_through(self):
        """
        Несколько DB-items с разными суффиксами — ни один не совпадает по key
        с кандидатом → НЕ дубль.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            st.save(_mk_item(
                title="Fed raises rates by 25 bps",
                source="Reuters",
                url="https://reuters.com/fed25",
                cluster_key="other|abc111|2026-10-21",
            ))
            st.save(_mk_item(
                title="Fed raises rates by 50 bps",
                source="FT",
                url="https://ft.com/fed50",
                cluster_key="moscow|def222|2026-10-21",
            ))
            st.save(_mk_item(
                title="ECB holds rates steady",
                source="Bloomberg",
                url="https://bloomberg.com/ecb",
                cluster_key="other|ghi333|2026-10-21",
            ))

            # Кандидат с НОВЫМ cluster_key
            candidate = _mk_item(
                title="BoJ cuts rates unexpectedly",
                source="Nikkei",
                url="https://nikkei.com/boj",
                cluster_key="other|xyz999|2026-10-21",
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0, "Новый cluster_key → НЕ дубль")
            self.assertEqual(len(new_items), 1,
                "Кандидат с новым cluster_key проходит")
            st.close()
        finally:
            _cleanup_db(path)