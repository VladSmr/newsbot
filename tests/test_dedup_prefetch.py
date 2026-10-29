"""
Регрессионные тесты для Major#10-фикса: prefetch-оптимизация _domain_recent_duplicate_legacy.

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
from newsbot.dedup import filter_new, _domain_recent_duplicate_legacy
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
        При prefetch=None (ошибка получения) _domain_recent_duplicate_legacy
        возвращает False (не дубль), сохраняя fail-open инвариант.
        """
        item = _mk_item(
            title="Fed news",
            source="Reuters",
            url="https://reuters.com/fed",
            cluster_key="other|test123|2026-10-21",
        )

        # prefetch=None → fail-open
        result = _domain_recent_duplicate_legacy(item, None)

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
        Несколько кандидатов с одинаковым cluster_key и тем же началом
        (first-2 tokens) — все отклоняются через Major-2 (within-cycle).

        Баг до Major-2: любой item с тем же cluster_key блокировался
        БЕЗ проверки содержания, даже если заголовки были о РАЗНЫХ темах.
        После Major-2: first-2 tokens — дешёвый способ отличить тему.

        В этом тесте все 5 кандидатов имеют first-2 = ('eur/usd', 'rate'),
        то же что existing → все дубли по first-2 + cluster_key.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Существующий item — first-2 = ('eur/usd', 'rate'), без чисел
            existing = _mk_item(
                title="EUR/USD rate forecast and outlook",
                source="Reuters",
                url="https://reuters.com/eurusd",
                cluster_key="other|eurusd09|2026-10-21",
            )
            st.save(existing)

            # 5 кандидатов с тем же cluster_key, все first-2 = ('eur/usd', 'rate')
            # (то же что у existing) → все должны быть dup.
            titles = [
                "EUR/USD rate update and market analysis",
                "EUR/USD rate forecast and trading review",
                "EUR/USD rate signals and currency forecast",
                "EUR/USD rate assessment and pair outlook",
                "EUR/USD rate outlook and weekly strategy",
            ]
            candidates = [
                _mk_item(
                    title=titles[i],
                    source=f"Source{i}",
                    url=f"https://s{i}.com/eur",
                    cluster_key=f"moscow|eurusd09|2026-10-21",
                )
                for i in range(5)
            ]

            new_items, total, dup = filter_new(candidates, st, dedup_config=cfg)

            # Все 5 — дубли (тот же cluster_key + first-2 tokens совпадают).
            self.assertEqual(dup, 5,
                f"Все 5 кандидатов с тем же cluster_key и first-2 tokens "
                f"должны быть дублями, dup={dup}")
            self.assertEqual(len(new_items), 0,
                f"new_items должен быть пустым, got {len(new_items)}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_mixed_candidates(self):
        """
        5 candidates: 2 different topics (Bitcoin/Gold) + 1 repeated topic.
        Domain dedup checks: same cluster_key + same first-2 + same numbers = dup.
        - existing: "Bitcoin reaches 80000 dollars today"
          -> first-2=('bitcoin','reaches'), numbers={80000}
        - candidate1: "Bitcoin drops to 70000 amid selloff"
          -> first-2=('bitcoin','drops') != existing -> NEW
        - candidate2: "Bitcoin reaches 70000 today"
          -> first-2=('bitcoin','reaches') == existing, but numbers {70000} != {80000} -> NEW
        - candidate3: "Bitcoin reaches 80000 dollars"
          -> first-2==existing, numbers==existing -> DUP
        - candidate4/5: different cluster_key + topic -> NEW

        Expected: dup=1 (only candidate3), new_items=4.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title="Bitcoin reaches 80000 dollars today",
                source="CoinDesk",
                url="https://coindesk.com/btc80k",
                cluster_key="other|stem123|2026-10-21",
            )
            st.save(existing)

            candidates = [
                # Другая тема: числа иные → first-2 не совпадает с existing
                _mk_item(
                    title="Bitcoin drops to 70000 amid selloff",
                    source="Bloomberg",
                    url="https://bloomberg.com/btc70k",
                    cluster_key="moscow|stem123|2026-10-21",
                ),
                # Другая тема: тот же предмет, другие числа → first-2 отличается
                _mk_item(
                    title="Bitcoin reaches 70000 today",
                    source="CoinTelegraph",
                    url="https://cointelegraph.com/btc70ktoday",
                    cluster_key="moscow|stem123|2026-10-21",
                ),
                # Та же тема: first-2 совпадает с existing
                _mk_item(
                    title="Bitcoin reaches 80000 dollars",
                    source="Reuters",
                    url="https://reuters.com/btc80kexact",
                    cluster_key="moscow|stem123|2026-10-21",
                ),
                # Другая тема: Gold, другой cluster_key
                _mk_item(
                    title="Gold rises sharply in Asian markets",
                    source="Bloomberg",
                    url="https://bloomberg.com/goldrise",
                    cluster_key="other|gold99|2026-10-21",
                ),
                # Другая тема: Gold, другой cluster_key
                _mk_item(
                    title="Gold update and market analysis",
                    source="Reuters",
                    url="https://reuters.com/goldupdate",
                    cluster_key="other|gold99|2026-10-21",
                ),
            ]

            new_items, total, dup = filter_new(candidates, st, dedup_config=cfg)

            # Только candidate3 — дубль (то же first-2 что existing).
            # 1,2 — разные темы (разные first-2); 4,5 — другая тема + cluster_key.
            self.assertEqual(dup, 1,
                f"Только 1 дубль (first-2 совпадает с existing), dup={dup}")
            self.assertEqual(len(new_items), 4,
                f"4 новых ожидались, new_items={len(new_items)}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_opposite_word_order_not_duplicate(self):
        """
        Major-2 негативный: «Russia strikes Ukraine» и «Ukraine strikes Russia»
        имеют РАЗНЫЕ first-2 tokens → НЕ дубли, оба проходят.

        Без Major-2 (старая семантика cluster_key): оба давали один
        cluster_key → первый блокировал второй навсегда → новость потеряна.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            item1 = _mk_item(
                title="Russia strikes Ukraine with new missiles",
                source="Reuters",
                url="https://reuters.com/rs1",
                cluster_key="moscow|war123|2026-10-21",
            )
            item2 = _mk_item(
                title="Ukraine strikes Russia with new missiles",
                source="TASS",
                url="https://tass.com/rs2",
                cluster_key="moscow|war123|2026-10-21",
            )

            new_items, total, dup = filter_new([item1, item2], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                f"Противоположный порядок слов → НЕ дубль, dup={dup}")
            self.assertEqual(len(new_items), 2,
                f"Обе новости должны пройти, got {len(new_items)}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_source_prefix_stripped_before_first2(self):
        """
        MAJOR-2: source-prefix ('Reuters: ...') stripped from title
        before first-2 computation. Without it 'Reuters: Fed cuts rates'
        gives first-2=('reuters:','fed'), but 'Fed cuts rates' ->
        first-2=('fed','cuts') -> duplicate not detected.
        """
        path = _tmp_db()
        try:
            cfg = {'cross_source': False, 'fuzzy': False, 'window_days': 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title='Reuters: Fed cuts interest rates by 25 bps',
                source='Reuters',
                url='https://reuters.com/fed25',
                cluster_key='other|fed123|2026-10-21',
            )
            st.save(existing)

            candidate = _mk_item(
                title='Fed cuts interest rates by 25 bps',
                source='Bloomberg',
                url='https://bloomberg.com/fed25',
                cluster_key='moscow|fed123|2026-10-21',
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 1,
                'Source prefix stripped: both give same first-2 -> dup, dup=%s' % dup)
            self.assertEqual(len(new_items), 0)
            st.close()
        finally:
            _cleanup_db(path)

    def test_same_first2_different_numbers_prefetch_path(self):
        """
        Prefetch-path: existing item in DB, candidate with SAME first-2
        but DIFFERENT numbers → NOT duplicate (numbers guard fires).
        This is the prefetch dedup path (not within-cycle).
        """
        path = _tmp_db()
        try:
            cfg = {'cross_source': False, 'fuzzy': False, 'window_days': 3}
            st = Storage(Path(path), dedup_config=cfg)

            existing = _mk_item(
                title='Bitcoin reaches 80000 dollars today',
                source='CoinDesk',
                url='https://coindesk.com/btc80k',
                cluster_key='other|stem123|2026-10-21',
            )
            st.save(existing)

            candidate = _mk_item(
                title='Bitcoin reaches 90000 dollars update',
                source='Bloomberg',
                url='https://bloomberg.com/btc90k',
                cluster_key='moscow|stem123|2026-10-21',
            )

            new_items, total, dup = filter_new([candidate], st, dedup_config=cfg)
            self.assertEqual(dup, 0,
                'Same first-2 but different numbers -> NOT dup, dup=%s' % dup)
            self.assertEqual(len(new_items), 1,
                'Candidate with different numbers passes domain guard')
            st.close()
        finally:
            _cleanup_db(path)

    def test_same_first2_different_numbers_domain_dedup(self):
        """
        Domain dedup (prefetch path): existing item with same cluster_key
        and same first-2 but different numbers → numbers guard blocks dedup,
        candidate passes to new_items.
        """
        path = _tmp_db()
        try:
            cfg = {'cross_source': False, 'fuzzy': False, 'window_days': 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Item 80000 — saved to DB (establishes prefetch cache)
            item_80000 = _mk_item(
                title='Bitcoin reaches 80000 dollars today',
                source='CoinDesk',
                url='https://coindesk.com/btc80k',
                cluster_key='btc|rate|2026-10-21',
            )
            st.save(item_80000)

            # Item 90000 — candidate, same first-2 ('bitcoin', 'reaches'),
            # different numbers → numbers guard blocks domain dedup → NOT duplicate
            item_90000 = _mk_item(
                title='Bitcoin reaches 90000 dollars update',
                source='CoinDesk',
                url='https://coindesk.com/btc90k',
                cluster_key='btc|rate|2026-10-21',
            )

            # After st.save(), item_80000 is in seen_titles. When item_80000 is
            # passed to filter_new, is_duplicate → True (self-dup), counted as dup=1.
            # item_90000 passes: prefetch has item_80000 with same first-2 but
            # different numbers → domain dedup skips it, not in seen_titles → new.
            new_items, total, dup = filter_new(
                [item_80000, item_90000], st, dedup_config=cfg)
            # item_80000: is_duplicate=True (self in seen_titles after st.save)
            # item_90000: prefetch dedup checks first-2+numbers, different numbers → new
            self.assertEqual(dup, 1,
                'item_80000 self-dup (seen_titles), item_90000 not dup (diff nums), dup=%s' % dup)
            self.assertEqual(len(new_items), 1,
                'Only item_90000 passes: new_items=%d' % len(new_items))
            self.assertEqual(new_items[0].title,
                             'Bitcoin reaches 90000 dollars update')
            st.close()
        finally:
            _cleanup_db(path)

    def test_real_reprint_caught_by_fuzzy(self):
        """
        MAJOR-1: fuzzy dedup catches semantic reprints when domain dedup misses.
        - cross_source=True → fuzzy_enabled=True (dedup.py:664-667)
        - Same source → candidate sees existing in _recent_cache[source]
        - Pair: ('ECB raises interest rates by 50 bps in surprise move',
                 'ECB raises interest rates by 50 bps in a surprise move')
          Same first-2 ('ecb', 'raises') + same numbers → domain dedup fires first.
        - At threshold=0.85: ratio=0.984 ≥ 0.85, jaccard=1.0 ≥ 0.8 → fuzzy catches.
        - CONTRAST: same pair with threshold=0.995 (above ratio) → fuzzy misses,
          but domain dedup STILL fires (same first-2, same numbers) → dup=1.
          This proves BOTH domain and fuzzy paths independently.
        """
        path = _tmp_db()
        try:
            cfg_base = {'cross_source': True, 'window_days': 3}
            st = Storage(Path(path), dedup_config=cfg_base)

            # Existing item in DB
            existing = _mk_item(
                title='ECB raises interest rates by 50 basis points in surprise move',
                source='CoinDesk',
                url='https://coindesk.com/ecb50',
                cluster_key='other|ecb50|2026-10-21',
            )
            st.save(existing)

            # Candidate: same story, same first-2, same numbers, tiny diff ('in' vs 'in a')
            # Ratio=0.984, jaccard=1.0 — passes both token_guard and ratio checks
            candidate = _mk_item(
                title='ECB raises interest rates by 50 basis points in a surprise move',
                source='CoinDesk',  # SAME source: _recent_cache works
                url='https://coindesk.com/ecb50b',
                cluster_key='other|ecb50|2026-10-21',
            )

            # Part 1: threshold=0.85 → fuzzy detects (and domain dedup also fires)
            new_items, total, dup = filter_new(
                [candidate], st, dedup_config={**cfg_base, 'fuzzy': True,
                                               'threshold': 0.85, 'token_threshold': 0.8})
            self.assertEqual(dup, 1,
                'Fuzzy detects (ratio=0.984≥0.85, jaccard=1.0≥0.8), dup=%s' % dup)
            self.assertEqual(len(new_items), 0)

            # Part 2: threshold=0.995 → fuzzy MISSES (0.984 < 0.995),
            # but domain dedup STILL catches (same first-2 + numbers).
            # This is the domain-path proof: dedup works even when fuzzy misses.
            new_items2, _, dup2 = filter_new(
                [candidate], st, dedup_config={**cfg_base, 'fuzzy': True,
                                               'threshold': 0.995, 'token_threshold': 0.8})
            self.assertEqual(dup2, 1,
                'Fuzzy misses (ratio=0.984<0.995) but DOMAIN dedup fires, dup=%s' % dup2)

            # Part 3: fuzzy=False, different cluster_key → domain dedup fires,
            # proving domain path independently.
            new_items3, _, dup3 = filter_new(
                [candidate], st, dedup_config={**cfg_base, 'fuzzy': False,
                                               'threshold': 0.85})
            self.assertEqual(dup3, 1,
                'fuzzy=False: domain dedup fires (same first-2, numbers), dup=%s' % dup3)

            st.close()
        finally:
            _cleanup_db(path)

    def test_within_cycle_same_topic_is_duplicate(self):
        """
        Major-2 positive: внутри одного цикла items с одинаковой темой
        (first-2 tokens совпадают) и одинаковым cluster_key склеиваются.
        Первый — new, второй — skipped.

        NOTE: в prefetch нет ничего (fresh DB), работает only within-cycle dedup.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Первый item — тема Bitcoin rate, first-2 = ('bitcoin', 'rate')
            item1 = _mk_item(
                title="Bitcoin rate surges past 100000 dollars",
                source="CoinDesk",
                url="https://coindesk.com/btc100k",
                cluster_key="moscow|btc100k|2026-10-21",
            )
            # Второй item — та же тема (first-2 = ('bitcoin', 'rate') совпадает),
            # другое время/URL
            item2 = _mk_item(
                title="Bitcoin rate breaks above 100000-dollar mark",
                source="CoinTelegraph",
                url="https://cointelegraph.com/btc100kv2",
                cluster_key="moscow|btc100k|2026-10-21",
            )

            new_items, total, dup = filter_new([item1, item2], st, dedup_config=cfg)
            self.assertEqual(dup, 1,
                f"Внутри цикла: та же тема (first-2 совпадает) → дубль, dup={dup}")
            self.assertEqual(len(new_items), 1,
                f"Только первый item новый, got {len(new_items)}")
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


class TestPrefetchWindowHoursConfig(unittest.TestCase):
    """MAJOR-prefetch-фикс: dedup.prefetch_window_hours управляет prefetch-окном."""

    def test_prefetch_window_from_config(self):
        """
        dedup.prefetch_window_hours=3 → get_recent_for_dedup вызывается с hours=3.
        Поведенческий тест: мокаем get_recent_for_dedup и проверяем переданный hours.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3,
                   "prefetch_window_hours": 3}
            st = Storage(Path(path), dedup_config=cfg)

            # Мокаем: запоминаем, с каким hours вызван
            received_hours = []
            original = st.get_recent_for_dedup

            def spy(*args, **kwargs):
                received_hours.append(kwargs.get("hours"))
                return original(*args, **kwargs)

            st.get_recent_for_dedup = spy

            candidate = _mk_item(
                title="Test item", source="A",
                url="https://a.com/test",
                cluster_key="other|test123|2026-10-21",
            )
            filter_new([candidate], st, dedup_config=cfg)

            self.assertEqual(received_hours, [3],
                f"get_recent_for_dedup должна быть вызвана с hours=3, "
                f"получено hours={received_hours}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_prefetch_window_default_is_2(self):
        """
        dedup.prefetch_window_hours отсутствует → get_recent_for_dedup
        вызывается с hours=2 (обратная совместимость).
        """
        path = _tmp_db()
        try:
            # НЕТ prefetch_window_hours в конфиге
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3}
            st = Storage(Path(path), dedup_config=cfg)

            received_hours = []
            original = st.get_recent_for_dedup

            def spy(*args, **kwargs):
                received_hours.append(kwargs.get("hours"))
                return original(*args, **kwargs)

            st.get_recent_for_dedup = spy

            candidate = _mk_item(
                title="Test", source="B",
                url="https://b.com/test",
                cluster_key="other|test456|2026-10-21",
            )
            filter_new([candidate], st, dedup_config=cfg)

            self.assertEqual(received_hours, [2],
                f"get_recent_for_dedup должна вызываться с hours=2 по дефолту, "
                f"получено hours={received_hours}")
            st.close()
        finally:
            _cleanup_db(path)

    def test_prefetch_window_garbage_logs_warning_and_defaults_to_2(self):
        """
        dedup.prefetch_window_hours="три" (мусор) → warning + hours=2.
        filter_new не падает, prefetch работает с дефолтом.
        """
        path = _tmp_db()
        try:
            cfg = {"cross_source": False, "fuzzy": False, "window_days": 3,
                   "prefetch_window_hours": "три"}
            st = Storage(Path(path), dedup_config=cfg)

            received_hours = []
            original = st.get_recent_for_dedup

            def spy(*args, **kwargs):
                received_hours.append(kwargs.get("hours"))
                return original(*args, **kwargs)

            st.get_recent_for_dedup = spy

            import logging
            with self.assertLogs("newsbot.dedup", level="WARNING") as cm:
                candidate = _mk_item(
                    title="Test", source="C",
                    url="https://c.com/test",
                    cluster_key="other|test789|2026-10-21",
                )
                filter_new([candidate], st, dedup_config=cfg)

            self.assertTrue(
                any("prefetch_window_hours" in msg for msg in cm.output),
                "Должен быть warning о prefetch_window_hours, got: %s" % cm.output,
            )
            self.assertEqual(received_hours, [2],
                f"При мусоре должен использоваться дефолт hours=2, "
                f"получено hours={received_hours}")
            st.close()
        finally:
            _cleanup_db(path)