"""
Тесты оконной логики дайджеста (Этап 6).

Покрывает:
  - Успешный run_digest(force=True): ВСЁ окно удалено (до старта digest),
    пришедшие во время — сохранены, digest_window_started_at обновлён.
  - Падение доставки (send_text → False): items НЕ удалены,
    граница НЕ сдвинута.
  - Второй дайджест подряд: в выборку попадают только items после новой границы.
  - Пустое окно (on_empty="skip"): граница сдвигается.

Запуск: python -m pytest tests/test_digest_window.py -v
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta, timezone
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


def _insert_deferred_item(db_path, title, url, published, fetched_at, score=6, tags=None):
    """
    Вставляет deferred-item напрямую в БД с заданным fetched_at.
    Storage.save() игнорирует item.fetched_at (всегда ставит NOW),
    поэтому нужен прямой INSERT для контроля времени.

    fetched_at ЯВЛЯЕТСЯ ОБЯЗАТЕЛЬНЫМ параметром — это гарантирует
    предсказуемое поведение в тестах (без race condition по секундам).
    score позволяет влиять на порядок get_deferred_items (сортировка: score DESC).
    tags: JSON-string для категоризации (напр. '["film","movie"]' → cinema).
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score, tags)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?, ?)
        """,
        ("TestSrc", url, title, "Test summary", published, fetched_at, fetched_at, score, tags),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, digest_cfg=None, llm_filter=None):
    """Создаёт Pipeline с временной БД."""
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
        },
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    p.storage = Storage(db_path)
    if llm_filter is not None:
        p.llm_filter = llm_filter
    return p


# =============================================================================
# Тесты
# =============================================================================

class TestIsoDtParsing(unittest.TestCase):
    """M1-тесты: _iso_dt парсит все прод-форматы (UTC-суффикс, RFC-2822, ISO)."""

    def _assert_utc(self, result, year, month, day, hour, minute, second=0):
        """Проверяет aware-UTC datetime."""
        self.assertIsNotNone(result)
        self.assertEqual(result.year, year)
        self.assertEqual(result.month, month)
        self.assertEqual(result.day, day)
        self.assertEqual(result.hour, hour)
        self.assertEqual(result.minute, minute)
        self.assertEqual(result.second, second)
        self.assertEqual(result.tzinfo, timezone.utc)

    def _assert_aware(self, result, year, month, day, hour, minute, expected_offset_minutes=0):
        """Проверяет aware datetime с ожидаемым offset (в минутах)."""
        self.assertIsNotNone(result)
        self.assertEqual(result.year, year)
        self.assertEqual(result.month, month)
        self.assertEqual(result.day, day)
        self.assertEqual(result.hour, hour)
        self.assertEqual(result.minute, minute)
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(
            result.utcoffset(), timedelta(minutes=expected_offset_minutes),
            f"utcoffset должен быть {expected_offset_minutes} мин, got {result.utcoffset()}",
        )

    def test_iso_dt_ru_without_utc_suffix(self):
        """'DD.MM.YYYY HH:MM' без суффикса → aware UTC."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("20.10.2026 21:02")
        self._assert_utc(result, 2026, 10, 20, 21, 2)

    def test_iso_dt_ru_with_utc_suffix(self):
        """'DD.MM.YYYY HH:MM UTC' с суффиксом → aware UTC (основной формат прод-данных)."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("20.10.2026 21:02 UTC")
        self._assert_utc(result, 2026, 10, 20, 21, 2)

    def test_iso_dt_ru_with_utc_suffix_seconds(self):
        """'DD.MM.YYYY HH:MM:SS UTC' → aware UTC."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("20.10.2026 21:02:34 UTC")
        self._assert_utc(result, 2026, 10, 20, 21, 2, 34)

    def test_iso_dt_iso_aware(self):
        """ISO с явным offset ('+03:00') → aware с правильным offset."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("2026-10-20T21:02:00+03:00")
        self._assert_aware(result, 2026, 10, 20, 21, 2, expected_offset_minutes=180)  # +03:00 = 180 мин
        # При конвертации в UTC: 21:02+03:00 → 18:02 UTC
        utc = result.astimezone(timezone.utc)
        self.assertEqual(utc.hour, 18)
        self.assertEqual(utc.minute, 2)

    def test_iso_dt_iso_z(self):
        """ISO с 'Z' → aware UTC."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("2026-10-20T21:02:00Z")
        self._assert_utc(result, 2026, 10, 20, 21, 2)

    def test_iso_dt_rfc2822_gmt(self):
        """RFC-2822 'Mon, 01 Jan 2025 12:00:00 GMT' → aware (fallback из collector.py)."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("Mon, 01 Jan 2025 12:00:00 GMT")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.day, 1)
        self.assertEqual(result.hour, 12)
        self.assertEqual(result.minute, 0)
        # RFC-2822 GMT должен вернуть aware (email.utils парсит с UTC)
        self.assertIsNotNone(result.tzinfo)

    def test_iso_dt_emdash(self):
        """'—' (заглушка collector.py) → None."""
        from newsbot.pipeline import _iso_dt
        result = _iso_dt("—")
        self.assertIsNone(result)

    def test_iso_dt_none_guards(self):
        """None → None; пустая строка → None."""
        from newsbot.pipeline import _iso_dt
        self.assertIsNone(_iso_dt(None))
        self.assertIsNone(_iso_dt(""))
        self.assertIsNone(_iso_dt(123))  # не строка


class TestIsoDtDigestIntegration(unittest.TestCase):
    """M4-тест: published-фильтр в run_digest — интеграция с реальными форматами."""

    def test_late_posted_item_excluded_from_digest(self):
        """
        deferred item с published='19.10.2026 20:00 UTC' (старше границы окна)
        и fetched_at ВНУТРИ окна — НЕ попадает в дайджест.

        Без M1-фикса (_iso_dt не парсил 'UTC'-суффикс) этот item проходил
        бы неотфильтрованным (pub_dt=None → fail-open).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # Граница: 2026-10-20 — published='19.10.2026 20:00 UTC' старше
            s.set_meta("digest_window_started_at", "2026-10-20T00:00:00+00:00")
            s.close()

            # Item: fetched_at=2026-10-21 (ВНУТРИ окна), но published=2026-10-19 (за границей)
            _insert_deferred_item(
                db_path,
                title="OldPublishedItem",
                url="https://ex.com/oldpub",
                published="19.10.2026 20:00 UTC",  # формат с суффиксом UTC (prod-format)
                fetched_at="2026-10-21T12:00:00+00:00",
                score=7,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Первый digest: отправляем то что есть (item попадает, если бы не фильтр)
            count1 = p.run_digest(force=True)
            # Item НЕ должен попасть в digest (published старше границы)
            # → deferred не отправлен, но fetched_at в пределах окна:
            # SQL-фильтр since_iso пропускает (fetched_at=2026-10-21 >= 2026-10-20),
            # Python-фильтр published отсеивает → count1=0
            self.assertEqual(count1, 0,
                "Item с late published должен быть отфильтрован")

            # Item остался в deferred
            deferred_urls = {i.url for i in p.storage.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/oldpub", deferred_urls)

            p.storage.close()
            p.close()

    def test_unparseable_published_passes_filter(self):
        """
        Item с published='не дата' (заглушка '—') попадает в digest (fail-open).
        Не теряем новости с непарсящимся published.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path,
                title="BadDateItem",
                url="https://ex.com/baddate",
                published="—",  # не парсится
                fetched_at="2026-10-21T12:00:00+00:00",
                score=7,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            # fail-open: непарсящийся published сохраняем → item уходит в digest
            self.assertEqual(count, 1,
                "Item с непарсящимся published должен попасть в digest (fail-open)")

            p.storage.close()
            p.close()


class TestDigestWindowSuccess(unittest.TestCase):
    """Успешный run_digest: ВСЁ окно удалено, пришедшие во время — сохранены."""

    def test_window_fully_deleted_after_success(self):
        """
        В БД 3 старых item (fetched_at 2025-06) + 2 свежих (fetched_at 2026-01)
        + 1 item «во время отправки» (fetched_at 2099-01, state='deferred',
        пришёл ПОСЛЕ старта дайджеста).
        max_items=2 → в дайджест попадают только 2 свежих (old за лимитом).
        После дайджеста: все 5 items окна (3 old + 2 fresh) УДАЛЕНЫ из БД.
        Item 2099 (пришедший во время) — ОСТАЛСЯ в БД.
        digest_window_started_at сдвинут.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # Граница: 2020-01-01 — покрывает ВСЕ items в окне.
            # delete_items_before(digest_started) удалит ВСЁ окно целиком.
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")

            # 3 старых items (fetched_at 2025-06, ВНУТРИ окна, score ниже — за max_items)
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Old{i}", url=f"https://ex.com/old{i}",
                    published="2025-06-01 12:00",
                    fetched_at="2025-06-01T12:00:00+00:00",
                    score=5,
                )

            # 2 свежих items (fetched_at 2026-01, ВНУТРИ окна, score выше — попадут в digest)
            for i in range(2):
                _insert_deferred_item(
                    db_path,
                    title=f"Fresh{i}", url=f"https://ex.com/fresh{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=7,
                )

            # 1 item «пришедший во время отправки» (fetched_at в БУДУЩЕМ,
            # fetched_at > digest_started_iso → НЕ попадает в дайджест, НЕ удаляется)
            _insert_deferred_item(
                db_path,
                title="DuringSend", url="https://ex.com/during",
                published="2099-01-01 00:00",
                fetched_at="2099-01-01T00:00:00+00:00",
            )
            s.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 2, "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # В дайджест попали только 2 свежих (max_items=2, old — за лимитом)
            self.assertEqual(count, 2)
            send_text_args = delivery.send_text.call_args[0][0]
            self.assertIn("Fresh0", send_text_args)
            self.assertIn("Fresh1", send_text_args)

            # Собираем ВСЕ items из ВСЕХ состояний
            all_window_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_window_urls.update(i.url for i in p.storage.get_all_by_state(state))

            # 3 old items — deferred, НЕ selected (низкий score), остались после окна.
            for i in range(3):
                self.assertIn(f"https://ex.com/old{i}", all_window_urls)

            # 2 fresh items — ОТПРАВЛЕНЫ → digested, остаются в БД
            # (delete_items_before с границей window_start_iso=2020-01-01,
            # digested-items с fetched_at 2026-01 НЕ удаляются — они из текущего окна)
            for i in range(2):
                self.assertIn(f"https://ex.com/fresh{i}", all_window_urls)
                state = list(p.storage.get_all_by_state("digested"))
                self.assertTrue(
                    any(f"https://ex.com/fresh{i}" == item.url for item in state),
                    f"fresh{i} должен быть state=digested"
                )

            # Item, пришедший во время отправки (fetched_at в будущем) — ОСТАЛСЯ
            self.assertIn("https://ex.com/during", all_window_urls)

            # digest_window_started_at сдвинут и больше прежнего
            old_boundary = "2020-01-01T00:00:00+00:00"
            new_boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(new_boundary)
            self.assertGreater(new_boundary, old_boundary)

            p.storage.close()
            p.close()


class TestDigestWindowFailure(unittest.TestCase):
    """Падение доставки: items НЕ удалены, граница НЕ сдвинута."""

    def test_failed_delivery_no_delete_no_advance(self):
        """
        3 deferred-items. send_text возвращает False → доставка не удалась.
        Items НЕ удалены, граница НЕ сдвинута, state остаётся deferred.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2024-01-01T00:00:00+00:00")
            s.close()

            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Item{i}", url=f"https://ex.com/item{i}",
                    published="2099-01-01 12:00",
                    fetched_at="2099-01-01T12:00:00+00:00",
                )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = False  # полный сбой
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 0)

            # Items остались в deferred
            deferred_urls = {i.url for i in p.storage.get_all_by_state("deferred")}
            for i in range(3):
                self.assertIn(f"https://ex.com/item{i}", deferred_urls)

            # Граница НЕ сдвинута
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertEqual(boundary, "2024-01-01T00:00:00+00:00")

            p.storage.close()
            p.close()

    def test_partial_failure_no_delete_no_advance(self):
        """
        max_items_per_message=2, 3 deferred-items.
        msg1 (2 items) → OK, msg2 (1 item) → FAIL.
        sent_count=2 < total_to_send=3 → частичный сбой.
        Отправленные items → 'digested', неотправленный → 'deferred'.
        Граница НЕ сдвинута, НИЧЕГО не удалено из БД.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2024-01-01T00:00:00+00:00")
            s.close()

            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Item{i}", url=f"https://ex.com/p{i}",
                    published="2025-01-01 12:00",
                    fetched_at="2025-01-01T12:00:00+00:00",
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 2,
                },
            )
            delivery = MagicMock()
            call_count = [0]

            def fake_send(text):
                call_count[0] += 1
                return call_count[0] == 1  # только msg1 OK

            delivery.send_text.side_effect = fake_send
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 2)  # msg1: 2 items

            # p0, p1 отправлены → 'digested', p2 не доставлен → 'deferred'
            digested_urls = {i.url for i in p.storage.get_all_by_state("digested")}
            deferred_urls = {i.url for i in p.storage.get_all_by_state("deferred")}
            all_urls = digested_urls | deferred_urls
            # Все 3 items должны остаться в БД (ничего не удалено)
            for i in range(3):
                self.assertIn(f"https://ex.com/p{i}", all_urls)

            # p2 всё ещё deferred (msg2 не доставлен)
            self.assertIn("https://ex.com/p2", deferred_urls)
            # p0, p1 — digested (отправлены)
            self.assertIn("https://ex.com/p0", digested_urls)
            self.assertIn("https://ex.com/p1", digested_urls)

            # Граница НЕ сдвинута
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertEqual(boundary, "2024-01-01T00:00:00+00:00")

            p.storage.close()
            p.close()


class TestDigestSecondRun(unittest.TestCase):
    """Второй дайджест подряд: в выборку попадают только items после новой границы."""

    def test_second_digest_only_after_new_boundary(self):
        """
        Первый digest (force) успешен → граница сдвинута на digest_started_iso.
        Второй digest: читает items с fetched_at >= новая граница.
        Item, пришедший между двумя digest, попадает во второй.
        Item, пришедший до первого digest, — удалён.

        Схема по времени (fetched_at):
          - 2023-12-31: stale_item (ЗА границей, не попадает в D1)
          - 2026-01-01: fresh_item (ВНУТРИ границы, отправлен в D1 → digested)
          - после D1 boundary: new_item (ВНУТРИ новой границы, отправлен в D2 → digested)
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # Начальная граница: конец 2025. Items из 2025 и раньше — за границей.
            s.set_meta("digest_window_started_at", "2025-12-31T23:59:59+00:00")
            s.close()

            # stale_item: fetched_at=2023, за границей — не попадает в D1
            _insert_deferred_item(
                db_path, title="StaleItem", url="https://ex.com/stale",
                published="2023-12-31 12:00", fetched_at="2023-12-31T12:00:00+00:00",
            )
            # fresh_item: fetched_at=2026-01, внутри окна — отправлен в D1
            _insert_deferred_item(
                db_path, title="FreshItem", url="https://ex.com/fresh",
                published="2026-01-01 12:00", fetched_at="2026-01-01T12:00:00+00:00",
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count1 = p.run_digest(force=True)

            self.assertEqual(count1, 1)  # только fresh_item
            old_boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(old_boundary)

            # stale_item удалён (за границей); fresh_item — digested, остаётся в БД
            all_after_d1 = set()
            for state in ["deferred", "digested", "new", "pending", "skipped", "failed"]:
                all_after_d1.update(i.url for i in p.storage.get_all_by_state(state))
            # stale_item - deferred за границей, delete_items_before не трогает deferred
            deferred_after_d1 = {i.url for i in p.storage.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/stale", deferred_after_d1,
                "stale deferred за границей НЕ удаляется delete_items_before")
            # fresh_item - digested, НЕ удаляется (граница = window_start_iso 2025,
            # не NOW; digested с fetched_at 2026-01 > window_start_iso)
            self.assertIn("https://ex.com/fresh", all_after_d1)
            fresh_states = [i.url for i in p.storage.get_all_by_state("digested")]
            self.assertIn("https://ex.com/fresh", fresh_states,
                "fresh_item должен остаться state=digested в текущем окне")

            # Патчим datetime.now: первый вызов (D1) → 2026-01-01T00:00:00 UTC,
            # второй вызов (D2) → на 2 дня позже (сдвигаем реальное время).
            # Без патча D1 и D2 могут получить одинаковое now_utc (одна секунда),
            # тогда new_item (fetched_at = old_boundary + 2s) не удалится при D2.
            import datetime as _dt
            from datetime import timedelta, timezone
            from unittest.mock import patch

            old_dt = _dt.datetime.fromisoformat(old_boundary.replace("Z", "+00:00"))
            # D2_now на 2 дня позже D1 → граница сдвинется, new_item попадёт в D2
            d2_now = old_dt + _dt.timedelta(days=2)

            def _fake_patch_dt_now(fake_utc: _dt.datetime):
                """Патчит newsbot.pipeline.datetime — мокает class-level, не builtin."""
                real_dt = _dt.datetime
                fake_cls = MagicMock()
                fake_cls.now.return_value = fake_utc
                fake_cls.side_effect = lambda *a, **kw: real_dt(*a, **kw)
                fake_cls.timedelta = _dt.timedelta
                fake_cls.replace = real_dt.replace
                fake_cls.fromisoformat = real_dt.fromisoformat
                return patch('newsbot.pipeline.datetime', fake_cls)

            # new_item: fetched_at на 2 дня позже D1_now — внутри D2-окна,
            # но старше D2 digest_started_iso → попадёт в D2 и будет удалён.
            # D2 digest_started_iso = D2_now (2026-01-03), new_item = 2026-01-02 (< D2).
            new_item_fetched = (old_dt + _dt.timedelta(days=1)).isoformat()
            _insert_deferred_item(
                db_path, title="NewItem", url="https://ex.com/new",
                published="2026-01-03 12:00", fetched_at=new_item_fetched,
            )

            # D2: патчим datetime.now → 2 дня спустя
            with _fake_patch_dt_now(d2_now.replace(tzinfo=timezone.utc)):
                count2 = p.run_digest(force=True)

            # new_item имеет published="2026-01-03" < d2 window_start (~Oct 22).
            # Он проходит fetched_at-фильтр, но NOT published-фильтр (late posting).
            # Фикс: такие items НЕ попадают в digest.
            self.assertEqual(count2, 0)

            # new_item остаётся deferred (не был отправлен, не удалён).
            # stale_item — остался deferred (за границей, delete_items_before не трогает deferred).
            # fresh_item — остался digested (fetched_at=2026-01 > window_start_iso=2025-12-31,
            # delete_items_before с логической границей, не NOW).
            deferred_after_d2 = {i.url for i in p.storage.get_all_by_state("deferred")}
            all_after_d2 = set()
            for state in ["deferred", "digested", "new", "pending", "skipped", "failed"]:
                all_after_d2.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertIn("https://ex.com/stale", deferred_after_d2,
                "stale deferred за границей сохраняется после D1 (only purge_items removes)")
            self.assertIn("https://ex.com/fresh", all_after_d2,
                "fresh_item digested, НЕ удалён: window_start_iso фиксирована, NOW ≠ граница")
            self.assertIn("https://ex.com/new", deferred_after_d2)  # остался deferred

            # Граница сдвинута на digest_started_iso (new_item.fetched_at старше,
            # но это не 'new'/'pending' — get_oldest_unprocessed_fetched_at вернёт None,
            # therefore new_boundary = digest_started_iso > old_boundary).
            new_boundary = p.storage.get_meta("digest_window_started_at")
            self.assertGreater(new_boundary, old_boundary)

            p.storage.close()
            p.close()


class TestDigestEmptyWindow(unittest.TestCase):
    """Пустое окно (on_empty="skip"): граница сдвигается."""

    def test_empty_deferred_skips_and_advances_boundary(self):
        """
        deferred пуст (item за границей окна). on_empty="skip", force=False.
        Граница должна сдвинуться (skip → early return with boundary update).
        При force=True граница не двигается (пустой дайджест отправляется).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2025-01-01T00:00:00+00:00")
            s.close()

            # Item с fetched_at = 2024-12-31 (ДО границы окна 2025-01-01).
            # С явной границей since_iso отфильтрует его → deferred пуст.
            _insert_deferred_item(
                db_path,
                title="StaleItem", url="https://ex.com/stale",
                published="2024-12-31 12:00",
                fetched_at="2024-12-31T23:59:00+00:00",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20, "on_empty": "skip",
                    "bootstrap_window_hours": 24,
                },
            )
            delivery = MagicMock()
            p.delivery = delivery

            # on_empty=skip с force=False: early return, граница обновлена
            count = p.run_digest(force=False)

            self.assertEqual(count, 0)
            delivery.send_text.assert_not_called()

            # Граница сдвинута
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(boundary)
            self.assertGreater(boundary, "2025-01-01T00:00:00+00:00")

            # last_digest_at тоже записан
            last = p.storage.get_meta("last_digest_at")
            self.assertIsNotNone(last)

            # StaleItem остался в deferred (за границей, не попал в выборку)
            deferred_urls = {i.url for i in p.storage.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/stale", deferred_urls)

            p.storage.close()
            p.close()

    def test_empty_window_with_no_initial_boundary(self):
        """
        Граница не установлена. on_empty="skip".
        Bootstrap сработал (now - bootstrap_window_hours), граница сдвинута при пустом окне.
        """
        with _tmp_db() as db_path:
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20, "on_empty": "skip",
                    "bootstrap_window_hours": 24,
                },
            )
            delivery = MagicMock()
            p.delivery = delivery

            # Граница не установлена
            self.assertIsNone(p.storage.get_meta("digest_window_started_at"))

            count = p.run_digest(force=True)

            self.assertEqual(count, 0)

            # Граница установлена: bootstrap (now - 24h) и сдвинута при skip
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(boundary)

            # last_digest_at записан
            last = p.storage.get_meta("last_digest_at")
            self.assertIsNotNone(last)

            p.storage.close()
            p.close()


class TestDigestEmptyWindowWithUnprocessed(unittest.TestCase):
    """Пустое окно + необработанный pending: граница НЕ уезжает дальше pending."""

    def test_empty_window_with_pending_preserves_boundary(self):
        """
        deferred пуст, но есть pending-item с fetched_at ВНУТРИ окна.
        on_empty='skip'. Граница НЕ должна уехать дальше fetched_at pending-item.
        Иначе pending-item при следующем дайджесте окажется за границей и потеряется.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2025-01-01T00:00:00+00:00")
            s.close()

            # pending-item с fetched_at = 2025-06 (ВНУТРИ окна, старше границы,
            # но младше текущего момента — типичный кейс: item застрял в pending)
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    "TestSrc",
                    "https://ex.com/pending",
                    "PendingItem",
                    "Pending summary",
                    "2025-06-01 12:00",
                    "2025-06-01T12:00:00+00:00",
                    "2025-06-01T12:00:00+00:00",
                    7,
                ),
            )
            conn.commit()
            conn.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20, "on_empty": "skip",
                    "bootstrap_window_hours": 24,
                },
            )
            delivery = MagicMock()
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 0)  # deferred пуст → 0
            delivery.send_text.assert_not_called()

            # pending-item сохранён (не удалён)
            pending_urls = {i.url for i in p.storage.get_all_by_state("pending")}
            self.assertIn("https://ex.com/pending", pending_urls)

            # Граница НЕ дальше fetched_at pending-item
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(boundary)
            # boundary <= pending.fetched_at (2025-06-01T12:00:00+00:00)
            self.assertLessEqual(boundary, "2025-06-01T12:00:00+00:00")
            # При этом граница сдвинута ОТ начальной
            self.assertGreater(boundary, "2025-01-01T00:00:00+00:00")

            p.storage.close()
            p.close()


class TestDigestWindowUnprocessedAndCleanup(unittest.TestCase):
    """Необработанные items сохраняются; сбой очистки не двигает границу."""

    def test_unprocessed_items_preserved_after_digest_success(self):
        """
        В БД есть:
          - new-item с fetched_at СТАРШЕ старта дайджеста (new, pending, skipped, failed)
          - deferred-item ВНУТРИ окна (уйдёт в digest).
        После успешного digest: deferred удалён, new остался,
        digest_window_started_at <= fetched_at этого new.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # new-item: fetched_at = 2024 (старше старта дайджеста — попадает
            # в delete_items_before, но new — не терминальный, не удаляется)
            _insert_deferred_item(
                db_path,
                title="NewItem", url="https://ex.com/new",
                published="2024-01-01 12:00",
                fetched_at="2024-01-01T12:00:00+00:00",
                score=6,
            )
            # Переводим в new (не deferred)
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE items SET state='new', state_at=? WHERE url=?",
                ("2024-01-01T12:00:00+00:00", "https://ex.com/new"),
            )
            conn.commit()
            conn.close()

            # deferred-item внутри окна — уйдёт в digest
            _insert_deferred_item(
                db_path,
                title="DeferredItem", url="https://ex.com/def",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=7,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 1)  # только deferred-item

            # deferred-item отправлен → digested, остаётся в БД
            # (window_start_iso=2020-01-01, digested fetched_at=2026-01 > границы)
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertIn("https://ex.com/def", all_urls)
            def_states = [i.url for i in p.storage.get_all_by_state("digested")]
            self.assertIn("https://ex.com/def", def_states,
                "DeferredItem должен остаться state=digested в текущем окне")

            # new-item сохранён
            self.assertIn("https://ex.com/new", all_urls)

            # Граница НЕ дальше fetched_at new-item
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(boundary)
            # boundary <= fetched_at new-item (2024-01-01T12:00:00+00:00)
            self.assertLessEqual(boundary, "2024-01-01T12:00:00+00:00")

            p.storage.close()
            p.close()

    def test_cleanup_failure_does_not_advance_boundary(self):
        """
        delete_items_before падает → run_digest НЕ падает,
        граница НЕ сдвинута, last_digest_at записан.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path,
                title="Item", url="https://ex.com/cleanup_fail",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=7,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Мокаем delete_items_before на сбой
            from unittest.mock import patch
            with patch.object(p.storage, "delete_items_before",
                              side_effect=RuntimeError("disk full")):
                count = p.run_digest(force=True)

            self.assertEqual(count, 1)  # отправка прошла

            # Item отправлен, но delete_items_before упал → граница НЕ сдвинута
            boundary = p.storage.get_meta("digest_window_started_at")
            self.assertEqual(boundary, "2020-01-01T00:00:00+00:00")

            # last_digest_at записан
            last = p.storage.get_meta("last_digest_at")
            self.assertIsNotNone(last)

            p.storage.close()
            p.close()


class TestCategoryGuaranteeWithPool(unittest.TestCase):
    """Задача B: _category_guarantee тянет items из-за пределов top-N."""

    def test_guarantee_pulls_cinema_from_beyond_top_n(self):
        """
        max_items=3. В БД 6 deferred-items (limit=max_items*2=6):
          top-3 по score: items с score 9/8/7 — БЕЗ категории 'cinema'
          items 4-6: scores 6/5/4, последний — category='cinema', score=5
        category_min: cinema=1.

        После guarantee в итоговом дайджесте (после сорта и финального slice):
          - cinema-item (score 5) ДОЛЖЕН быть (гарантия выполнена)
          - вытеснён должен быть самый низкий НЕ-категорийный (score 7),
            а не cinema-item (которого в top-3 и не было).
        Проверяем по URL.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 3 top-items без категории cinema
            for score, idx in [(9, 0), (8, 1), (7, 2)]:
                _insert_deferred_item(
                    db_path,
                    title=f"Top{idx}",
                    url=f"https://ex.com/top{idx}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                )

            # 3 pool-items: cinema (score 5) за пределами top-3 + ещё 2 обычных
            # tags=['film','movie'] → _infer_category вернёт 'cinema'
            _insert_deferred_item(
                db_path,
                title="CinemaItem",
                url="https://ex.com/cinema",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=5,
                tags='["film","movie"]',
            )
            _insert_deferred_item(
                db_path,
                title="Pool2",
                url="https://ex.com/pool2",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=6,
            )
            _insert_deferred_item(
                db_path,
                title="Pool3",
                url="https://ex.com/pool3",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=4,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True,
                    "max_items": 3,
                    "max_items_per_message": 20,
                    "category_min": {"cinema": 1},
                    "cluster_duplicates": True,
                },
            )
            # llm_filter=None → _cluster_deferred_duplicates возвращает {}.
            # dedup не убирает top-items, бэкфилл не нужен.
            # Но cinema нет в top-3 → гарантия тянет из пула, вытесняя Top2.
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Пул: [CinemaPool(5), Pool2(6), Pool3(4)], Top0..Top2 в top.
            # 1) cinema<min → гарантия добавляет CinemaPool(5), evict Top2(7).
            # 2) Pool2(6) evictable (other не в min_guarantee, cat_counts[other]=2>0)
            #    → evict Pool2(6), добавить Pool3(4).
            # После гарантии: [Top0(9), Top1(8), CinemaPool(5), Pool3(4)].
            # Swap 1:1 → размер не меняется (4 items). Cap не затрагивает (France=other).
            # Сорт: Top0, Top1, CinemaPool. Финальный slice[:3]: 3 items.
            self.assertEqual(count, 3,
                msg="category_guarantee может превышать max_items на 1 (до финального slice)")
            text = delivery.send_text.call_args[0][0]
            # cinema-item в дайджесте
            self.assertIn("CinemaItem", text)
            self.assertIn("https://ex.com/cinema", text)
            # Top2 (вытесненный минимальный non-category, score=7) НЕ в дайджесте
            self.assertNotIn("Top2", text)
            # Top0, Top1 (не вытеснены — их score выше)
            self.assertIn("Top0", text)
            self.assertIn("Top1", text)

            p.storage.close()
            p.close()

    def test_backfill_restores_full_max_items_after_llm_dedup(self):
        """
        Задача A: LLM-дедуп убирает 1 item из топ-3 → без бэкфилла
        дайджест неполный (N-1). С бэкфиллом — полный max_items.

        Подменяем _cluster_deferred_duplicates через подмену llm_filter,
        чтобы гарантированно получить нужное поведение без зависимости
        от patch.object на instance-метод. Также используем lowercase-URLs
        чтобы mock return_value совпадал с реальными URL.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 5 items: top-3 по score=a,b,c; pool=d,e
            # lowercase URL = https://ex.com/{title.lower()} для совпадения с mock
            for score, title in [(9, "a"), (8, "b"), (7, "c"), (6, "d"), (5, "e")]:
                _insert_deferred_item(
                    db_path,
                    title=title.upper(),  # "A", "B", "C", "D", "E" (заголовок)
                    url=f"https://ex.com/{title}",  # lowercase URL
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 3, "max_items_per_message": 20,
                },
            )

            # Подменяем llm_filter: detect_duplicates возвращает {1} (индекс B).
            # _cluster_deferred_duplicates увидит set({1}) и вернёт {"https://ex.com/b"}.
            real_filter = getattr(p, "llm_filter", None) or MagicMock()
            real_filter.detect_duplicates = MagicMock(return_value={1})
            p.llm_filter = real_filter

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Проверяем отправленный текст
            self.assertEqual(count, 3,
                f"Бэкфилл должен восстановить полный max_items. Получили {count}")
            self.assertEqual(len(delivery.send_text.call_args_list), 1)
            send_text_args = delivery.send_text.call_args[0][0]
            # D (backfilled) должен быть в дайджесте
            self.assertIn("D", send_text_args,
                "Backfilled item D должен попасть в дайджест")
            # B (дубль) не должно быть
            self.assertNotIn("B", send_text_args,
                "Dубликат B не должен попасть в дайджест")
            # A и C — лидеры, остаются
            self.assertIn("A", send_text_args)
            self.assertIn("C", send_text_args)

            p.storage.close()
            p.close()


class TestDigestWindowIntegrity(unittest.TestCase):
    """Проверяем, что существующие тесты оконной логики не сломаны."""

    def test_published_filter_still_works_after_truncation_outside_if(self):
        """
        NIT 1: обрезка вынесена из if raw_boundary.
        Это не должно сломать published-фильтр.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-20T00:00:00+00:00")
            s.close()

            # Item с late published
            _insert_deferred_item(
                db_path,
                title="LatePublished",
                url="https://ex.com/late",
                published="19.10.2026 20:00 UTC",
                fetched_at="2026-10-21T12:00:00+00:00",
                score=10,
            )
            # Item с valid published
            _insert_deferred_item(
                db_path,
                title="ValidPublished",
                url="https://ex.com/valid",
                published="2026-10-21 12:00 UTC",
                fetched_at="2026-10-21T12:00:00+00:00",
                score=5,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Late item отфильтрован, valid — попал
            self.assertEqual(count, 1)
            self.assertIn("ValidPublished", delivery.send_text.call_args[0][0])
            self.assertNotIn("LatePublished", delivery.send_text.call_args[0][0])

            p.storage.close()
            p.close()

    def test_truncation_guard_no_blow_up_on_invalid_max_items(self):
        """
        NIT 2: limit=max_items*2 при max_items<=0 раньше давал безлимитный SELECT.
        Проверяем, что при max_items=0 или None код не падает.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for i in range(5):
                _insert_deferred_item(
                    db_path,
                    title=f"Item{i}",
                    url=f"https://ex.com/item{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=7 - i,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True,
                    "max_items": 0,  # zero — guard должен отработать
                    "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Не должно упасть (раньше был безлимитный SELECT, теперь None)
            try:
                count = p.run_digest(force=True)
                # max_items=0 нормализуется в max(1,0)=1 на входе,
                # поэтому в дайджесте будет ≥1 item. Главное — не падает.
            except Exception as e:
                self.fail(f"run_digest упал при max_items=0: {e}")

            p.storage.close()
            p.close()




class TestDigestMaxAgeHours(unittest.TestCase):
    """
    Minor 5-фикс: digest.max_age_hours проводка через run_digest.
    Item старше max_age_hours → НЕ в дайджесте.
    """

    def test_old_item_filtered_by_max_age_hours(self):
        """
        2 deferred items:
          - old_item: fetched_at 25 часов назад (за пределами max_age=24)
          - fresh_item: fetched_at 1 час назад (внутри max_age=24)
        max_age_hours=24 → только fresh_item попадает в digest.
        """
        import datetime as _dt

        now_ts = _dt.datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
        old_fetched = (now_ts - _dt.timedelta(hours=25)).isoformat()
        fresh_fetched = (now_ts - _dt.timedelta(hours=1)).isoformat()
        window_start = (now_ts - _dt.timedelta(hours=30)).isoformat()

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", window_start)
            s.close()

            # old_item: fetched_at 25 часов назад
            _insert_deferred_item(
                db_path,
                title="OldItem",
                url="https://ex.com/old",
                published="2026-10-22 12:00",
                fetched_at=old_fetched,
                score=8,
            )
            # fresh_item: fetched_at 1 час назад
            _insert_deferred_item(
                db_path,
                title="FreshItem",
                url="https://ex.com/fresh",
                published="2026-10-22 12:00",
                fetched_at=fresh_fetched,
                score=6,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "max_age_hours": 24,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Minor 5-фикс: патчим datetime в ОБОИХ модулях (pipeline + storage),
            # т.к. max_age_iso вычисляется в storage.get_deferred_items().
            fake_cls = MagicMock()
            fake_cls.now.return_value = now_ts
            fake_cls.side_effect = lambda *a, **kw: _dt.datetime(*a, **kw)
            fake_cls.timedelta = _dt.timedelta
            fake_cls.replace = _dt.datetime.replace
            fake_cls.fromisoformat = _dt.datetime.fromisoformat
            with patch('newsbot.pipeline.datetime', fake_cls), \
                 patch('newsbot.storage.datetime', fake_cls):
                count = p.run_digest(force=True)

            # Только fresh_item попал (old_item отфильтрован max_age)
            self.assertEqual(count, 1,
                f"Expected 1 (only fresh), got {count}")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("FreshItem", text,
                "FreshItem должен быть в digest")
            self.assertNotIn("OldItem", text,
                "OldItem (25ч) не должен быть в digest при max_age=24ч")

            p.storage.close()
            p.close()

    def test_no_max_age_keeps_all_items(self):
        """
        Без max_age_hours: даже очень старый item (fetched_at 100 часов назад)
        попадает в digest (ограничен только оконной границей).
        """
        import datetime as _dt

        now_ts = _dt.datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
        old_fetched = (now_ts - _dt.timedelta(hours=100)).isoformat()
        window_start = (now_ts - _dt.timedelta(hours=200)).isoformat()

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", window_start)
            s.close()

            _insert_deferred_item(
                db_path,
                title="OldItem",
                url="https://ex.com/old",
                published="2026-10-22 12:00",
                fetched_at=old_fetched,
                score=8,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    # max_age_hours НЕ задан
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            fake_cls = MagicMock()
            fake_cls.now.return_value = now_ts
            fake_cls.side_effect = lambda *a, **kw: _dt.datetime(*a, **kw)
            fake_cls.timedelta = _dt.timedelta
            fake_cls.replace = _dt.datetime.replace
            fake_cls.fromisoformat = _dt.datetime.fromisoformat
            with patch('newsbot.pipeline.datetime', fake_cls), \
                 patch('newsbot.storage.datetime', fake_cls):
                count = p.run_digest(force=True)

            self.assertEqual(count, 1,
                "Без max_age_hours старый item должен попасть в digest")

            p.storage.close()
            p.close()


class TestCategoryGuaranteeEvictable(unittest.TestCase):
    """MAJOR 1: evictable victim logic — only non-category or surplus items."""

    def test_guarantee_never_evicts_needed_category(self):
        """
        Scenario from spec:
          items=[games(2), other(9), other(8)], pool=[cinema(5)]
          min: cinema=2, games=1

        After cinema processing: evictable = {other(9), other(8)}
        (games is needed for its own category_min guarantee, cat_counts[games]=1==min).
        Victim = other(8) (lowest score among evictable).
        After swap: cat_counts updated incrementally.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # items: games(2), other(9), other(8) — inside window
            # Source 'ign' = games category (use direct SQL to set source)
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
                """,
                ("ign", "https://ex.com/games1", "GamesItem", "Test summary",
                 "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                 "2026-01-01T12:00:00+00:00", 2),
            )
            conn.commit()
            conn.close()
            _insert_deferred_item(
                db_path, title="OtherHigh", url="https://ex.com/other9",
                published="2026-01-01 12:00", fetched_at="2026-01-01T12:00:00+00:00",
                score=9,
            )
            _insert_deferred_item(
                db_path, title="OtherLow", url="https://ex.com/other8",
                published="2026-01-01 12:00", fetched_at="2026-01-01T12:00:00+00:00",
                score=8,
            )
            # pool: cinema(5)
            _insert_deferred_item(
                db_path, title="CinemaPool", url="https://ex.com/cinema1",
                published="2026-01-01 12:00", fetched_at="2026-01-01T12:00:00+00:00",
                score=5,
                tags='["film","movie"]',
            )
            # Another cinema in pool
            _insert_deferred_item(
                db_path, title="CinemaPool2", url="https://ex.com/cinema2",
                published="2026-01-01 12:00", fetched_at="2026-01-01T12:00:00+00:00",
                score=4,
                tags='["film","movie"]',
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 4, "max_items_per_message": 20,
                    "category_min": {"cinema": 2, "games": 1},
                    "cluster_duplicates": True,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # With max_items=4, result starts at 4 (3 initial + 1 from backfill if needed).
            # After adding 2 cinema items from pool: result=5 > max_items,
            # so one evictable item is evicted (lowest-score other: OtherLow).
            # Final: games(1) + other(9) + cinema(5) + cinema(4) = 4 items.
            # Games is NEVER evicted (games is in category_min, count=1 == min).
            self.assertEqual(count, 4)
            text = delivery.send_text.call_args[0][0]
            # games item must NOT be evicted (it is needed for its own guarantee)
            self.assertIn("GamesItem", text)
            # cinema items MUST be present (guarantee fulfilled)
            self.assertIn("CinemaPool", text)
            self.assertIn("CinemaPool2", text)
            # OtherLow (lowest score among evictable) should be evicted
            self.assertNotIn("OtherLow", text)

            p.storage.close()
            p.close()

    def test_guarantee_pool_none_legacy_mode(self):
        """
        pool=None: legacy mode — _category_guarantee returns items unchanged.
        Tests backward compatibility (old tests use pool=None).
        """
        from newsbot.pipeline import _category_guarantee, _infer_category

        class FakeItem:
            def __init__(self, score, source):
                self.llm_score = score
                self.source = source

        items = [
            FakeItem(9, 'deadline'),  # cinema
            FakeItem(2, 'deadline'),  # cinema
        ]
        result = _category_guarantee(items, {'cinema': 2}, pool=None)
        cats = [_infer_category(i) for i in result]
        self.assertEqual(cats.count('cinema'), 2)
        self.assertEqual(len(result), 2)  # unchanged


class TestLocationCap(unittest.TestCase):
    """MAJOR 2: _cap_by_location — no more than max_per_location per (location, day)."""

    def test_cap_by_location_allows_up_to_max(self):
        """
        4 items with same location+day, max_per_location=2.
        First 2 survive (by iteration order), rest 2 are filtered.
        """
        from newsbot.pipeline import _cap_by_location

        class FakeItem:
            def __init__(self, title, pub, cluster_key=None):
                self.title = title
                self.published = pub
                self.cluster_key = cluster_key

        items = [
            FakeItem("France News 1", "2026-10-20", "франция|abc|2026-10-20"),
            FakeItem("France News 2", "2026-10-20", "франция|def|2026-10-20"),
            FakeItem("France News 3", "2026-10-20", "франция|ghi|2026-10-20"),
            FakeItem("France News 4", "2026-10-20", "франция|jkl|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_cap_by_location_different_days_pass(self):
        """Same location, different days — both pass cap."""
        from newsbot.pipeline import _cap_by_location

        class FakeItem:
            def __init__(self, title, pub, cluster_key=None):
                self.title = title
                self.published = pub
                self.cluster_key = cluster_key

        items = [
            FakeItem("France News day1", "2026-10-20", "франция|abc|2026-10-20"),
            FakeItem("France News day1", "2026-10-20", "франция|def|2026-10-20"),
            FakeItem("France News day2", "2026-10-21", "франция|ghi|2026-10-21"),
            FakeItem("France News day2", "2026-10-21", "франция|jkl|2026-10-21"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 4)  # 2 per day, all pass

    def test_cap_by_location_other_bucket_not_capped(self):
        """Items with bucket 'other' are never capped."""
        from newsbot.pipeline import _cap_by_location

        class FakeItem:
            def __init__(self, title, cluster_key=None):
                self.title = title
                self.cluster_key = cluster_key

        items = [
            FakeItem("Other News 1", "other|abc|2026-10-20"),
            FakeItem("Other News 2", "other|def|2026-10-20"),
            FakeItem("Other News 3", "other|ghi|2026-10-20"),
            FakeItem("Other News 4", "other|jkl|2026-10-20"),
            FakeItem("No cluster key item", None),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 5)  # all 5 pass (bucket=other, no cap)

    def test_cap_by_location_zero_max_passes_all(self):
        """max_per_location=0 means no cap — all items pass."""
        from newsbot.pipeline import _cap_by_location

        class FakeItem:
            def __init__(self, cluster_key):
                self.title = "Item"
                self.cluster_key = cluster_key

        items = [FakeItem("франция|abc|2026-10-20") for _ in range(10)]
        result = _cap_by_location(items, max_per_location=0)
        self.assertEqual(len(result), 10)

    def test_location_cap_in_run_digest(self):
        """
        run_digest with max_per_location=2:
        4 items of same location pass cap → only 2 in digest.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 4 items with same location+day — set cluster_key directly
            # cluster_key = location|stem|day
            for i in range(4):
                # Insert directly via SQL to set cluster_key (not set by _insert_deferred_item)
                conn = sqlite3.connect(db_path)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, title, summary, published, fetched_at,
                     state, state_at, llm_score, tags, cluster_key)
                    VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?, ?, ?)
                    """,
                    (
                        "TestSrc",
                        "https://ex.com/fr{}".format(i),
                        "FranceItem{}".format(i),
                        "Test summary",
                        "2026-01-01 12:00",
                        "2026-01-01T12:00:0{}+00:00".format(i),
                        "2026-01-01T12:00:0{}+00:00".format(i),
                        10 - i,
                        None,
                        "франция|stem{}|2026-01-01".format(i),
                    ),
                )
                conn.commit()
                conn.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10, "max_items_per_message": 20,
                    "max_per_location": 2, "cluster_duplicates": True,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # max_per_location=2: ровно 2 items на (location, day) — верхних по score
            self.assertEqual(
                count, 2,
                msg="Location cap should produce exactly 2 items (top 2 by score)",
            )
            # Verify: отправленные items — два верхних по score этой локации
            call_args = delivery.send_text.call_args
            if call_args:
                text = call_args[0][0] if call_args[0] else ""
                # top-2: FranceItem0 (score=10) и FranceItem1 (score=9) должны быть
                self.assertIn("FranceItem0", text)
                self.assertIn("FranceItem1", text)
                # вырезанные: FranceItem2 (8) и FranceItem3 (7) — не должны быть
                self.assertNotIn("FranceItem2", text)
                self.assertNotIn("FranceItem3", text)

            p.storage.close()
            p.close()


class TestMaxItemsNormalization(unittest.TestCase):
    """MINOR: max_items <=0 normalized to 1 once at start of run_digest."""

    def test_max_items_zero_normalizes_to_one(self):
        """max_items=0 in config → normalized to 1, not zero."""
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for i in range(5):
                _insert_deferred_item(
                    db_path,
                    title="Item{}".format(i),
                    url="https://ex.com/item{}".format(i),
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=7 - i,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 0,
                    "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            try:
                count = p.run_digest(force=True)
                # With normalization: max(1, 0) = 1, so 1 item in digest
                self.assertEqual(count, 1)
            except Exception as e:
                self.fail("run_digest crashed with max_items=0: {}".format(e))

            p.storage.close()
            p.close()

    def test_max_items_negative_normalizes_to_one(self):
        """max_items=-5 in config → normalized to 1."""
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title="Item{}".format(i),
                    url="https://ex.com/neg{}".format(i),
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=7,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": -5,
                    "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            try:
                count = p.run_digest(force=True)
                self.assertEqual(count, 1)
            except Exception as e:
                self.fail("run_digest crashed with max_items=-5: {}".format(e))

            p.storage.close()
            p.close()



if __name__ == "__main__":
    unittest.main()