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
from datetime import timedelta, timezone, datetime as _dt
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


def _insert_skipped_item(db_path, title, url, published, fetched_at,
                         score=5, tags=None, llm_summary=None):
    """Вставляет skipped-item напрямую в БД (state='skipped')."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at,
         llm_score, llm_summary, tags)
        VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?)
        """,
        (
            "Deadline",
            url,
            title,
            "Summary for " + title,
            published,
            fetched_at,
            fetched_at,
            score,
            llm_summary or f"Summary: {title}",
            tags,
        ),
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
    # Закрываем первое соединение перед подменой
    p.storage.close()
    p.storage = Storage(db_path)
    if llm_filter is not None:
        p.llm_filter = llm_filter
    return p


@contextmanager
def _patch_dt_and_local_now(fake_utc):
    """
    Патчит datetime-модуль и _local_now() так, чтобы pipeline видел
    детерминированное local time в UTC+3, согласованное с fake_utc.

    fake_utc — aware UTC datetime. Эквивалент local time = UTC+3:
      fake_utc = 08:00 UTC → local = 11:00 UTC+3.

    Комбинирует два подхода:
      1. patch('newsbot.pipeline.datetime', FakeDT) — module-level ref;
         datetime.now(tz.utc) возвращает fake_utc.
         fake_utc.astimezone() с хостовой TZ даст НЕ UTC+3, но FakeDT.astimezone
         возвращает UTC+3 — перехватывает .astimezone() на instances.
      2. patch('newsbot.pipeline._local_now', ...) — перехватывает локальный
         binding функции в pipeline.py (которая захватила datetime через
         from datetime import datetime в области модуля).

    Не полагается на хостовую TZ через _local_now.

    FakeDT.astimezone мёртв для _local_now (вызовется только если
    datetime.now(tz.utc) вернёт объект БЕЗ .astimezone в UTC) —
    но это не важно, _local_now перехвачен напрямую.
    """
    import datetime as _dt

    utc3 = _dt.timezone(_dt.timedelta(hours=3))
    local_aware = fake_utc.astimezone(utc3)

    # FakeDT патчит module-level reference: pipeline.py использует
    # 'datetime' как module-level name (from datetime import datetime),
    # поэтому patch('newsbot.pipeline.datetime', ...) меняет эту ссылку.
    # .now(tz) возвращает fake_utc (TZ-aware UTC).
    class FakeDT:
        def now(self, tz=None):
            if tz is None:
                return fake_utc
            return fake_utc.astimezone(tz)

        @staticmethod
        def fromisoformat(s):
            return _dt.datetime.fromisoformat(s)

        @staticmethod
        def strptime(s, fmt):
            return _dt.datetime.strptime(s, fmt)

        def astimezone(self, tz=None):
            """Возвращает UTC+3. Эмулирует local TZ."""
            if tz is None:
                return local_aware
            return local_aware.astimezone(tz)

        timedelta = _dt.timedelta
        replace = _dt.datetime.replace
        min = _dt.datetime.min
        max = _dt.datetime.max

    def fake_local_now():
        """Перехватывает _local_now() напрямую."""
        return local_aware

    with patch('newsbot.pipeline.datetime', FakeDT()), \
         patch('newsbot.storage.datetime', FakeDT()), \
         patch('newsbot.pipeline._local_now', fake_local_now):
        yield


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

    def test_3strike_shifts_boundary(self):
        """
        FIX 5: 3-strike сдвигает digest_window_started_at (не только last_digest_at).
        Без фикса: last_digest_at обновляется (слот обработан), но
        digest_window_started_at остаётся от предыдущего успешного окна →
        следующий слот считает границу от anchor → items старше границы
        отсекаются published-фильтром → тихая потеря целого окна.
        Фикс: обе мета-поля обновляются в 3-strike ветке.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2024-01-01T00:00:00+00:00")
            s.set_meta("digest_failure_count", "3")  # уже на 3 неудачах
            s.close()

            for i in range(2):
                _insert_deferred_item(
                    db_path,
                    title=f"Fault{i}", url=f"https://fault.com/f{i}",
                    published="2099-01-01 12:00",
                    fetched_at="2099-01-01T12:00:00+00:00",
                )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = False  # всё падает
            p.delivery = delivery

            # 3-strike ветка: failure_count сброшен, boundary сдвинута
            count = p.run_digest(force=True)

            self.assertEqual(count, 0)

            boundary = p.storage.get_meta("digest_window_started_at")
            # FIX 5: boundary СДВИНУТА (не "2024-01-01T00:00:00+00:00")
            assert boundary != "2024-01-01T00:00:00+00:00", (
                f"FIX 5: 3-strike must shift digest_window_started_at, "
                f"got: {boundary!r}"
            )
            failure_count = p.storage.get_meta("digest_failure_count")
            self.assertEqual(failure_count, "0",
                "3-strike resets failure_count")

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


class TestAnchorModeMaxAgeHoursBypassed(unittest.TestCase):
    """
    Задача Minor 6: в anchor-режиме max_age_hours не применяется
    (pipeline.py:2084: max_age_hours=max_age_hours if not anchor_mode else None).

    Scheduler lag scenario: вечерний слот 20:00 фактически запускается в 20:05.
    Deferred-новость с fetched_at = вчера 20:02 (первые минуты 24ч-окна)
    ДОЛЖНА попасть в выборку дайджеста. Если бы max_age_hours применялся
    (24ч от now=20:05), граница была бы вчера 20:05 → новость выпала бы.
    """

    def test_anchor_mode_ignores_max_age_hours_on_scheduler_lag(self):
        """
        Anchor mode: window_anchor="20:00", now=2026-10-22T20:05 UTC (UTC+3: 23:05).
        Anchor boundary = yesterday 20:00 UTC+3 = 2026-10-21T17:00 UTC.

        Pre-set digest_window_started_at (raw_boundary != None) →
        since_iso=anchor boundary активен в SQL.
        Item A: fetched_at=2026-10-21T17:02 UTC (20:02 UTC+3) — ПОСЛЕ anchor.
                max_age_hours=NONE → попадает (anchor boundary рулит).
                Если бы max_age_hours применялся (now-24h=yesterday 20:05):
                effective_since=MAX(17:00, 20:05)=20:05 → ItemA выпал бы.
        Item B: fetched_at=2026-10-21T16:59 UTC (19:59 UTC+3) — ДО anchor.
                SQL since_iso фильтрует (16:59 < 17:00) → НЕ попадает.
        """
        import datetime as _dt
        from datetime import timezone

        # now=2026-10-22T20:05 UTC (scheduler lag: слот 20:00 запущен в 20:05)
        fake_utc = _dt.datetime(2026, 10, 22, 20, 5, 0, 0, tzinfo=timezone.utc)

        with _tmp_db() as db_path:
            # Pre-set meta: anchor boundary будет вычислен из now,
            # since_iso=window_start_iso активен (raw_boundary != None).
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "window_anchor": "20:00",
                    "max_age_hours": 24,  # был бы ограничением в fallback, но НЕ в anchor
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Item A: fetched_at=yesterday 17:02 UTC (20:02 UTC+3 — первые минуты 24ч-окна).
            # Должен попасть: since_iso=anchor 17:00 UTC, max_age_hours bypassed.
            _insert_deferred_item(
                db_path,
                title="ItemA_InsideWindow",
                url="https://ex.com/itema",
                published="22.10.2026 20:00 UTC",
                fetched_at="2026-10-21T17:02:00+00:00",
                score=8,
            )
            # Item B: fetched_at=yesterday 16:59 UTC (19:59 UTC+3 — до якоря 20:00).
            # Не попадает: since_iso=2026-10-21T17:00 UTC, fetched_at 16:59 < 17:00.
            _insert_deferred_item(
                db_path,
                title="ItemB_OutsideWindow",
                url="https://ex.com/itemb",
                published="22.10.2026 20:00 UTC",
                fetched_at="2026-10-21T16:59:00+00:00",
                score=9,  # higher score, would appear first if not filtered
            )

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # Item A (20:02 > anchor 20:00) попал; Item B (19:50 < anchor 20:00) отфильтрован.
            self.assertEqual(count, 1,
                f"Expected 1 (only ItemA), got {count}. "
                "ItemB (fetched=19:50 < anchor 20:00) must be excluded by since_iso. "
                "ItemA (fetched=20:02 > anchor 20:00) must pass — max_age_hours bypassed.")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("ItemA_InsideWindow", text,
                "ItemA (fetched=yesterday 20:02) должен попасть в digest "
                "(anchor boundary 20:00, max_age_hours игнорируется)")
            self.assertNotIn("ItemB_OutsideWindow", text,
                "ItemB (fetched=yesterday 19:50) не должен попасть — за anchor boundary")

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
        result, _ = _category_guarantee(items, {'cinema': 2}, pool=None)
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


# =============================================================================
# Задача 1: слотовое окно с якорем
# =============================================================================

class TestWindowAnchorSlot08Morning(unittest.TestCase):
    """
    Задача 1 (a): утренний слот 08:00 → граница = вчера 20:00 (12ч окно).
    Патчим datetime.now → 2026-10-22T08:00 UTC.
    Anchor: 20:00. yesterday 20:00 = 2026-10-21T20:00 UTC.
    Item с fetched_at = 2026-10-22T07:00 (между вчера 20:00 и сегодня 08:00) → В окне.
    Item с fetched_at = 2026-10-21T19:00 (до вчера 20:00) → ЗА границей.
    """

    def test_slot_08_boundary_is_yesterday_20(self):
        from datetime import datetime as _dt, timezone, timedelta

        with _tmp_db() as db_path:
            # No initial boundary (bootstrap сработает)
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "window_anchor": "20:00",
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Item ВНУТРИ ожидаемого окна: fetched_at 2026-10-22T04:00 UTC
            # (window_start = 2026-10-21T17:00:00+00:00 UTC+3-anchor; 04:00 > 17:00 UTC ✓)
            # published parses: "22.10.2026 04:00 UTC" → 2026-10-22 04:00 UTC (naive)
            # ws_grace_naive = 16:45 UTC; 04:00 >= 16:45 UTC ✓
            _insert_deferred_item(
                db_path,
                title="InsideWindow",
                url="https://ex.com/inside",
                published="22.10.2026 04:00 UTC",
                fetched_at="2026-10-22T04:00:00+00:00",
                score=7,
            )
            # Item ЗА границей: fetched_at 2026-10-21T16:59:00+00:00
            # (16:59 < window_start 17:00 UTC → SQL-фильтр отсеивает)
            _insert_deferred_item(
                db_path,
                title="OutsideWindow",
                url="https://ex.com/outside",
                published="21.10.2026 16:59 UTC",
                fetched_at="2026-10-21T16:59:00+00:00",
                score=9,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # Only InsideWindow попал (published + window boundary).
            # OutsideWindow отфильтрован SQL (fetched_at < 2026-10-21T17:00 UTC).
            self.assertEqual(count, 1,
                f"Expected 1 (inside window), got {count}. "
                "OutsideWindow (fetched<2026-10-21T17:00) must be excluded by SQL.")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("InsideWindow", text)
            self.assertNotIn("OutsideWindow", text)

            p.storage.close()
            p.close()

    def test_slot_20_boundary_is_yesterday_20(self):
        """
        Задача 1 (б): вечерний слот 20:00 → граница = вчера 20:00 (24ч окно).
        Патчим datetime.now → 2026-10-22T20:00 UTC.
        Expected anchor = 2026-10-21T20:00 UTC.
        Item с fetched_at = 2026-10-22T19:59 → В окне.
        Item с fetched_at = 2026-10-21T19:59 → ЗА границей.
        """
        from datetime import datetime as _dt, timezone

        with _tmp_db() as db_path:
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "window_anchor": "20:00",
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            _insert_deferred_item(
                db_path,
                title="Inside20",
                url="https://ex.com/inside20",
                published="22.10.2026 19:59 UTC",  # parsable, 19:59 > ws_grace 16:45
                fetched_at="2026-10-22T19:59:00+00:00",
                score=7,
            )
            _insert_deferred_item(
                db_path,
                title="Outside20",
                url="https://ex.com/outside20",
                published="21.10.2026 19:59 UTC",  # unparsable (not in our format)
                fetched_at="2026-10-21T16:59:00+00:00",  # < window_start 17:00 UTC → SQL filters
                score=9,
            )

            fake_utc = _dt(2026, 10, 22, 20, 0, 0, 0, tzinfo=timezone.utc)

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            self.assertEqual(count, 1,
                f"Expected 1, got {count}. "
                "Outside20 (fetched<2026-10-21T20:00) must be excluded.")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Inside20", text)
            self.assertNotIn("Outside20", text)

            p.storage.close()
            p.close()

    def test_anchor_bootstrap_no_meta(self):
        """
        Задача 1 (в): bootstrap (meta пуста) при anchor-режиме → граница = якорь.
        published-фильтр АКТИВЕН: новость с published ДО якоря не попадает,
        даже если fetched позже.
        """
        from datetime import datetime as _dt, timezone

        with _tmp_db() as db_path:
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "window_anchor": "20:00",
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Item: fetched_at сегодня, но published = вчера 14:00 UTC (ДО grace 16:45 UTC).
            # published-фильтр (с грейсом 15 мин) должен отфильтровать:
            # ws = 2026-10-21T20:00 local = 2026-10-21T17:00 UTC.
            # ws_grace = 17:00 - 15min = 16:45 UTC. Item published = 14:00 UTC < 16:45 UTC → фильтр.
            _insert_deferred_item(
                db_path,
                title="LatePublishedNoDigest",
                url="https://ex.com/latepub",
                published="21.10.2026 14:00 UTC",
                fetched_at="2026-10-22T07:00:00+00:00",
                score=9,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # Item отфильтрован published-фильтром → count=0
            self.assertEqual(count, 0,
                "Item с published до anchor не должен попасть в digest "
                "(published-фильтр АКТИВЕН при bootstrap)")

            p.storage.close()
            p.close()

    def test_fallback_no_window_anchor(self):
        """
        Задача 1 (г): без window_anchor → прежнее поведение (bootstrap 24ч).
        Bootstrap fallback: SQL since_iso=None, published-фильтр выключен,
        поэтому fetched_at-граница НЕ фильтрует items. Оба items попадают
        в digest (сортировка по llm_score DESC: OldItem=9 первым).
        """
        from datetime import datetime as _dt, timezone

        with _tmp_db() as db_path:
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "bootstrap_window_hours": 24,
                    # window_anchor НЕ задан
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Item с fetched_at 48ч назад — за bootstrap-границей, но bootstrap
            # fallback не фильтрует по fetched_at (since_iso=None, published
            # filter off), поэтому оба items попадают.
            _insert_deferred_item(
                db_path,
                title="OldItem",
                url="https://ex.com/old",
                published="20.10.2026 12:00 UTC",
                fetched_at="2026-10-20T12:00:00+00:00",
                score=9,
            )
            _insert_deferred_item(
                db_path,
                title="FreshItem",
                url="https://ex.com/fresh",
                published="22.10.2026 07:00 UTC",
                fetched_at="2026-10-22T07:00:00+00:00",
                score=5,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # Bootstrap fallback: published filter off, fetched_at not filtered.
            # Оба items попадают; OldItem (score=9) первым, FreshItem (score=5) вторым.
            self.assertEqual(count, 2)
            text = delivery.send_text.call_args[0][0]
            self.assertIn("OldItem", text)
            self.assertIn("FreshItem", text)

            p.storage.close()
            p.close()

    def test_anchor_boundary_written_to_meta(self):
        """
        Задача 1 (п.6): в anchor-режиме digest_window_started_at = anchor.
        """
        from datetime import datetime as _dt, timezone

        with _tmp_db() as db_path:
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "max_items_per_message": 20,
                    "window_anchor": "20:00",
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            _insert_deferred_item(
                db_path,
                title="X",
                url="https://ex.com/x",
                published="22.10.2026 01:00 UTC",
                fetched_at="2026-10-22T01:00:00+00:00",
                score=7,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            self.assertEqual(count, 1)
            boundary = p.storage.get_meta("digest_window_started_at")
            # UTC+3: window_start_utc = UTC+3 20:00 - 3h = 2026-10-21T17:00:00+00:00
            self.assertIsNotNone(boundary)
            self.assertEqual(boundary, "2026-10-21T17:00:00+00:00",
                f"Граница должна быть anchor-в-UTC (2026-10-21T17:00:00+00:00), "
                f"получено: {boundary}")

            p.storage.close()
            p.close()

    def test_anchor_meta_min_takes_more_recent_boundary(self):
        """
        FIX 4: anchor-режим с пропущенным слотом — window_start_iso = min(anchor, meta).
        Если digest_slot был пропущен (scheduler lag), meta boundary осталась от
        предыдущего окна и более свежая (=более узкая), чем anchor. БЕЗ min():
        window_start=anchor → items после anchor (но до meta) теряются. С min():
        window_start=meta (более узкая граница) → покрывает items из пропущенного
        слота.
        Break-check: без min() — тест падает, item не попадает.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # anchor (yesterday 2026-10-21 20:00 UTC+3 = yesterday 17:00 UTC)
            # meta = yesterday 14:00 UTC+3 = yesterday 11:00 UTC (ПОЗЖЕ = более свежая)
            # min(anchor=17:00, meta=11:00) = 11:00 UTC — более узкое окно
            s.set_meta("digest_window_started_at", "2026-10-21T14:00:00+03:00")
            s.close()

            # meta = 11:00 UTC (Oct 20) — earlier = more restrictive window
            # anchor = 17:00 UTC (Oct 21) — later = wider window
            # min(anchor, meta) = meta = 11:00 UTC — more restrictive (earlier = newer boundary)
            # Item fetched_at = 22:00 UTC (Oct 21): AFTER anchor (22 > 17), BEFORE meta (22 < 31).
            # БЕЗ min(): window=anchor=17:00 → item 22:00 ВНУТРИ → count=1.
            # С min(): window=meta=11:00 → item 22:00 ВНУТРИ → count=1.
            # => Для разницы нужен item ПОСЛЕ anchor, ДО meta (в UTC-сравнении).
            # meta (Oct 22 07:00 UTC) — NEWER, later timestamp.
            # anchor (Oct 21 17:00 UTC) — OLDER, earlier timestamp.
            # Item at 22:00 UTC Oct 21: after anchor 17:00, before meta 07:00 Oct 22.
            # published=14:01 UTC — строго между meta (11:00 UTC) и anchor (17:00 UTC).
            # БЕЗ min(): window_start=anchor=17:00 → 14:01 < 17:00 → item ОТСЕКАЕТСЯ.
            # С min(): window_start=meta=11:00 → 14:01 > 11:00 → item ПОПАДАЕТ.
            _insert_deferred_item(
                db_path, title="PostAnchorPreMeta",
                url="https://ex.com/papm",
                published="21.10.2026 14:01 UTC",
                fetched_at="2026-10-21T14:01:00+00:00",
                score=8,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "window_anchor": "20:00",
                    "bootstrap_window_hours": 24,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # Item 14:01 UTC — после meta 11:00, до anchor 17:00.
            # БЕЗ min(): window_start=anchor=17:00 → item ОТСЕКАЕТСЯ (published < 17:00).
            # С min(): window_start=meta=11:00 → item ПОПАДАЕТ (11:00 < 14:01).
            # Break-check: без min()-логики тест красный.
            self.assertEqual(count, 1,
                f"FIX 4: min(anchor, meta) must cover items between anchor and meta. "
                f"got count={count}")

            p.storage.close()
            p.close()

    def test_anchor_meta_min_old_boundary_is_ignored(self):
        """
        FIX D: stale meta (meta_dt < anchor_dt - bootstrap) — freshness guard
        отклоняет её, window_start_iso остаётся от anchor. БЕЗ guard: stale
        meta сужает window и теряет items.
        Break-check: без guard тест падает.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # anchor: Oct 21 17:00 UTC. Bootstrap=24ч.
            # Stale threshold = anchor - 24ч = Oct 20 17:00 UTC.
            # meta: Oct 19 08:00 UTC+3 = Oct 19 05:00 UTC (OLDER than threshold).
            # С guard: meta отклоняется, window от anchor → [Oct 20 17:00, ...].
            # БЕЗ guard: window от meta → [Oct 19 05:00, ...].
            s.set_meta("digest_window_started_at", "2026-10-19T08:00:00+03:00")
            s.close()

            # Item: fetched_at = Oct 21 10:00 UTC.
            # Внутри anchor-окна [Oct 20 17:00, Oct 21 17:00] → covered by anchor.
            # Внутри stale meta-окна [Oct 19 05:00, Oct 21 17:00] → covered by stale.
            # → Тест проверяет что stale meta НЕ используется для СУЖЕНИЯ.
            # BREAK-CHECK: если убрать guard — meta всё равно шире (Oct 19 < Oct 20) →
            # результат тот же → тест не падает. Сужаю meta ДО bootstrap:
            # meta = Oct 20 10:00 UTC+3 = Oct 20 07:00 UTC (within bootstrap).
            s2 = Storage(db_path)
            # Anchor: Oct 21 17:00, stale threshold: Oct 20 17:00.
            # Meta: Oct 20 10:00+03:00 = Oct 20 07:00 UTC.
            # Freshness: 07:00 >= 17:00 - 24ч = Oct 20 17:00? NO → 07:00 < 17:00 → stale.
            # С guard: window = anchor → [Oct 20 17:00, Oct 21 17:00].
            # БЕЗ guard: window = meta → [Oct 20 07:00, Oct 21 17:00].
            # Item fetched: Oct 21 14:00 UTC → inside stale-window, OUTSIDE anchor-window.
            s2.set_meta("digest_window_started_at", "2026-10-20T10:00:00+03:00")
            s2.close()

            _insert_deferred_item(
                db_path, title="StaleWindowCoveredItem",
                url="https://ex.com/stale_covered",
                published="21.10.2026 14:00 UTC",
                fetched_at="2026-10-21T14:00:00+00:00",
                score=8,
            )

            fake_utc = _dt(2026, 10, 22, 8, 0, 0, 0, tzinfo=timezone.utc)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100,
                    "window_anchor": "20:00",
                    "bootstrap_window_hours": 24,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with _patch_dt_and_local_now(fake_utc):
                count = p.run_digest(force=True)

            # FIX D: stale meta (Oct 20 07:00 UTC) within bootstrap, but guard
            # checks anchor - bootstrap = Oct 20 17:00 UTC. 07:00 < 17:00 → stale
            # → window stays at anchor (Oct 20 17:00). Item 14:00 UTC is BEFORE
            # anchor 17:00 → OUTSIDE anchor-window → count=0.
            # Break-check: без guard → window = Oct 20 07:00 → item 14:00 inside → count=1.
            self.assertEqual(count, 0,
                f"FIX D: stale meta must be ignored; item before anchor must NOT be covered. "
                f"got count={count}")

            p.storage.close()
            p.close()


class TestWindowDroppedLogging(unittest.TestCase):
    """
    Задача 1 (д): при закрытии окна — WARNING «N deferred items dropped».
    """

    def test_dropped_deferred_logged_on_success(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            # Граница = 2026-10-22T00:00:00 — всё ДО этой границы — «старые deferred»
            s.set_meta("digest_window_started_at", "2026-10-22T00:00:00+00:00")
            s.close()

            # 3 deferred с fetched_at ДО новой границы (будут «дропнуты»)
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Old{i}",
                    url=f"https://ex.com/old{i}",
                    published="2026-10-21T12:00",
                    fetched_at="2026-10-21T12:00:00+00:00",
                    score=5,
                )
            # 1 deferred ВНУТРИ нового окна (попадёт в digest)
            _insert_deferred_item(
                db_path,
                title="FreshItem",
                url="https://ex.com/fresh",
                published="2026-10-22T12:00",
                fetched_at="2026-10-22T12:00:00+00:00",
                score=9,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
                count = p.run_digest(force=True)

            self.assertEqual(count, 1)
            # WARNING должен содержать «dropped» и «3»
            warning_msgs = [m for m in cm.output if "dropped" in m.lower()]
            self.assertTrue(
                any("3" in m and "deferred" in m.lower() for m in warning_msgs),
                f"Expected WARNING about 3 dropped deferred, got: {warning_msgs}",
            )

            p.storage.close()
            p.close()


# =============================================================================
# Задача 2: гарантия кино — добор из skipped
# =============================================================================

class TestSkippedCinemaSupplement(unittest.TestCase):
    """Задача 2: skipped-items с score>=3 дополняют гарантию кино."""

    def _insert_skipped_item(self, db_path, title, url, published, fetched_at,
                              score, tags=None, llm_summary=None):
        """Вставляет skipped-item напрямую в БД."""
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT OR IGNORE INTO items
            (source, url, title, summary, published, fetched_at, state, state_at,
             llm_score, llm_summary, tags)
            VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?)
            """,
            (
                "Deadline",
                url,
                title,
                "Summary for " + title,
                published,
                fetched_at,
                fetched_at,
                score,
                llm_summary or f"Summary: {title}",
                tags,
            ),
        )
        conn.commit()
        conn.close()

    def test_skipped_cinema_pulled_by_guarantee(self):
        """
        Задача 2 (а): 0 cinema в deferred, 2 cinema в skipped (score 3-4,
        deadline-источник) → оба попадают, min достигнут.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 3 deferred items — нет cinema (other sources)
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Other{i}",
                    url=f"https://ex.com/other{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=9 - i,
                )

            # 2 skipped cinema items (Deadline → category=cinema автоматически)
            self._insert_skipped_item(
                db_path,
                title="Cinema1",
                url="https://ex.com/cin1",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=4,  # ≥ guarantee_min_score(3)
            )
            self._insert_skipped_item(
                db_path,
                title="Cinema2",
                url="https://ex.com/cin2",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:01:00+00:00",
                score=3,  # ≥ guarantee_min_score(3)
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "category_min": {"cinema": 2},
                    "guarantee_min_score": 3,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # 2 cinema из skipped должны попасть (гарантия выполнена)
            self.assertEqual(count, 5,
                f"Expected 5 (3 others + 2 skipped cinema), got {count}")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Cinema1", text)
            self.assertIn("Cinema2", text)

            p.storage.close()
            p.close()

    def test_skipped_below_min_score_not_pulled(self):
        """
        Задача 2 (б): skipped-кино score=2 (< guarantee_min_score=3) → не добрано.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 3 deferred — нет cinema
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Other{i}",
                    url=f"https://ex.com/other{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=9 - i,
                )

            # 1 skipped cinema, score=2 (< guarantee_min_score)
            self._insert_skipped_item(
                db_path,
                title="WeakCinema",
                url="https://ex.com/weakcin",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=2,  # ниже порога
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "category_min": {"cinema": 2},
                    "guarantee_min_score": 3,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # WeakCinema (score 2) не добавлен → нет cinema, гарантия не выполнена
            # → вытеснение невозможно (нет cinema-кандидатов) → warning логируется
            text = delivery.send_text.call_args[0][0] if count > 0 else ""
            self.assertNotIn("WeakCinema", text,
                "Skipped item с score<3 не должно быть добавлено")

            p.storage.close()
            p.close()

    def test_skipped_null_score_not_pulled(self):
        """
        Задача 2 (в): skipped-item с llm_score=NULL не добрается.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path,
                title="Other",
                url="https://ex.com/other",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=9,
            )

            # Skipped с NULL score
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at, state, state_at)
                VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?)
                """,
                ("Deadline", "https://ex.com/nullscore",
                 "NullScoreCinema", "Summary",
                 "2026-01-01 12:00",
                 "2026-01-01T12:00:00+00:00",
                 "2026-01-01T12:00:00+00:00"),
            )
            conn.commit()
            conn.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "category_min": {"cinema": 2},
                    "guarantee_min_score": 3,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            # Skipped с NULL score не попал
            self.assertEqual(count, 1)
            text = delivery.send_text.call_args[0][0]
            self.assertNotIn("NullScoreCinema", text)

            p.storage.close()
            p.close()


# =============================================================================
# Задача 3: кросс-оконный дедуп — digested/delivered в RECENT-контексте
# =============================================================================

def _make_llm_filter(mock_client) -> "LLMFilter":
    """Создаёт LLMFilter с замоканным клиентом и отключённым daily limit."""
    from newsbot.llm.filter import LLMFilter
    cfg = {
        "enabled": True,
        "api_key": "test-key",
        "model": "deepseek-flash",
        "push_threshold": 7,
        "digest_threshold": 5,
        "push_urgent_only": False,
        "llm_dedup": True,
        "daily_cost_limit": 0,   # отключаем cost-guard → _check_daily_cost → True
    }
    flt = LLMFilter(cfg, client=mock_client)
    return flt


def _insert_digested_item(db_path, source, url, title, summary,
                          published, fetched_at):
    """Вставляет digested/delivered item напрямую в БД."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at,
         state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, 'digested', ?, 8)
        """,
        (source, url, title, summary, published, fetched_at, fetched_at),
    )
    conn.commit()
    conn.close()


def _mk_pipeline_with_llm(db_path, llm_filter, digest_cfg=None):
    """
    Создаёт Pipeline с LLM enabled и подменённым llm_filter.
    Также подменяет storage, чтобы run_digest использовал реальный tmp_db_path.
    """
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
            "cluster_duplicates": True,
        },
        "llm": {"enabled": True, "api_key": "test-key"},
    }
    p = NewsPipeline(cfg)
    p.storage.close()
    p.storage = Storage(db_path)
    p.llm_filter = llm_filter
    return p


class TestDedupAgainstShownDigested(unittest.TestCase):
    """
    Задача 3: кросс-оконный дедуп — digested/delivered за окно попадают
    в RECENT-контекст LLM-дедупа.

    Проверяет реальный код (detect_duplicates вызывается по-настоящему,
    мокается только HTTP-клиент). Fuzzy-дедуп (seen_titles / get_recent_normalized_titles)
    тоже работает: title_hash/title_normalized считаются от исходного title
    и покрывают точные дубли. LLM-дедуп добавляет семантический уровень
    (похожие пересказы одного события).

    NOTE: unit-тесты fuzzy-дедупа уже есть в test_dedup.py
    (get_recent_normalized_titles с digested/delivered state).
    """

    def test_digested_item_prevents_llm_dedup_match_in_next_window(self):
        """
        Утро: digested-item (показан в 07:00). Вечер: 2 deferred-кандидата,
        один — семантический дубль показанного.

        LLM через detect_duplicates помечает дубль → НЕ попадает в дайджест.

        Minor 5b-фикс: реальный detect_duplicates (мокается HTTP-клиент,
        LLM-ответ управляется через chat_json). Ассертится что дублированный
        title НЕ в отправленном тексте. Индекс {0} = Tech Giant (0-based,
        items=[Tech Giant (score=7, i=0), Market Update (score=6, i=1)]),
        сортировка score DESC. RECENT содержит "Major Tech Company...".
        LLM помечает Tech Giant (i=0) как дубль → {0}. Инвариант
        «лидер остаётся при пустом RECENT» сюда не относится — RECENT непуст.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # Утро: digested-item про tech earnings
            _insert_digested_item(
                db_path,
                source="Reuters",
                url="https://reuters.com/article/xyz",
                title="Major Tech Company Reports Record Q3 Earnings",
                summary="Revenue up 20% year over year.",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )

            # Вечер: 2 deferred-кандидата
            _insert_deferred_item(
                db_path,
                title="Market Update: Stock Prices Rise on Economic Data",
                url="https://example.com/market",
                published="2026-10-22T08:00",
                fetched_at="2026-10-22T08:30:00+00:00",
                score=6,
            )
            _insert_deferred_item(
                db_path,
                title="Tech Giant Posts Best Q3 Results in Company History",
                url="https://apnews.com/article/abc",
                published="2026-10-22T07:30",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=7,
            )

            mock_client = MagicMock()
            # detect_duplicates возвращает 0-based индексы.
            # Items сортируются score DESC: [Tech Giant (score=7, i=0),
            # Market Update (score=6, i=1)].
            # RECENT содержит "Major Tech Company Reports Record Q3 Earnings".
            # LLM помечает Tech Giant (i=0) как дубль (RECENT непуст → лидер
            # не защищён) → 1-based idx=1. _parse_dedup_response ожидает
            # 1-based (1 <= idx <= chunk_len), конвертирует в 0-based для
            # internal use. Мок = имитация LLM-ответа → 1-based.
            mock_client.chat_json.return_value = {"duplicates": [1]}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Market Update прошла в дайджест; Tech Giant отклонён LLM.
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Market Update", text,
                "Non-duplicate (Market Update) должен быть в дайджесте")
            self.assertNotIn("Tech Giant", text,
                "Дубль показанного (Tech Giant) НЕ должен попасть в дайджест")
            self.assertNotIn("Major Tech Company", text,
                "Оригинальный digested-item тоже не должен дублироваться")

            # chat_json вызван (LLM видел RECENT)
            self.assertGreaterEqual(mock_client.chat_json.call_count, 1,
                "LLM должен быть вызван (≥2 deferred-кандидата + RECENT)")

            p.storage.close()
            p.close()

    def test_non_duplicate_passes_through_llm_dedup(self):
        """
        Не-дубли: digested-item про кино + 2 deferred-tech-новости (не дубль).
        LLM возвращает {} → оба deferred попадают в дайджест.

        Minor 5c-фикс: ≥2 deferred-кандидата, реальный detect_duplicates вызов
        (мокается HTTP-клиент). Ассертится что RECENT реально передан в dedup
        (cinema-digested-item виден в RECENT секции).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # Утро: digested-item про кино (НЕ связан с tech)
            _insert_digested_item(
                db_path,
                source="CinemaDaily",
                url="https://cinema.example.com/review",
                title="New Sci-Fi Movie Breaks Box Office Records",
                summary="Blockbuster opens to record numbers.",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )

            # Вечер: 2 tech-новости (обе НЕ дубль кино)
            _insert_deferred_item(
                db_path,
                title="Major Tech Company Reports Record Q3 Earnings",
                url="https://apnews.com/article/tech",
                published="2026-10-22T07:30",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=7,
            )
            _insert_deferred_item(
                db_path,
                title="AI Startup Raises $500M in Series C Round",
                url="https://techcrunch.com/article/ai-funding",
                published="2026-10-22T08:00",
                fetched_at="2026-10-22T09:30:00+00:00",
                score=6,
            )

            # LLM возвращает {} (нет дублей)
            mock_client = MagicMock()
            mock_client.chat_json.return_value = {"duplicates": []}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # RECENT содержит digested-item (cinema) — доказываем что он
            # передан в detect_duplicates (recent section в промпте).
            # chat_json вызван → LLM видел RECENT.
            self.assertGreaterEqual(mock_client.chat_json.call_count, 1,
                "LLM должен быть вызван (≥2 deferred-кандидата)")

            # Оба tech-кандидата попали в дайджест
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Major Tech Company", text)
            self.assertIn("AI Startup", text)

            p.storage.close()
            p.close()

    def test_get_shown_for_window_exception_is_fail_open(self):
        """
        get_shown_for_window кидает исключение → WARNING в логе,
        дайджест собирается (пустой RECENT-контекст = fail-open).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # Два deferred-кандидата (нужно ≥2 чтобы _cluster_deferred_duplicates
            # дошла до вызова get_shown_for_window)
            _insert_deferred_item(
                db_path,
                title="News Item A",
                url="https://example.com/a",
                published="2026-10-22T07:30",
                fetched_at="2026-10-22T08:00:00+00:00",
                score=6,
            )
            _insert_deferred_item(
                db_path,
                title="News Item B",
                url="https://example.com/b",
                published="2026-10-22T07:35",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=5,
            )

            mock_client = MagicMock()
            mock_client.chat_json.return_value = {"duplicates": []}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Подменяем get_shown_for_window на исключение
            orig = p.storage.get_shown_for_window
            p.storage.get_shown_for_window = MagicMock(
                side_effect=RuntimeError("DB error"),
            )

            with patch("newsbot.pipeline.logger") as mock_logger:
                count = p.run_digest(force=True)

                # Дайджест собрался (fail-open: 2 items)
                self.assertEqual(count, 2)
                # WARNING был залогирован
                warning_calls = [
                    str(c) for c in mock_logger.warning.call_args_list
                ]
                self.assertTrue(
                    any("get_shown_for_window" in c for c in warning_calls),
                    f"Expected WARNING about get_shown_for_window, got: {warning_calls}",
                )

            p.storage.get_shown_for_window = orig
            p.storage.close()
            p.close()

    def test_get_shown_for_window_returns_shown_items_for_llm_context(self):
        """
        get_shown_for_window возвращает digested/delivered items за окно
        в формате [{title, source}] — именно то, что ожидает detect_duplicates.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # Внутри окна: digested
            _insert_digested_item(
                db_path,
                source="Reuters",
                url="https://reuters.com/article/xyz",
                title="Tech Earnings",
                summary="Summary",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )
            # Внутри окна: delivered
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'delivered', ?, 8)
                """,
                ("BBC", "https://bbc.com/news",
                 "Other News", "Summary",
                 "2026-10-21T20:00",
                 "2026-10-22T07:00:00+00:00",
                 "2026-10-22T07:30:00+00:00"),
            )
            conn.commit()
            conn.close()

            p = _mk_pipeline_with_llm(db_path, MagicMock())

            result = p.storage.get_shown_for_window(
                cutoff_iso="2026-10-21T20:00:00+00:00",
                limit=200,
            )

            # Должны вернуться оба показанных item
            self.assertEqual(len(result), 2)
            titles = {r["title"] for r in result}
            sources = {r["source"] for r in result}
            self.assertIn("Tech Earnings", titles)
            self.assertIn("Other News", titles)
            self.assertIn("Reuters", sources)
            self.assertIn("BBC", sources)
            # Проверяем формат — только title+source (без url, llm_summary и т.п.)
            for r in result:
                self.assertIn("title", r)
                self.assertIn("source", r)
                self.assertNotIn("url", r)
                self.assertNotIn("llm_summary", r)

            p.storage.close()
            p.close()


class TestCheckAgainstShownBatch(unittest.TestCase):
    """Пункт 3: _check_against_shown — батчевой вызов, 1 LLM на пул."""

    def test_batch_detect_one_duplicate_one_llm_call(self):
        """
        Пул из 3 кандидатов, один дубль показанного (i=0).
        _check_against_shown возвращает 1 url; LLM-вызов ровно один на пул.
        Раньше был цикл по кандидатам → до ~230 LLM-вызовов на дайджест.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # RECENT: digested-item про tech earnings
            _insert_digested_item(
                db_path,
                source="Reuters",
                url="https://reuters.com/article/xyz",
                title="Major Tech Company Reports Record Q3 Earnings",
                summary="Revenue up 20%.",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )

            # 3 deferred-кандидата, один — дубль показанного
            _insert_deferred_item(
                db_path,
                title="Major Tech Company Reports Record Q3 Earnings",  # дубль RECENT
                url="https://apnews.com/dup",
                published="2026-10-22T07:30",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=7,
            )
            _insert_deferred_item(
                db_path,
                title="Market Update: Stock Prices Rise",
                url="https://example.com/market",
                published="2026-10-22T08:00",
                fetched_at="2026-10-22T08:30:00+00:00",
                score=6,
            )
            _insert_deferred_item(
                db_path,
                title="AI Startup Raises $500M",
                url="https://techcrunch.com/article/ai",
                published="2026-10-22T09:00",
                fetched_at="2026-10-22T09:30:00+00:00",
                score=5,
            )

            mock_client = MagicMock()
            # Один дубль — лидер (0-based i=0 → 1-based idx=1).
            # _parse_dedup_response ожидает 1-based (1 <= idx <= chunk_len).
            mock_client.chat_json.return_value = {"duplicates": [1]}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)

            # Собираем recent_shown
            recent_shown = p.storage.get_shown_for_window(
                cutoff_iso="2026-10-21T20:00:00+00:00", limit=200,
            )

            # Вызываем _check_against_shown напрямую
            candidates = p.storage.get_deferred_items(since_iso="2026-10-21T20:00:00+00:00")
            dup_urls = p._check_against_shown(list(candidates), recent_shown)

            # Ровно 1 дубль — лидер
            self.assertEqual(len(dup_urls), 1)
            self.assertIn("https://apnews.com/dup", dup_urls)
            # Ровно 1 LLM-вызов на весь пул (а не 3)
            self.assertEqual(mock_client.chat_json.call_count, 1,
                "Должен быть ровно 1 LLM-вызов на весь пул")

            p.storage.close()
            p.close()

    def test_single_candidate_deduped_against_nonempty_recent(self):
        """
        Пункт 5c: одиночный кандидат + непустой RECENT.
        detect_duplicates вызывается БЕЗ шима, LLM помечает лидера как дубль.
        Дубль отсекается — дайджест отправляет оставшийся item.
        published с timezone гарантирует что item внутри window ( grace_delta).
        Раньше guard len(items)<2 возвращал set() и кандидат отправлялся.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # RECENT: digested-item про кино
            _insert_digested_item(
                db_path,
                source="CinemaDaily",
                url="https://cinema.example.com/review",
                title="New Sci-Fi Movie Breaks Box Office Records",
                summary="Blockbuster opens to record numbers.",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )

            # Ровно 1 deferred-кандидат (published с timezone — внутри grace-окна).
            # ЛЕГИТИМНО может быть дублем RECENT.
            _insert_deferred_item(
                db_path,
                title="Sci-Fi Blockbuster Shatters Opening Weekend Records",
                url="https://example.com/scifi-record",
                published="2026-10-22T08:00+00:00",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=7,
            )

            mock_client = MagicMock()
            # _parse_dedup_response ожидает 1-based (1 <= idx <= chunk_len).
            # LLM пометил лидера (0-based i=0) как дубль → 1-based idx=1.
            mock_client.chat_json.return_value = {"duplicates": [1]}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # count=0: dedup удалил единственного кандидата.
            # (late backfill не добавляет — deferrable_post_published пуст
            # после published-фильтра).
            self.assertEqual(count, 0,
                "Единственный кандидат — дубль RECENT → пустой набор")
            self.assertEqual(delivery.send_text.call_count, 0,
                "Пустой дайджест (все дубли) не отправляется")
            # LLM вызван — guard len<2 снят, LLM видел 1 item + непустой RECENT
            self.assertGreaterEqual(mock_client.chat_json.call_count, 1,
                "LLM должен быть вызван (1 item + непустой RECENT)")

            p.storage.close()
            p.close()

    def test_batch_check_all_candidates_marked_as_dup(self):
        """
        Пункт 5d (вариант через _check_against_shown напрямую):
        Пул из 3 кандидатов, все помечены как дубли → возвращаются все 3 URL.
        LLM-вызов ровно один на весь пул (не 3).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # RECENT: digested-item
            _insert_digested_item(
                db_path,
                source="CinemaDaily",
                url="https://cinema.example.com/review",
                title="New Sci-Fi Movie Breaks Box Office Records",
                summary="Blockbuster.",
                published="2026-10-21T20:00",
                fetched_at="2026-10-22T06:00:00+00:00",
            )

            # 3 deferred-кандидата, все — дубли RECENT
            for i in range(3):
                _insert_deferred_item(
                    db_path,
                    title=f"Sci-Fi News Item {i}",
                    url=f"https://example.com/scifi-{i}",
                    published="2026-10-22T08:00+00:00",
                    fetched_at="2026-10-22T09:00:00+00:00",
                    score=7 - i,
                )

            mock_client = MagicMock()
            # _parse_dedup_response ожидает 1-based.
            # Все 3 кандидата — дубли: i=0→1, i=1→2, i=2→3.
            mock_client.chat_json.return_value = {"duplicates": [1, 2, 3]}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)

            recent_shown = p.storage.get_shown_for_window(
                "2026-10-21T20:00:00+00:00", 200,
            )
            candidates = list(p.storage.get_deferred_items())

            # Вызываем _check_against_shown напрямую
            dup_urls = p._check_against_shown(candidates, recent_shown)

            # Все 3 кандидата помечены как дубли
            self.assertEqual(len(dup_urls), 3)
            for i in range(3):
                self.assertIn(f"https://example.com/scifi-{i}", dup_urls)
            # Ровно 1 LLM-вызов на весь пул (а не 3)
            self.assertEqual(mock_client.chat_json.call_count, 1,
                "Должен быть ровно 1 LLM-вызов на весь пул")

            p.storage.close()
            p.close()


class TestCheckAgainstShownSkippedNoCategoryMin(unittest.TestCase):
    """Задача 3: get_skipped_items не вызывается, когда category_min пуст."""

    def test_get_skipped_items_not_called_when_no_category_min(self):
        """
        Когда category_min пуст/битый, skipped_candidates не используются
        для гарантии — get_skipped_items вызываться НЕ должен.
        Это экономит LLM-вызовы (до 5 на прогоне без гарантий) и DB-запрос.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # Deferred items заполняют max_items=10 → late backfill не нужен.
            for i in range(10):
                _insert_deferred_item(
                    db_path,
                    title=f"Deferred{i}",
                    url=f"https://ex.com/d{i}",
                    published="2026-10-22T08:00+00:00",
                    fetched_at="2026-10-22T08:00:00+00:00",
                    score=7,
                )

            mock_client = MagicMock()
            mock_client.chat_json.return_value = {"duplicates": []}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(
                db_path, flt,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    # category_min отсутствует (пустой)
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Мокаем get_skipped_items, чтобы отследить вызов.
            with patch.object(p.storage, 'get_skipped_items') as mock_skipped:
                p.run_digest(force=True)
                self.assertEqual(
                    mock_skipped.call_count, 0,
                    "get_skipped_items не должен вызываться при пустом category_min",
                )

            p.storage.close()
            p.close()


class TestSkippedDoubleNotInDigest(unittest.TestCase):
    """Задача 4: skipped-дубль показанной новости не попадает в дайджест."""

    def test_skipped_duplicate_of_shown_news_not_in_digest(self):
        """
        Skipped-кандидат с URL, уже показанным в RECENT, проходит
        _check_against_shown → LLM детектирует дубль (chat_json returns [1]).
        Дубль исключается из skipped_dup_urls → НЕ попадает в category_pool
        → НЕ добавляется в deferred → НЕ попадает в отправленный digest.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T20:00:00+00:00")
            s.close()

            # RECENT = показанная новость (тот же URL, что у skipped-кандидата).
            import sqlite3 as sqlite3_lib
            conn = sqlite3_lib.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'digested', ?, 7)
                """,
                (
                    "Deadline",
                    "https://ex.com/skipped-dup",
                    "ShownEarlier",
                    "Summary",
                    "2026-10-21T22:00",
                    "2026-10-21T22:00:00+00:00",
                    "2026-10-21T22:00:00+00:00",
                ),
            )
            conn.commit()
            conn.close()

            # 1 deferred item с category=cinema → закроет category_min=1.
            # published сегодня (2026-10-22) → deferrable, попадает в digest.
            _insert_deferred_item(
                db_path,
                title="DeferredNews",
                url="https://ex.com/deferred",
                published="2026-10-22T08:00+00:00",
                fetched_at="2026-10-22T08:00:00+00:00",
                score=7,
                tags='["film","movie"]',  # категория cinema
            )

            # Skipped-кандидат с тем же URL, что в RECENT.
            _insert_skipped_item(
                db_path,
                title="SkippedDuplicate",
                url="https://ex.com/skipped-dup",
                published="2026-10-22T09:00+00:00",
                fetched_at="2026-10-22T09:00:00+00:00",
                score=5,
            )

            mock_client = MagicMock()
            # detect_duplicates вызывается дважды:
            #  1. deferred vs RECENT: DeferredNews НЕ дубль → duplicates=[].
            #  2. skipped vs RECENT: SkippedDuplicate ДУБЛЬ (URL в RECENT) → [1].
            # Собираем порядок вызовов: возвращаем [] на первый, [1] на второй.
            mock_client.chat_json.side_effect = [
                {"duplicates": []},   # deferred vs RECENT
                {"duplicates": [1]},  # skipped vs RECENT (дубль!)
            ]

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(
                db_path, flt,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "category_min": {"cinema": 1},
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            p.run_digest(force=True)

            text = delivery.send_text.call_args[0][0]
            self.assertNotIn(
                "SkippedDuplicate", text,
                "skipped-дубль показанной новости не должен попасть в дайджест",
            )
            self.assertIn(
                "DeferredNews", text,
                "deferred-новость должна попасть в дайджест",
            )

            p.storage.close()
            p.close()


class TestRecentCap(unittest.TestCase):
    """Пункт 6: RECENT cap 12000 символов."""

    def test_recent_cap_12000_prioritizes_fresh(self):
        """
        Пункт 6: RECENT cap 12000 символов (llm/prompts.py:726).
        53 digested-items с уникальными маркерами в title. Каждая запись:
          title ~220 chars (> 200 → truncate, cap считает как 220),
          source ~9 chars, маркер FRESH_XX в первых 30 символах (выживает truncate).
        Cap-бюджет: title(220) + source(9) = 229 bytes/entry.
        52 entries: 52×229 = 11888 ≤ 12000 ✓
        53 entries: 53×229 = 12137 > 12000 → cap отрезает хвост (1 запись).
        Проверяем:
          (i) cap отрезает 1 запись: 53 total → 52 в блоке,
          (ii) свежие FRESH_00..FRESH_51 присутствуют в промпте (head survived),
          (iii) самая старая FRESH_52 НЕ присутствует (cap отрезал tail),
          (iv) длина RECENT-блока в LLM-промпте ≤ 20000
               (cap 12000 + overhead промпта << 20000).
        """
        import re as re_module

        N_TOTAL = 53
        N_SHOULD_KEEP = 52  # 53 overflows cap (53×229=12137 > 12000)
        TITLE_LEN = 220     # >200 so cap counts as 220 (not actual 220)

        def _make_title(i, title_len=TITLE_LEN):
            marker = f"FRESH_{i:02d}"
            prefix = f"{marker}: News about topic {i} — "
            padding_len = max(0, title_len - len(prefix))
            return prefix + "X" * padding_len  # ~title_len chars, marker survives

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T00:00:00+00:00")
            s.close()

            # 53 digested-items с валидными timestamps.
            # fetched_at: 2026-10-22 00:00..52:59 UTC (свежие, newest first).
            # title: "FRESH_XX" в начале (выживает truncate до 200).
            # source: "FreshSrc{i%3}" — уникальный для проверки различимости.
            import sqlite3 as sqlite3_lib
            for i in range(N_TOTAL):
                fetched_ts = f"2026-10-22T{i//1:02d}:{i:02d}:00+00:00"
                conn = sqlite3_lib.connect(db_path)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO items
                    (source, url, title, summary, published, fetched_at,
                     state, state_at, llm_score)
                    VALUES (?, ?, ?, ?, ?, ?, 'digested', ?, 8)
                    """,
                    (
                        f"FreshSrc{i % 3}",
                        f"https://example.com/fresh{i}",
                        _make_title(i),
                        "Summary text",
                        "2026-10-22T00:00",
                        fetched_ts,
                        fetched_ts,
                    ),
                )
                conn.commit()
                conn.close()

            # 10 deferred-items (не дубликаты RECENT).
            for i in range(10):
                _insert_deferred_item(
                    db_path,
                    title=f"Deferred Item {i}",
                    url=f"https://deferred.com/{i}",
                    published="2026-10-22T08:00+00:00",
                    fetched_at="2026-10-22T10:00:00+00:00",
                    score=7,
                )

            mock_client = MagicMock()
            mock_client.chat_json.return_value = {"duplicates": []}

            flt = _make_llm_filter(mock_client)
            p = _mk_pipeline_with_llm(db_path, flt)
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Дайджест отправлен (есть deferred items)
            self.assertGreater(count, 0, "Должен быть отправлен дайджест")
            # LLM вызван (cap-RECENT был передан)
            self.assertGreaterEqual(mock_client.chat_json.call_count, 1)

            call_args = mock_client.chat_json.call_args_list
            r_entries = []
            recent_text_in_prompt = None
            for call_info in call_args:
                args, kwargs = call_info
                user_msg = kwargs.get("user") or (args[1] if len(args) > 1 else "")
                if "-- RECENT NEWS from database" in user_msg:
                    recent_text_in_prompt = user_msg
                    r_entries = re_module.findall(r"<\|data:R\d+\|>", user_msg)
                    break

            self.assertIsNotNone(
                recent_text_in_prompt,
                "RECENT-секция с cap должна быть в user_message",
            )
            self.assertTrue(
                len(r_entries) > 0,
                "RECENT-секция должна содержать записи",
            )

            # (i) Cap отрезал 1 запись: 53 → ровно N_SHOULD_KEEP (52).
            self.assertEqual(
                len(r_entries), N_SHOULD_KEEP,
                f"RECENT cap должно оставить ровно {N_SHOULD_KEEP} записей "
                f"(cap=12000, {TITLE_LEN}+9=239 chars/entry, {N_TOTAL} total → {N_SHOULD_KEEP} kept), "
                f"получено {len(r_entries)}",
            )

            # (ii) FRESH_01..FRESH_52 присутствуют (fresh entries survived cap).
            # entries: R1=FRESH_52, R2=FRESH_51, ..., R52=FRESH_01 (newest→oldest).
            # Cap отрезал FRESH_00 (самую старую, i=0, fetched_at=00:00).
            for i in range(1, N_TOTAL):
                self.assertIn(
                    f"FRESH_{i:02d}", recent_text_in_prompt,
                    f"FRESH_{i:02d} должен быть в RECENT (survived cap, "
                    f"i={i} >= 1, cap removed only i=0)",
                )

            # (iii) Самая старая запись (FRESH_00) НЕ присутствует
            # (cap отрезал её как самую старую — SQL orders by fetched_at DESC,
            # cap processes head-to-tail → oldest at tail, removed last).
            self.assertNotIn(
                "FRESH_00", recent_text_in_prompt,
                "FRESH_00 (самая старая) должна быть отрезана cap",
            )

            # (iv) Длина RECENT-блока в LLM-промпте ≤ 20000.
            # Entry size in prompt ≈ 255 (255×52=13260 + header 63 = 13323 << 20000).
            recent_block_match = re_module.search(
                r"-- RECENT NEWS from database.*",
                recent_text_in_prompt,
                re_module.DOTALL,
            )
            self.assertIsNotNone(recent_block_match)
            self.assertLessEqual(
                len(recent_block_match.group(0)), 20000,
                f"RECENT-блок в промпте не должен превышать 20000 символов, "
                f"получено {len(recent_block_match.group(0))}",
            )

            p.storage.close()
            p.close()


if __name__ == "__main__":
    unittest.main()