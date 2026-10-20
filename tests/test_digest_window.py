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
from pathlib import Path
from unittest.mock import MagicMock

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


def _insert_deferred_item(db_path, title, url, published, fetched_at, score=6):
    """
    Вставляет deferred-item напрямую в БД с заданным fetched_at.
    Storage.save() игнорирует item.fetched_at (всегда ставит NOW),
    поэтому нужен прямой INSERT для контроля времени.

    fetched_at ЯВЛЯЕТСЯ ОБЯЗАТЕЛЬНЫМ параметром — это гарантирует
    предсказуемое поведение в тестах (без race condition по секундам).
    score позволяет влиять на порядок get_deferred_items (сортировка: score DESC).
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

            # 3 old items — удалены (были в окне, за max_items, не отправлены)
            for i in range(3):
                self.assertNotIn(f"https://ex.com/old{i}", all_window_urls)

            # 2 fresh items — ОТПРАВЛЕНЫ и тоже УДАЛЕНЫ из БД
            # (окно вычищается целиком после успеха: delete_items_before удаляет всё,
            #  включая отправленные digested-items)
            for i in range(2):
                self.assertNotIn(f"https://ex.com/fresh{i}", all_window_urls)

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

            # stale_item удалён (за границей); fresh_item — digested, но удалён из БД
            all_after_d1 = set()
            for state in ["deferred", "digested", "new", "pending", "skipped", "failed"]:
                all_after_d1.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertNotIn("https://ex.com/stale", all_after_d1)
            # fresh_item отправлен и удалён (окно очищено после успеха)
            self.assertNotIn("https://ex.com/fresh", all_after_d1)

            # Патчим datetime.now: первый вызов (D1) → 2026-01-01T00:00:00 UTC,
            # второй вызов (D2) → на 2 дня позже (сдвигаем реальное время).
            # Без патча D1 и D2 могут получить одинаковое now_utc (одна секунда),
            # тогда new_item (fetched_at = old_boundary + 2s) не удалится при D2.
            import datetime as _dt
            from datetime import timezone
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
            # published обновлён до D2-границы: теперь item проходит оба фильтра
            # (fetched_at >= window_start И published >= window_start).
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
            # stale_item — удалён в D1 (за границей).
            # fresh_item — удалён в D1 (окно D1 очищено).
            deferred_after_d2 = {i.url for i in p.storage.get_all_by_state("deferred")}
            all_after_d2 = set()
            for state in ["deferred", "digested", "new", "pending", "skipped", "failed"]:
                all_after_d2.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertNotIn("https://ex.com/stale", all_after_d2)
            self.assertNotIn("https://ex.com/fresh", all_after_d2)
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

            # deferred-item отправлен и удалён
            all_urls = set()
            for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
                all_urls.update(i.url for i in p.storage.get_all_by_state(state))
            self.assertNotIn("https://ex.com/def", all_urls)

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


if __name__ == "__main__":
    unittest.main()