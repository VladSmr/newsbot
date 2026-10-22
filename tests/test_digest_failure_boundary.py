"""
Регрессионные тесты по вердикту критика (C1 + M1 + Minor m1 + Nit).

C1: неуспешная отправка дайджеста НЕ сдвигает digest_window_started_at —
    накопленный deferred-бэклог остаётся видим для следующего дайджеста.
M1: исключение на одном сообщении дайджеста не обрывает остальные.
Minor m1: groups и item_buttons_per_msg фильтруются синхронно — кнопки
    👍/👎 не съезжают при удалении пустой НЕ последней группы.
Nit: «N из TOTAL» считает только фактически отрендеренные items;
    title=None в except-блоке не даёт TypeError.

Второй раунд ревью (m1 Major + Minor 1 + Minor 2):
m1: выбитый feedback-target item отсутствует И в тексте сообщения, И в
    кнопках (текст пересобирается из оставшихся items).
Minor 1: счётчик 3-strike инкрементируется только при sent_count == 0
    AND total_to_send > 0 (частичный успех окно не пропускает).
Minor 2: если set_state('failed') в ветке «нет сообщений» падает —
    граница окна НЕ двигается.
"""
import logging
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
from newsbot.digest import build_digest_messages, run_digest
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
    published="2025-01-01 12:00",
    **kwargs,
) -> NewsItem:
    if url is None:
        url = f"https://example.com/{abs(hash(title)) & 0xFFFF:05d}"
    return NewsItem(
        source=source, url=url, title=title, summary=summary,
        published=published, llm_score=llm_score, **kwargs,
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
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
        """,
        ("TestSrc", url, title, "Test summary", published, fetched_at,
         fetched_at, score),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, digest_cfg=None):
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": 15,
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
            "max_per_location": 2,
        },
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    p.storage = Storage(db_path)
    return p


# =============================================================================
# C1: неуспешный дайджест не двигает границу окна
# =============================================================================

class TestFailureDoesNotMoveWindowBoundary(unittest.TestCase):
    """C1: 3 неуспешных дайджеста подряд — граница окна остаётся прежней."""

    def test_three_failures_keep_boundary_and_backlog(self):
        old_boundary = "2020-01-01T00:00:00+00:00"
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", old_boundary)
            s.close()

            _insert_deferred_item(
                db_path,
                title="BacklogItem",
                url="https://ex.com/backlog",
                published="2025-01-01 12:00",
                fetched_at="2025-01-01T12:00:00+00:00",
                score=6,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            delivery.send_text.side_effect = RuntimeError("telegram unavailable")
            p.delivery = delivery

            for i in range(3):
                count = p.run_digest(force=True)
                self.assertEqual(count, 0, f"попытка {i + 1}: ничего не отправлено")
                boundary = p.storage.get_meta("digest_window_started_at")
                self.assertEqual(
                    boundary, old_boundary,
                    f"C1: попытка {i + 1} не должна двигать границу окна "
                    f"(было {old_boundary}, стало {boundary})",
                )

            # 3-strike: счётчик сброшен, но items по-прежнему deferred
            self.assertEqual(
                int(p.storage.get_meta("digest_failure_count")), 0,
                "После 3-strike счётчик должен сбрасываться",
            )
            self.assertIn(
                "https://ex.com/backlog",
                {i.url for i in p.storage.get_all_by_state("deferred")},
            )

            # ...и бэклог виден следующему дайджесту: published-фильтр
            # (грейс = poll_interval) его НЕ отсекает.
            delivery.send_text.side_effect = None
            delivery.send_text.return_value = True
            count = p.run_digest(force=True)
            self.assertEqual(
                count, 1,
                "C1: после сбоев бэклог должен попасть в следующий дайджест",
            )
            self.assertIn(
                "https://ex.com/backlog",
                {i.url for i in p.storage.get_all_by_state("digested")},
            )
            p.storage.close()
            p.close()


# =============================================================================
# M1: изоляция сообщений дайджеста
# =============================================================================

class TestPerMessageIsolation(unittest.TestCase):
    """M1: падение одного сообщения не отменяет остальные."""

    def test_exception_on_second_message_sends_first_and_third(self):
        with _tmp_db() as db_path:
            storage = Storage(db_path)
            urls = [f"https://ex.com/m{i}" for i in range(3)]
            for i, url in enumerate(urls):
                storage.save(_mk_item(title=f"M{i}", url=url, llm_score=9 - i))
                storage.set_state([url], "deferred")

            items = storage.get_deferred_items(limit=10)
            items.sort(key=lambda x: -(x.llm_score or 0))
            groups, _, _ = build_digest_messages(items, max_items_per_message=1)
            self.assertEqual(len(groups), 3)

            delivery = MagicMock()
            delivery.send_digest.side_effect = [
                True,
                RuntimeError("boom on msg 2"),
                True,
            ]
            buttons = [[{"target_id": i + 1}] for i in range(3)]

            count, sent_urls, failed = run_digest(
                storage, delivery, max_items=10, max_items_per_message=1,
                item_buttons_per_msg=buttons, groups=groups,
            )

            self.assertEqual(count, 2, "M1: должны уйти 1-е и 3-е сообщения")
            self.assertEqual(delivery.send_digest.call_count, 3)
            self.assertEqual(len(sent_urls), 2)
            for i in (0, 2):
                self.assertIn(urls[i], sent_urls)
                self.assertIn(
                    urls[i],
                    {x.url for x in storage.get_all_by_state("digested")},
                )
            # 2-й item остался deferred (at-least-once)
            self.assertIn(
                urls[1],
                {x.url for x in storage.get_all_by_state("deferred")},
            )
            self.assertEqual(failed, [])
            storage.close()


# =============================================================================
# Minor m1: синхронная фильтрация groups + кнопок
# =============================================================================

class TestButtonsStayAlignedAfterEmptyGroup(unittest.TestCase):
    """Minor m1: пустая средняя группа не смещает кнопки на чужие новости."""

    def test_middle_empty_group_buttons_map_to_own_items(self):
        from newsbot.delivery.telegram import TelegramDelivery

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "First", "https://ex.com/first",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )
            # title=None — заодно проверяем Nit (title or "")[:60] в логе
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at, state,
                 state_at, llm_score)
                VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
                """,
                ("TestSrc", "https://ex.com/middle", None, "S",
                 "2025-01-01 12:00", "2025-01-01T12:00:00+00:00",
                 "2025-01-01T12:00:00+00:00", 8),
            )
            conn.commit()
            conn.close()
            _insert_deferred_item(
                db_path, "Third", "https://ex.com/third",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=7,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 1, "max_per_location": 2,
                },
            )
            p.config["feedback"] = {"enabled": True, "digest_enabled": True}

            delivery = MagicMock(spec=TelegramDelivery)
            delivery.send_digest.return_value = True
            p.delivery = delivery

            target_ids = {
                "https://ex.com/first": 11,
                "https://ex.com/third": 33,
            }

            def target(item, kind):
                if item.url == "https://ex.com/middle":
                    raise RuntimeError("DB down")
                return target_ids[item.url]

            with patch.object(
                p.storage, "get_or_create_feedback_target_from_item",
                side_effect=target,
            ):
                count = p.run_digest(force=True)

            self.assertEqual(count, 2)
            self.assertEqual(delivery.send_digest.call_count, 2,
                             "средняя (опустевшая) группа не отправляется")
            calls = delivery.send_digest.call_args_list
            # 1-е сообщение — кнопка First, 2-е — кнопка Third
            self.assertEqual(calls[0][0][1][0]["target_id"], 11)
            self.assertEqual(calls[1][0][1][0]["target_id"], 33)
            self.assertIn("First", calls[0][0][0])
            self.assertIn("Third", calls[1][0][0])
            p.storage.close()
            p.close()


# =============================================================================
# Nit: total только по отрендеренным
# =============================================================================

class TestRenderedTotalCounter(unittest.TestCase):
    """Nit: знаменатель «N из TOTAL» — только отрендеренные items."""

    def test_total_excludes_render_failures(self):
        good = [
            _mk_item(title="Good0", url="https://ex.com/g0", llm_score=9),
            _mk_item(title="Good1", url="https://ex.com/g1", llm_score=8),
        ]
        bad = _mk_item(title=None, url="https://ex.com/bad", llm_score=7)

        real_render = __import__(
            "newsbot.digest", fromlist=["_render_item_raw"]
        )._render_item_raw

        def fake_render(item):
            if item.url == "https://ex.com/bad":
                raise RuntimeError("render boom")
            return real_render(item)

        with patch("newsbot.digest._render_item_raw", side_effect=fake_render):
            groups, included, failed = build_digest_messages(
                [good[0], bad, good[1]], max_items_per_message=20,
            )

        self.assertEqual(failed, ["https://ex.com/bad"])
        self.assertEqual(len(included), 2)
        msg = groups[0][0]
        self.assertIn("1–2 из 2 новостей", msg)
        self.assertNotIn("из 3 новостей", msg)


# =============================================================================
# m1 (Major): выбитый item не виден в тексте и кнопках
# =============================================================================

class TestFailedTargetItemRemovedFromText(unittest.TestCase):
    """m1: item с упавшим feedback-target выбивается из текста И кнопок."""

    def test_middle_item_excluded_from_text_and_buttons(self):
        from newsbot.delivery.telegram import TelegramDelivery

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "FirstNews", "https://ex.com/first",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )
            _insert_deferred_item(
                db_path, "MiddleNews", "https://ex.com/middle",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=8,
            )
            _insert_deferred_item(
                db_path, "ThirdNews", "https://ex.com/third",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=7,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 3, "max_per_location": 2,
                },
            )
            p.config["feedback"] = {"enabled": True, "digest_enabled": True}

            delivery = MagicMock(spec=TelegramDelivery)
            delivery.send_digest.return_value = True
            p.delivery = delivery

            def target(item, kind):
                if item.url == "https://ex.com/middle":
                    raise RuntimeError("DB down")
                return {"https://ex.com/first": 11,
                        "https://ex.com/third": 33}[item.url]

            with patch.object(
                p.storage, "get_or_create_feedback_target_from_item",
                side_effect=target,
            ):
                count = p.run_digest(force=True)

            self.assertEqual(count, 2)
            self.assertEqual(delivery.send_digest.call_count, 1,
                             "выбитый item не даёт отдельного сообщения")
            sent_msg, sent_buttons = delivery.send_digest.call_args[0][0], \
                delivery.send_digest.call_args[0][1]

            # m1: выбитого item НЕТ в тексте (ни заголовка, ни URL)
            self.assertNotIn("MiddleNews", sent_msg)
            self.assertNotIn("https://ex.com/middle", sent_msg)
            self.assertIn("FirstNews", sent_msg)
            self.assertIn("ThirdNews", sent_msg)
            # ...и заголовок перенумерован под фактический состав
            self.assertIn("1–2 из 2 новостей", sent_msg)
            self.assertNotIn("из 3 новостей", sent_msg)
            # кнопки соответствуют оставшимся items (порядок сохранён)
            self.assertEqual(
                [b["target_id"] for b in sent_buttons], [11, 33],
            )
            # выбитый item — терминальный 'failed', не дублируется
            self.assertIn(
                "https://ex.com/middle",
                {x.url for x in p.storage.get_all_by_state("failed")},
            )
            self.assertIn(
                "https://ex.com/first",
                {x.url for x in p.storage.get_all_by_state("digested")},
            )
            p.storage.close()
            p.close()


# =============================================================================
# Minor 1: счётчик 3-strike — только при полном провале доставки
# =============================================================================

class TestStrikeCounterIgnoresPartialSuccess(unittest.TestCase):
    """Minor 1: 3 частичных успеха подряд не пропускают следующее окно."""

    def test_three_partial_successes_do_not_trip_3strike(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for i in range(4):
                _insert_deferred_item(
                    db_path, f"Item{i}", f"https://ex.com/p{i}",
                    "2025-01-01 12:00", "2025-01-01T12:00:00+00:00",
                    score=9 - i,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 1, "max_per_location": 2,
                },
            )
            delivery = MagicMock()
            p.delivery = delivery

            state = {"n": 0}

            def send(msg, *args, **kwargs):
                state["n"] += 1
                if state["n"] == 1:
                    return True          # первое сообщение уходит
                raise RuntimeError("telegram flaky")  # остальные — нет

            delivery.send_text.side_effect = send

            # 3 прогона: 1/4, 1/3, 1/2 — три частичных успеха подряд
            for run in range(3):
                state["n"] = 0
                count = p.run_digest(force=True)
                self.assertEqual(count, 1, f"прогон {run + 1}: частичная доставка")
                self.assertIn(
                    p.storage.get_meta("digest_failure_count"), (None, "0"),
                    f"прогон {run + 1}: частичный сбой не инкрементирует 3-strike",
                )

            # 4-й прогон не пропущен: остался 1 item, доставка успешна
            state["n"] = 0
            self.assertGreaterEqual(len(p.storage.get_all_by_state("deferred")), 1)
            count = p.run_digest(force=True)
            self.assertEqual(count, 1,
                             "окно не должно быть пропущено после 3 частичных успехов")
            self.assertEqual(p.storage.get_meta("digest_failure_count"), "0")
            p.storage.close()
            p.close()


# =============================================================================
# Minor 2: сбой set_state('failed') не двигает границу окна
# =============================================================================

class TestSetStateFailureKeepsBoundary(unittest.TestCase):
    """Minor 2: границу окна двигает только успех."""

    def test_set_state_exception_keeps_window_boundary(self):
        old_boundary = "2020-01-01T00:00:00+00:00"
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", old_boundary)
            s.close()

            _insert_deferred_item(
                db_path, "Broken", "https://ex.com/broken",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            p = _mk_pipeline(db_path)
            delivery = MagicMock()
            p.delivery = delivery

            def boom_render(item):
                raise RuntimeError("render boom")

            with patch("newsbot.digest._render_item_raw", side_effect=boom_render):
                with patch.object(
                    p.storage, "set_state",
                    side_effect=RuntimeError("sqlite locked"),
                ):
                    count = p.run_digest(force=True)

            self.assertEqual(count, 0)
            self.assertEqual(
                p.storage.get_meta("digest_window_started_at"), old_boundary,
                "Minor 2: сбой set_state('failed') не должен двигать границу окна",
            )
            # item остался deferred (не потерян) — повторится в следующем окне
            self.assertIn(
                "https://ex.com/broken",
                {i.url for i in p.storage.get_all_by_state("deferred")},
            )
            p.storage.close()
            p.close()


# =============================================================================
# Дополнительно: частичный сбой не двигает границу
# =============================================================================

class TestPartialDeliveryKeepsBoundary(unittest.TestCase):
    """sent>0, но sent<total → граница окна не двигается."""

    def test_partial_send_keeps_boundary(self):
        old_boundary = "2020-01-01T00:00:00+00:00"
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", old_boundary)
            s.close()

            _insert_deferred_item(
                db_path, "Ok", "https://ex.com/ok",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )
            _insert_deferred_item(
                db_path, "Lost", "https://ex.com/lost",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=8,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 1, "max_per_location": 2,
                },
            )
            delivery = MagicMock()
            delivery.send_text.side_effect = [True, False]
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 1)
            self.assertEqual(
                p.storage.get_meta("digest_window_started_at"), old_boundary,
                "частичный сбой не должен двигать границу окна",
            )
            self.assertIn(
                "https://ex.com/lost",
                {i.url for i in p.storage.get_all_by_state("deferred")},
            )
            self.assertIn(
                "https://ex.com/ok",
                {i.url for i in p.storage.get_all_by_state("digested")},
            )
            # частичный сбой не инкрементирует 3-strike
            self.assertIn(p.storage.get_meta("digest_failure_count"), (None, "0"))
            p.storage.close()
            p.close()


# =============================================================================
# P2: 3-strike терминализирует poison-pill items
# =============================================================================

class TestThreeStrikeTerminalizesPoisonPills(unittest.TestCase):
    """P2: ветка digest_failure_count>=3 возвращалась ДО set_state('failed')
    для битых feedback-target items → они вечно циклятся в deferred."""

    def test_poison_pill_marked_failed_on_3strike(self):
        from newsbot.delivery.telegram import TelegramDelivery

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.set_meta("digest_failure_count", "3")
            s.close()

            _insert_deferred_item(
                db_path, "Poison", "https://ex.com/poison",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            p = _mk_pipeline(db_path)
            p.config["feedback"] = {"enabled": True, "digest_enabled": True}
            delivery = MagicMock(spec=TelegramDelivery)
            delivery.send_digest.return_value = True
            p.delivery = delivery

            def boom_target(item, kind):
                raise RuntimeError("broken feedback target")

            with patch.object(
                p.storage, "get_or_create_feedback_target_from_item",
                side_effect=boom_target,
            ):
                count = p.run_digest(force=True)

            self.assertEqual(count, 0, "3-strike: дайджест пропущен")
            self.assertIn(
                "https://ex.com/poison",
                {i.url for i in p.storage.get_all_by_state("failed")},
                "P2: poison-pill item должен быть терминализирован в 'failed'",
            )
            p.storage.close()
            p.close()


# =============================================================================
# P2: shutdown во время дайджеста — не провал доставки
# =============================================================================

class TestShutdownDoesNotCountAsFailure(unittest.TestCase):
    """P2: остановка процесса — штатная ситуация, не 3-strike."""

    def test_shutdown_does_not_increment_failure_or_write_last_digest(self):
        import threading

        old_boundary = "2020-01-01T00:00:00+00:00"
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", old_boundary)
            s.close()

            _insert_deferred_item(
                db_path, "Backlog", "https://ex.com/backlog",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            ev = threading.Event()
            p = _mk_pipeline(db_path)
            p._shutdown_event = ev
            delivery = MagicMock()
            delivery.send_text.return_value = False  # как при shutdown в _api_call
            p.delivery = delivery

            ev.set()
            for i in range(3):
                count = p.run_digest(force=True)
                self.assertEqual(count, 0)
                self.assertIn(
                    p.storage.get_meta("digest_failure_count"), (None, "0"),
                    f"прогон {i + 1}: shutdown не инкрементирует 3-strike",
                )
                self.assertIsNone(
                    p.storage.get_meta("last_digest_at"),
                    "shutdown не должен писать last_digest_at",
                )
                self.assertEqual(
                    p.storage.get_meta("digest_window_started_at"), old_boundary,
                    "shutdown не должен двигать границу окна",
                )

            # Окно НЕ пропущено: после снятия shutdown бэклог уходит.
            ev.clear()
            delivery.send_text.return_value = True
            count = p.run_digest(force=True)
            self.assertEqual(count, 1, "3 деплоя подряд не должны пропускать окно")
            self.assertIn(
                "https://ex.com/backlog",
                {i.url for i in p.storage.get_all_by_state("digested")},
            )
            p.storage.close()
            p.close()


# =============================================================================
# Внешний except-путь run_digest (:1888)
# =============================================================================

class TestExternalRunDigestException(unittest.TestCase):
    """Исключения delivery перехватываются внутри digest.run_digest; до
    внешнего обработчика доходят только аварии самого run_digest —
    плюс защитный путь при нарушении контракта. Покрываем его явно."""

    def test_run_digest_exception_keeps_boundary_and_counts_failure(self):
        old_boundary = "2020-01-01T00:00:00+00:00"
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", old_boundary)
            s.close()

            _insert_deferred_item(
                db_path, "Any", "https://ex.com/any",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            p = _mk_pipeline(db_path)
            p.delivery = MagicMock()

            with patch(
                "newsbot.pipeline._run_digest",
                side_effect=RuntimeError("contract violation"),
            ):
                count = p.run_digest(force=True)

            self.assertEqual(count, 0)
            self.assertEqual(
                p.storage.get_meta("digest_window_started_at"), old_boundary,
                "внешний except-путь не должен двигать границу окна",
            )
            self.assertEqual(p.storage.get_meta("digest_failure_count"), "1")
            self.assertIn(
                "https://ex.com/any",
                {i.url for i in p.storage.get_all_by_state("deferred")},
            )
            p.storage.close()
            p.close()


# =============================================================================
# Minor 1 (wave 2): битый category_min не роняет дайджест каждый цикл
# =============================================================================

class TestBadCategoryMinConfigDoesNotBreakDigest(unittest.TestCase):
    """YAML может отдать category_min строкой; TypeError раньше летел ВНЕ
    try вокруг _run_digest → дайджест падал каждый цикл (и 3-strike)."""

    def test_string_category_min_and_max_per_location(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "Ok", "https://ex.com/ok",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20,
                    "max_per_location": "2",       # строка
                    "category_min": {"cinema": "1"},  # строка
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertEqual(count, 1, "строковый category_min не должен ломать дайджест")
            self.assertEqual(p.storage.get_meta("digest_failure_count"), "0")
            p.storage.close()
            p.close()

    def test_invalid_category_min_values_do_not_raise(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "Ok", "https://ex.com/ok2",
                "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10,
                    "max_items_per_message": 20, "max_per_location": 2,
                    # мусорные значения: нечисловая строка и None
                    "category_min": {"cinema": "abc", "science": None},
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertEqual(count, 1)
            self.assertEqual(p.storage.get_meta("digest_failure_count"), "0")
            p.storage.close()
            p.close()


# =============================================================================
# Major (wave 4): `.nan`/`.inf` в max_per_location не роняют дайджест
# =============================================================================

class TestNonFiniteMaxPerLocationDoesNotBreakDigest(unittest.TestCase):
    """YAML `.nan`/`.inf` — валидные числа; до фикса int() бросал
    ValueError/OverflowError на строке :1713 ВНЕ try вокруг _run_digest →
    дайджест падал каждый цикл (и копил 3-strike)."""

    def test_nan_and_inf_max_per_location(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad), _tmp_db() as db_path:
                s = Storage(db_path)
                s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
                s.close()

                _insert_deferred_item(
                    db_path, "Ok", "https://ex.com/ok",
                    "2025-01-01 12:00", "2025-01-01T12:00:00+00:00", score=9,
                )

                p = _mk_pipeline(
                    db_path,
                    digest_cfg={
                        "enabled": True, "max_items": 10,
                        "max_items_per_message": 20,
                        "max_per_location": bad,
                    },
                )
                delivery = MagicMock()
                delivery.send_text.return_value = True
                p.delivery = delivery

                with self.assertLogs("newsbot.pipeline", level="WARNING") as cm:
                    count = p.run_digest(force=True)

                self.assertEqual(count, 1, f"{bad!r}: дайджест не должен падать")
                self.assertEqual(p.storage.get_meta("digest_failure_count"), "0")
                self.assertTrue(
                    any("не конечное число" in m for m in cm.output),
                    f"{bad!r}: ожидался warning, получено {cm.output}",
                )
                self.assertIn(
                    "https://ex.com/ok",
                    {i.url for i in p.storage.get_all_by_state("digested")},
                )
                p.storage.close()
                p.close()


if __name__ == "__main__":
    unittest.main()