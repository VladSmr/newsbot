"""
Тесты стадийных счётчиков и всех M-фиксов в pipeline.

Покрывает:
  M1: cluster-max score поднимает score представителя до маршрутизации.
  M2: грейс published-фильтра (poll_interval).
  M3: poison pill — per-item failed + digest_failure_count (3 подряд).
  M4: backfill после cap.
  M5: нормализация source в _infer_category.
  M6: INFO-лог стадиальных счётчиков в run_digest.

Запуск: python -m pytest tests/test_pipeline_stages.py -v
"""
import logging
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from functools import wraps

import pytest

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.pipeline import NewsPipeline, _infer_category
from newsbot.digest import _render_item_raw as _real_render_item_raw
from newsbot.digest import build_digest_messages
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
    cluster_key=None,
    published="2025-01-01 12:00",
    published_parsed=None,
    **kwargs,
) -> NewsItem:
    if url is None:
        url = f"https://example.com/{hash(title) & 0xFFFF:05d}"
    item = NewsItem(
        source=source, url=url, title=title, summary=summary,
        published=published, published_parsed=published_parsed,
        llm_score=llm_score, **kwargs,
    )
    if cluster_key is not None:
        item.cluster_key = cluster_key
    return item


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


def _insert_deferred_item(db_path, title, url, published, fetched_at, score=6,
                           cluster_key=None, tags=None):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score, tags, cluster_key)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?, ?, ?)
        """,
        ("TestSrc", url, title, "Test summary", published, fetched_at, fetched_at, score, tags, cluster_key),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, digest_cfg=None, poll_interval_minutes=None):
    """
    poll_interval_minutes: Major-4-фикс — передаётся через конфиг (НЕ post-patch).
    При None используется 120 (grace-тесты).
    """
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": (
            poll_interval_minutes
            if poll_interval_minutes is not None
            else 120
        ),
        "digest": digest_cfg or {
            "enabled": True, "max_items": 100, "max_items_per_message": 20,
            "max_per_location": 2,
        },
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    # Закрываем первое соединение (создано в __init__ из db_path) перед подменой
    p.storage.close()
    p.storage = Storage(db_path)
    return p


# =============================================================================
# M5: _infer_category нормализация source
# =============================================================================

class TestInferCategoryNormalization(unittest.TestCase):
    """M5: нормализация source — lowercase, без пунктуации, алиасы."""

    def test_tom_hardware_aliases(self):
        """"Tom's Hardware", "tomshardware" → tech."""
        variants = [
            "Tom's Hardware",
            "tomshardware",
            "Tom’s Hardware",  # different apostrophe
            "Toms Hardware",
            "TOMS HARDWARE",
        ]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "tech" for c in cats), f"Все Tom's Hardware → tech, got: {cats}"

    def test_habr_aliases(self):
        """"Habr", "habr news", "Хабр" → tech."""
        variants = ["Habr", "habr", "habr news", "Habr News", "HABR"]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "tech" for c in cats), f"Все Habr → tech, got: {cats}"

    def test_krebs_aliases(self):
        """"KrebsOnSecurity", "Krebs On Security", "krebs" → cyber."""
        variants = ["KrebsOnSecurity", "Krebs On Security", "krebs", "KREBS"]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "cyber" for c in cats), f"Все Krebs → cyber, got: {cats}"

    def test_sciencedaily_aliases(self):
        """"ScienceDaily", "science daily" → science."""
        variants = ["ScienceDaily", "science daily", "SCIENCE DAILY", "Sciencedaily"]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "science" for c in cats), f"Все ScienceDaily → science, got: {cats}"

    def test_statnews_aliases(self):
        """"STAT News", "STAT", "statnews" → science."""
        variants = ["STAT News", "STAT", "statnews", "Statnews"]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "science" for c in cats), f"Все STAT → science, got: {cats}"

    def test_nasASpaceflight(self):
        """"NASASpaceflight" → space."""
        variants = ["NASASpaceflight", "nasa spaceflight", "NasaSpaceflight"]
        cats = [_infer_category(_mk_item(source=s)) for s in variants]
        assert all(c == "space" for c in cats), f"Все NASASpaceflight → space, got: {cats}"

    def test_unknown_source_returns_other(self):
        """Неизвестный source → other."""
        assert _infer_category(_mk_item(source="SomeUnknownSource")) == "other"
        assert _infer_category(_mk_item(source="")) == "other"

    def test_known_sources_still_work(self):
        """Ранее работавшие source остаются рабочими."""
        assert _infer_category(_mk_item(source="Deadline")) == "cinema"
        assert _infer_category(_mk_item(source="IGN")) == "games"
        assert _infer_category(_mk_item(source="SpaceNews")) == "space"
        assert _infer_category(_mk_item(source="BleepingComputer")) == "cyber"
        assert _infer_category(_mk_item(source="Phys.org")) == "science"


# =============================================================================
# M2: грейс published-фильтра
# =============================================================================

class TestGracePublishedFilter(unittest.TestCase):
    """
    M2/Major-4: published-фильтр с грейсом на poll_interval.

    Major-4: poll_interval_minutes задаётся через _mk_pipeline (НЕ post-patch),
    чтобы фикстура создавала pipeline с правильным конфигом.
    """

    def test_grace_published_passes_grace_window(self):
        """
        Item с published на (poll_interval - 1 мин) РАНЬШЕ window_start —
        попадает в окно (грейс).
        poll_interval=120 мин, published старше window на 14 мин.
        """
        with _tmp_db() as db_path:
            # Граница: 2026-10-21T12:00:00 UTC
            # published: 2026-10-21T11:46:00 UTC (на 14 мин раньше)
            # При poll_interval=120 мин → published >= 12:00 - 120:00 = 10:00
            # 11:46 >= 10:00 → ДОЛЖЕН попасть
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T12:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path,
                title="GraceItem",
                url="https://ex.com/grace",
                published="2026-10-21T11:46:00+00:00",
                fetched_at="2026-10-21T12:00:00+00:00",
                score=8,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 20,
                },
                poll_interval_minutes=120,  # Major-4: через конфиг, не post-patch
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Item должен попасть в digest (within grace=120 мин)
            self.assertEqual(count, 1, "Item within grace=120 мин должен попасть в digest")
            p.storage.close()
            p.close()

    def test_too_old_published_filtered_despite_grace(self):
        """
        Item с published раньше window_start - poll_interval — НЕ попадает.
        poll_interval=120 мин, published старше window на 200 мин (заведомо за grace).
        """
        with _tmp_db() as db_path:
            # Граница: 2026-10-21T12:00:00 UTC
            # published: 2026-10-21T09:20:00 UTC (на 200 мин раньше)
            # При poll_interval=120 → published >= 12:00 - 120:00 = 10:00
            # 09:20 >= 10:00 → FALSE → отфильтрован
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T12:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path,
                title="TooOld",
                url="https://ex.com/tooold",
                published="2026-10-21T09:20:00+00:00",
                fetched_at="2026-10-21T12:00:00+00:00",
                score=8,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Item отфильтрован (за пределами grace)
            self.assertEqual(count, 0, "Item за пределами grace должен быть отфильтрован")
            # Item остался в deferred
            deferred = {it.url for it in p.storage.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/tooold", deferred)
            p.storage.close()
            p.close()

    def test_grace_dead_zone_60min(self):
        """
        Major-4: кейс в 'мёртвой зоне' — published на 60 мин раньше окна.

        При poll_interval=120 мин: 12:00 - 120 мин = 10:00 (grace start).
        published=11:00 — попадает (11:00 >= 10:00).

        При poll_interval=15 мин: 12:00 - 15 мин = 11:45 (grace start).
        published=11:00 — НЕ попадает (11:00 < 11:45).
        Один item, два конфига → разный результат → регресс ловится.
        """
        # Item на 60 мин раньше window (published=11:00, window=12:00)
        base_item = dict(
            title="DeadZoneItem",
            url="https://ex.com/deadzone",
            published="2026-10-21T11:00:00+00:00",
            fetched_at="2026-10-21T12:00:00+00:00",
            score=8,
        )

        # Случай 1: poll_interval=120 → внутри grace → попадает
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T12:00:00+00:00")
            s.close()
            _insert_deferred_item(db_path, **base_item)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 20,
                },
                poll_interval_minutes=120,
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertEqual(
                count, 1,
                "Item на 60 мин раньше окна при grace=120 мин должен попасть",
            )
            p.storage.close()
            p.close()

        # Случай 2: poll_interval=15 → за пределами grace → НЕ попадает
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T12:00:00+00:00")
            s.close()
            _insert_deferred_item(db_path, **base_item)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 20,
                },
                poll_interval_minutes=15,  # Major-4: задаём через конфиг
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)
            self.assertEqual(
                count, 0,
                "Item на 60 мин раньше окна при grace=15 мин должен быть отфильтрован",
            )
            # Item остался в deferred
            deferred = {it.url for it in p.storage.get_all_by_state("deferred")}
            self.assertIn("https://ex.com/deadzone", deferred)
            p.storage.close()
            p.close()

    def test_grace_with_default_poll_interval(self):
        """
        При poll_interval=120 мин grace-окно = 120 мин.
        Item на 10 мин раньше window — внутри grace → попадает.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2026-10-21T12:00:00+00:00")
            s.close()

            # published на 10 мин раньше window — внутри grace=120 мин
            # grace_delta = 120 мин; ws_grace = 12:00 - 120 мин = 10:00
            # 11:50 >= 10:00 → попадает
            _insert_deferred_item(
                db_path,
                title="DefaultGrace",
                url="https://ex.com/defgrace",
                published="2026-10-21T11:50:00+00:00",
                fetched_at="2026-10-21T12:00:00+00:00",
                score=8,
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 100, "max_items_per_message": 20,
                    # poll_interval_minutes=120 из _mk_pipeline fixture
                    # → grace_period = 120 мин. Item на 10 мин внутри grace.
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # 11:50 >= 10:00 (grace=120 мин) → попадает
            self.assertEqual(count, 1, "Item на 10 мин внутри grace=120 мин должен попасть")
            p.storage.close()
            p.close()


# =============================================================================
# M4: backfill после cap
# =============================================================================

class TestBackfillAfterCap(unittest.TestCase):
    """M4: добор из пула после location-cap."""

    def test_backfill_fills_free_slots_after_cap(self):
        """
        max_items=3. 3 items в топе по score, но все из одной локации.
        Cap=2 на локацию → 1 item выпадает. Backfill добирает из другой локации.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 3 items все из France/2026-01-01 → cap оставит только 2
            for i, score in enumerate([9, 8, 7]):
                _insert_deferred_item(
                    db_path,
                    title=f"Top{i}", url=f"https://ex.com/top{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                    cluster_key="France|abc|2026-01-01",
                )
            # 1 item из другой локации (не France) — в deferrable_post_published
            # за пределами top-3, но в пуле (score 6)
            _insert_deferred_item(
                db_path,
                title="OtherRegion",
                url="https://ex.com/other",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=6,
                cluster_key="Russia|def|2026-01-01",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 3, "max_items_per_message": 20,
                    "max_per_location": 2,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Cap: France top-2 = Top0, Top1. Backfill: OtherRegion добирается → 3 items
            self.assertEqual(count, 3, f"Expected 3 (cap=2 + backfill=1), got {count}")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("OtherRegion", text, "Backfilled item должен быть в digest")
            self.assertNotIn("Top2", text, "Cap'нутый item Top2 не должен быть в digest")
            p.storage.close()
            p.close()

    def test_backfill_respects_other_location_cap(self):
        """
        Backfill НЕ берёт из (локация, день) которая уже исчерпала cap.
        France cap=2, два France в топе → France исчерпан, backfill не берёт
        из France даже если там свободный слот по score.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 2 France items в топе (France cap=2 → исчерпан)
            for i, score in enumerate([9, 8]):
                _insert_deferred_item(
                    db_path,
                    title=f"France{i}", url=f"https://ex.com/fr{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                    cluster_key="France|abc|2026-01-01",
                )
            # 1 France item за пределами top-2 (не France!) — не в топе, в пуле
            _insert_deferred_item(
                db_path,
                title="France3",
                url="https://ex.com/fr3",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=5,
                cluster_key="France|def|2026-01-01",
            )
            # 1 Russia item в пуле (не France, локация не исчерпана)
            _insert_deferred_item(
                db_path,
                title="Russia",
                url="https://ex.com/ru",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=6,
                cluster_key="Russia|ghi|2026-01-01",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 3, "max_items_per_message": 20,
                    "max_per_location": 2,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # France cap исчерпан → France3 не берётся. Russia берётся.
            self.assertEqual(count, 3)
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Russia", text)
            self.assertNotIn("France3", text,
                "France3 не должен попасть: France cap исчерпан")
            p.storage.close()
            p.close()

    def test_backfill_does_not_evict_guarantee_items(self):
        """
        Guarantee-items не вытесняются backfill'ом.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 2 non-cinema items в топе
            for i, score in enumerate([9, 8]):
                _insert_deferred_item(
                    db_path,
                    title=f"Tech{i}", url=f"https://ex.com/t{i}",
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                )
            # 1 cinema item за пределами top-3, но в пуле
            _insert_deferred_item(
                db_path,
                title="CinemaItem",
                url="https://ex.com/cinema",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=4,
                tags='["film","movie"]',
                cluster_key="USA|abc|2026-01-01",
            )
            # 1 non-cinema в пуле для backfill
            _insert_deferred_item(
                db_path,
                title="PoolTech",
                url="https://ex.com/pool",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=5,
                cluster_key="Russia|def|2026-01-01",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 3, "max_items_per_message": 20,
                    "category_min": {"cinema": 1},
                    "max_per_location": 5,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Гарантия cinema=1: CinemaItem вытеснил Tech1.
            # После cap: Tech0(9), Tech1(8→evicted), CinemaItem(4). Backfill: PoolTech(5).
            # PoolTech(5) вытеснил Tech1(8→evicted) — это гарантийный swap, ok.
            text = delivery.send_text.call_args[0][0]
            self.assertIn("CinemaItem", text, "Cinema item гарантии должен быть в digest")
            p.storage.close()
            p.close()


# =============================================================================
# M3: poison pill + digest_failure_count
# =============================================================================

class TestDigestPoisonPill(unittest.TestCase):
    """M3: битый item изолируется; 3 падения подряд — окно пропущено."""

    def test_item_raising_exception_marked_failed(self):
        """
        MAJOR 2a-фикс: per-item исключение при создании feedback_target
        (get_or_create_feedback_target_from_item) → item помечается 'failed'.
        Битый item НЕ возвращается в следующие окна.
        NOTE: should_feedback=True требует isinstance(delivery, TelegramDelivery).
        Используем spec=MagicMock+spec_set чтобы isinstance работал.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            bad_item = _mk_item(
                title="BadItem", url="https://ex.com/bad",
                llm_score=6, published="2025-01-01 12:00",
            )
            ok_item = _mk_item(
                title="OKItem", url="https://ex.com/ok",
                llm_score=7, published="2025-01-01 12:00",
            )
            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 5, "max_items_per_message": 20,
                },
            )
            p.storage.save(bad_item)
            p.storage.save(ok_item)
            p.storage.set_state(["https://ex.com/bad", "https://ex.com/ok"], "deferred")

            # MAJOR 2a: should_feedback=True (isinstance требует TelegramDelivery).
            # delivery_mock isinstance → True via spec, но send_text не реальный.
            from unittest.mock import MagicMock as _MM
            delivery = _MM()  # bare MagicMock — isinstance любого класса = False
            # Правильно: создаём mock с spec, чтобы isinstance работал
            from newsbot.delivery.telegram import TelegramDelivery
            delivery = _MM(spec=TelegramDelivery)
            delivery.send_text.return_value = True
            p.delivery = delivery

            # feedback.enabled=True + digest_enabled=True → should_feedback=True
            p.config["feedback"] = {"enabled": True, "digest_enabled": True}

            def bad_target(item, kind):
                if item.url == "https://ex.com/bad":
                    raise RuntimeError("DB connection failed")
                return 1

            with patch.object(
                p.storage, "get_or_create_feedback_target_from_item", side_effect=bad_target
            ):
                with patch.object(
                    p.storage, "set_state", wraps=p.storage.set_state
                ) as mock_set_state:
                    count1 = p.run_digest(force=True)

            # OK-item отправлен
            self.assertGreaterEqual(count1, 1)
            # set_state('failed') вызван для bad_item (MAJOR 2a)
            failed_calls = [
                c for c in mock_set_state.call_args_list if c[0][1] == "failed"
            ]
            self.assertEqual(len(failed_calls), 1,
                f"set_state('failed') должен быть вызван 1 раз: {mock_set_state.call_args_list}"
            )
            self.assertIn("https://ex.com/bad", failed_calls[0][0][0])

            # bad_item в БД — state='failed'
            all_failed = {i.url for i in p.storage.get_all_by_state("failed")}
            self.assertIn("https://ex.com/bad", all_failed,
                "bad_item должен быть state='failed'")
            # ok_item — digested
            all_digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn("https://ex.com/ok", all_digested)

            p.storage.close()
            p.close()

    def test_exception_in_send_path_increments_failure_count(self):
        """
        MAJOR 2b-фикс: исключение в _run_digest → failure_count++.
        После 3 неудачных попыток — окно пропущено, last_digest_at обновлён,
        items остаются deferred (НЕ 'failed' — это per-item, не global failure).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            item = _mk_item(
                title="PersistItem",
                url="https://ex.com/persist",
                llm_score=6,
                published="2025-01-01 12:00",
            )
            p = _mk_pipeline(db_path)
            p.storage.save(item)
            p.storage.set_state(["https://ex.com/persist"], "deferred")

            delivery = MagicMock()
            delivery.send_text.side_effect = RuntimeError("network error")
            p.delivery = delivery

            # 1-е падение
            count1 = p.run_digest(force=True)
            self.assertEqual(count1, 0)
            self.assertEqual(int(p.storage.get_meta("digest_failure_count")), 1)
            # Item остался deferred (global failure не помечает items)
            self.assertIn(
                "https://ex.com/persist",
                {i.url for i in p.storage.get_all_by_state("deferred")}
            )

            # 2-е падение
            count2 = p.run_digest(force=True)
            self.assertEqual(count2, 0)
            self.assertEqual(int(p.storage.get_meta("digest_failure_count")), 2)

            # 3-е падение — logged as error (>=3), счётчик сброшен,
            # last_digest_at обновлён, items остаются deferred.
            count3 = p.run_digest(force=True)
            self.assertEqual(count3, 0)
            self.assertEqual(
                int(p.storage.get_meta("digest_failure_count")), 0,
                "После 3 падений счётчик должен быть сброшен"
            )
            last_digest = p.storage.get_meta("last_digest_at")
            self.assertIsNotNone(last_digest,
                "last_digest_at должен быть записан после пропуска окна")
            # Item остался deferred
            self.assertIn(
                "https://ex.com/persist",
                {i.url for i in p.storage.get_all_by_state("deferred")}
            )

            p.storage.close()
            p.close()

    def test_success_resets_failure_count(self):
        """
        Успешный дайджест сбрасывает digest_failure_count.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.set_meta("digest_failure_count", "2")
            s.close()

            item = _mk_item(
                title="OKItem",
                url="https://ex.com/ok",
                llm_score=6,
                published="2025-01-01 12:00",
            )
            p = _mk_pipeline(db_path)
            p.storage.save(item)
            p.storage.set_state(["https://ex.com/ok"], "deferred")

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            self.assertEqual(count, 1)
            failure_count = p.storage.get_meta("digest_failure_count")
            self.assertEqual(int(failure_count), 0,
                "Успешный дайджест должен сбросить digest_failure_count")

            p.storage.close()
            p.close()


# =============================================================================
# M6: стадийные счётчики в INFO-логе
# =============================================================================

class TestStageCountersLog:
    """M6: caplog содержит строку со счётчиками стадий."""

    def test_stage_counters_logged(self, caplog):
        """run_digest логирует sql→published→top_n→dedup→backfill→guarantee→cap→final."""
        import logging

        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for i, score in enumerate([8, 7, 6]):
                _insert_deferred_item(
                    db_path,
                    title=f"Item{i}", url=f"https://ex.com/s{i}",
                    published="2025-01-01 12:00",
                    fetched_at="2025-01-01T12:00:00+00:00",
                    score=score,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 10, "max_items_per_message": 20,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            with caplog.at_level(logging.INFO, logger="newsbot.pipeline"):
                p.run_digest(force=True)

            # Ищем строку со счётчиками стадий
            stage_logs = [
                r for r in caplog.records
                if r.levelno >= logging.INFO
                and "sql=" in r.message
                and "published=" in r.message
                and "top_n=" in r.message
                and "dedup=" in r.message
            ]
            assert len(stage_logs) >= 1, (
                f"Expected INFO со стадиями, got: {[r.message for r in caplog.records]}"
            )
            # Проверяем что все 8 стадий присутствуют
            msg = stage_logs[0].message
            assert "sql=" in msg
            assert "published=" in msg
            assert "top_n=" in msg
            assert "dedup=" in msg
            assert "backfill=" in msg
            assert "guarantee=" in msg
            assert "cap=" in msg
            assert "final=" in msg

            p.storage.close()
            p.close()


# =============================================================================
# M1: cluster-max score применяется ДО маршрутизации
# =============================================================================

class TestClusterMaxBeforeRouting(unittest.TestCase):
    """M1: cluster-max score поднимает score представителя до route()."""

    def test_cluster_max_in_digest(self):
        """
        M1: 2 items в одном кластере (leader score=8, dup score=9).
        После cluster-max оба получают score=9. Оба попадают в digest.
        Cap по локации не ограничивает — max_per_location=5.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for url, title, score, ck in [
                ("https://ex.com/leader", "Leader", 8, "France|abc|2025-01-01"),
                ("https://ex.com/dup", "Dup", 9, "France|abc|2025-01-01"),
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2025-01-01 12:00",
                    fetched_at="2025-01-01T12:00:00+00:00",
                    score=score, cluster_key=ck,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 5, "max_items_per_message": 20,
                    "max_per_location": 5,
                    "cluster_duplicates": False,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # Оба в digest
            self.assertEqual(count, 2, f"Expected 2 in digest, got {count}")
            text = delivery.send_text.call_args[0][0]
            self.assertIn("Leader", text)
            self.assertIn("Dup", text)

            p.storage.close()
            p.close()


# =============================================================================
# CRITICAL-фикс: failed_digest_urls инициализация до использования,
# build_failed → state='failed' при early-return
# =============================================================================

class TestCriticalRenderExceptionNoUnboundError(unittest.TestCase):
    """
    CRITICAL: render exception → state='failed', остальные доставлены,
    UnboundLocalError не возникает.

    NOTE: build_digest_messages вызывает _render_item_raw на каждом item.
    Патчим _render_item_raw напрямую (модуль digest) — это функция,
    которую невозможно обмануть через patch.object на pipeline.
    """

    def test_render_exception_item_marked_failed_others_delivered(self):
        """
        3 items: 1 бросает исключение в _render_item_raw, 2 нормальных.
        Битый item → state='failed'. Остальные отправлены → state='digested'.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            bad_item = _mk_item(
                title="BadItem", url="https://ex.com/bad",
                llm_score=8, published="2025-01-01 12:00",
            )
            ok1 = _mk_item(
                title="OK1", url="https://ex.com/ok1",
                llm_score=7, published="2025-01-01 12:00",
            )
            ok2 = _mk_item(
                title="OK2", url="https://ex.com/ok2",
                llm_score=6, published="2025-01-01 12:00",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 5, "max_items_per_message": 20,
                },
            )
            p.storage.save(bad_item)
            p.storage.save(ok1)
            p.storage.save(ok2)
            p.storage.set_state(
                ["https://ex.com/bad", "https://ex.com/ok1", "https://ex.com/ok2"],
                "deferred",
            )

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Патчим _render_item_raw в модуле digest — бросает для bad_item.
            # Импортируем оригинал заранее, чтобы избежать рекурсии.
            def fake_render(item):
                if item.url == "https://ex.com/bad":
                    raise RuntimeError("render failed")
                return _real_render_item_raw(item)

            with patch("newsbot.digest._render_item_raw", side_effect=fake_render):
                count = p.run_digest(force=True)

            # 2 нормальных item отправлены
            self.assertEqual(count, 2,
                f"Expected 2 items delivered, got {count}")
            # bad_item → state='failed'
            all_failed = {i.url for i in p.storage.get_all_by_state("failed")}
            self.assertIn("https://ex.com/bad", all_failed,
                "bad_item должен быть state='failed'")
            # ok1, ok2 → state='digested'
            all_digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn("https://ex.com/ok1", all_digested)
            self.assertIn("https://ex.com/ok2", all_digested)
            # Не было UnboundLocalError (run_digest завершился)
            text = delivery.send_text.call_args[0][0]
            self.assertIn("OK1", text)
            self.assertIn("OK2", text)
            self.assertNotIn("BadItem", text)

            p.storage.close()
            p.close()

    def test_all_items_fail_early_return_marks_failed(self):
        """
        CRITICAL-фикс (3): ВСЕ items падают при рендеринге → early-return,
        но ДО return вызывается set_state(_build_failed, 'failed').
        Items НЕ отправлены, граница сдвинута, ничто не зависает в deferred.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            item1 = _mk_item(
                title="Fail1", url="https://ex.com/fail1",
                llm_score=8, published="2025-01-01 12:00",
            )
            item2 = _mk_item(
                title="Fail2", url="https://ex.com/fail2",
                llm_score=7, published="2025-01-01 12:00",
            )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 5, "max_items_per_message": 20,
                },
            )
            p.storage.save(item1)
            p.storage.save(item2)
            p.storage.set_state(["https://ex.com/fail1", "https://ex.com/fail2"], "deferred")

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            # Все items падают в рендеринге
            with patch("newsbot.digest._render_item_raw",
                       side_effect=RuntimeError("render error")):
                count = p.run_digest(force=True)

            # 0 отправлено (early-return)
            self.assertEqual(count, 0, f"Expected 0 (early-return), got {count}")
            # Ничего не отправлено — send_text не вызывался
            delivery.send_text.assert_not_called()

            # Оба items → state='failed' (критичная часть фикса)
            all_failed = {i.url for i in p.storage.get_all_by_state("failed")}
            self.assertIn("https://ex.com/fail1", all_failed,
                "fail1 должен быть state='failed'")
            self.assertIn("https://ex.com/fail2", all_failed,
                "fail2 должен быть state='failed'")

            # Ничего НЕ осталось в deferred
            deferred_urls = {i.url for i in p.storage.get_all_by_state("deferred")}
            self.assertNotIn("https://ex.com/fail1", deferred_urls,
                "fail1 не должен остаться deferred")
            self.assertNotIn("https://ex.com/fail2", deferred_urls,
                "fail2 не должен остаться deferred")

            # Граница сдвинута (маркеры окна обновлены)
            new_boundary = p.storage.get_meta("digest_window_started_at")
            self.assertIsNotNone(new_boundary)
            self.assertGreater(new_boundary, "2020-01-01T00:00:00+00:00")

            p.storage.close()
            p.close()

    def test_build_digest_messages_returns_failed_urls(self):
        """
        build_digest_messages возвращает urls выпавших items в 3-м элементе.
        """
        bad = _mk_item(
            title="Bad", url="https://ex.com/bad",
            llm_score=8, published="2025-01-01 12:00",
        )
        ok = _mk_item(
            title="OK", url="https://ex.com/ok",
            llm_score=7, published="2025-01-01 12:00",
        )

        def fake_render(item):
            if item.url == "https://ex.com/bad":
                raise RuntimeError("boom")
            return _real_render_item_raw(item)

        with patch("newsbot.digest._render_item_raw", side_effect=fake_render):
            groups, _, failed = build_digest_messages(
                [bad, ok], max_items_per_message=20,
            )

        # bad выпал, ok в groups
        self.assertEqual(len(failed), 1)
        self.assertIn("https://ex.com/bad", failed)
        # ok попал в digest
        self.assertEqual(len(groups), 1)
        _, ok_items = groups[0]
        self.assertEqual(len(ok_items), 1)
        self.assertEqual(ok_items[0].url, "https://ex.com/ok")


# =============================================================================
# MAJOR 3-фикс: running_counts инвариант в backfill
# =============================================================================

class TestBackfillRunningCountsInvariant(unittest.TestCase):
    """
    MAJOR: running_counts инкрементится сразу после append, не после цикла.
    Репро: max_items=3, топ-1 France, пул 2 France + 1 Russia, max_per_location=2.
    После backfill France ≤ 2.
    """

    def test_backfill_respects_max_per_location_with_sequential_candidates(self):
        """
        MAJOR (wave 2): exact bug repro — two-loop backfill bug.

        Bug: backfill_candidates формировался ПОЛНЫМ проходом (filter-only,
        running_counts не инкрементился), а инкремент был в ОТДЕЛЬНОМ цикле ПОСЛЕ.
        Все кандидаты одной (loc,day) проходили одновременно, превышая cap.

        Exact scenario from review:
          - max_items=6, max_per_location=2
          - deferred (top_n): France×1 (F9, score=9), Germany×4 (G8..G5, scores 8..5),
            Russia×1 (R4, score=4) — но Germany cap урежет его до Germany×2
          - after cap: deferred=[F9, G8, G7, R4], France=1 < cap=2, Germany=2=cap
          - pool (deferrable_post_published[6:]): France×2 (F7,F6 scores 3,2),
            Germany×4 (G4..G1 scores 1..−2) — Germany capped (skip), France has room
          - broken code (2 loops): loop1 accepts BOTH F7 and F6 (running_counts still 0
            for France), loop2 increments AFTER (no re-check) → France=3 > cap=2
          - fixed code (single loop): F7 increments after add (France=2), F6
            correctly rejected (France already at cap)
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # Deferred items (top_n). France=1, Germany=4, Russia=1.
            # Germany gets capped to 2. After cap: France=1, Germany=2, Russia=1.
            # Free slots: 6 - 4 = 2.
            _insert_deferred_item(
                db_path, title="F9", url="https://ex.com/f9",
                published="2026-01-01 12:00", fetched_at="2026-01-01T11:00:00+00:00",
                score=9, cluster_key="france|h1|2026-01-01",
            )
            for i, (url, score) in enumerate([
                ("https://ex.com/g8", 8), ("https://ex.com/g7", 7),
                ("https://ex.com/g6", 6), ("https://ex.com/g5", 5),
            ]):
                _insert_deferred_item(
                    db_path, title=f"G{8-i}", url=url,
                    published="2026-01-01 12:00", fetched_at="2026-01-01T10:30:00+00:00",
                    score=score, cluster_key="germany|h1|2026-01-01",
                )
            _insert_deferred_item(
                db_path, title="R4", url="https://ex.com/r4",
                published="2026-01-01 12:00", fetched_at="2026-01-01T10:00:00+00:00",
                score=4, cluster_key="russia|h1|2026-01-01",
            )

            # Pool (deferrable_post_published[6:]): 2 France + 4 Germany.
            # Germany already at cap=2 → skipped. France has 1 slot → only 1 passes.
            # F7: France count before=1 (<2), increments to 2 → appends
            # F6: France count before=2 (>=2), rejected
            for title, url, score in [
                ("F7", "https://ex.com/f7", 3),
                ("F6", "https://ex.com/f6", 2),
                ("G4", "https://ex.com/g4", 1),
                ("G3", "https://ex.com/g3", 0),
                ("G2", "https://ex.com/g2", -1),
                ("G1", "https://ex.com/g1", -2),
            ]:
                loc = "france" if title.startswith("F") else "germany"
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2026-01-01 12:00", fetched_at="2026-01-01T09:00:00+00:00",
                    score=score, cluster_key=f"{loc}|h1|2026-01-01",
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 6, "max_items_per_message": 20,
                    "max_per_location": 2,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            # After run_digest: deferred = [F9, G8, G7, R4] + F7 = 5 items
            # France = F9 + F7 = 2 ≤ max_per_location=2
            # Germany = G8 + G7 = 2 ≤ max_per_location=2
            # F6 must NOT be in digest (France already at cap after F7)
            self.assertGreaterEqual(count, 1)
            # Minor 5-фикс: собираем ВСЕ вызовы send_text (вдруг несколько
            # сообщений при low max_items_per_message). call_args_list сохраняет
            # порядок — склеиваем по порядку отправки.
            text = "".join(
                call_args[0][0]
                for call_args in delivery.send_text.call_args_list
            )

            # France-инвариант: строго ≤ 2
            france_count = sum(
                1 for title in ("F9", "F7", "F6")
                if title in text
            )
            self.assertLessEqual(
                france_count, 2,
                f"France count={france_count} (F9+F7+F6 in text) > max_per_location=2. "
                "Bug: both France pool candidates passed simultaneously (two-loop bug)."
            )
            self.assertIn("F9", text)
            self.assertIn("F7", text,
                "F7 должен быть в digest: France=1<cap=2, 1 slot available")
            self.assertNotIn("F6", text,
                "F6 не должен быть в digest: France уже на cap=2 после F7")

            p.storage.close()
            p.close()


# =============================================================================
# Minor 3: docstring про хеширование сырого title
# =============================================================================

class TestBuildClusterKeyRawTitleHash(unittest.TestCase):
    """pipeline.build_cluster_key хеширует СЫРОЙ title (без normalize)."""

    def test_emoji_title_uses_raw_title(self):
        """
        Два items с разными emoji-заголовками должны дать РАЗНЫЕ cluster_key.
        Раньше (normalize → stem fallback): оба дали бы hash от normalize()
        который может совпасть. После фикса: hash от сырого title.
        """
        from newsbot.pipeline import build_cluster_key
        key1 = build_cluster_key(
            "France", None, "2026-01-01", title="🤖 AI News 2024",
        )
        key2 = build_cluster_key(
            "France", None, "2026-01-01", title="🤖 AI News 2025",
        )
        self.assertNotEqual(key1, key2,
            "Разные заголовки должны дать разные cluster_key (сырой title hash)")
        # Оба содержат nostem: — признак per-title fallback
        self.assertIn("nostem:", key1)
        self.assertIn("nostem:", key2)

    def test_same_title_same_cluster_key(self):
        """Идентичный title → идентичный cluster_key."""
        from newsbot.pipeline import build_cluster_key
        key1 = build_cluster_key(
            "France", None, "2026-01-01", title="Same Headline",
        )
        key2 = build_cluster_key(
            "France", None, "2026-01-01", title="Same Headline",
        )
        self.assertEqual(key1, key2)


# =============================================================================
# Задача 2 (pipeline M1): дубли на границе max_items
# =============================================================================

class TestDupOnMaxItemsBoundary(unittest.TestCase):
    """
    Задача 2 (M1): событие X в top-N + его пересказ Y в хвосте deferred-пула
    НЕ должны оба попасть в дайджест. Пересказ Y фильтруется RECENT-контекстом,
    в который теперь добавлены title/source топ-N items.
    """

    def test_top_n_item_squelches_tail_duplicate_in_digest(self):
        """
        Задача 3 (M1): событие EventHead в top-N, пересказ TailDup в pool_tail.
        После внутреннего dedup в top-N освобождается слот, TailDup реально
        доходит до backfill и отсекается _check_against_shown именно потому,
        что title/source top-N items добавлены в RECENT (_digest_topn_shown).

        Схема (5 items, max_items=4):
          - deferrable_post_published = [Alpha(10), Bravo(9), EventHead(8),
            EventDup(7), TailDup(6)]
          - top-N (4): Alpha, Bravo, EventHead, EventDup
          - detect_duplicates отсеивает EventDup (внутренний дубль) →
            deferred=[Alpha, Bravo, EventHead] (3 < max_items)
          - _digest_topn_shown добавляет Alpha/Bravo/EventHead в RECENT
          - pool_tail = [TailDup]; fake_check видит EventHead в RECENT →
            TailDup дубль → _backfill_dup_urls={tail}; слот остаётся свободным
          - late backfill повторно пропускает TailDup (self._backfill_dup_urls)
          - digest: Alpha, Bravo, EventHead (3)

        БЕЗ M1-фикса (_digest_topn_shown): RECENT пуст → TailDup не дубль →
          ранний backfill добавляет TailDup → тест падает (TailDup в digest).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            for title, url, score in [
                ("Alpha",     "https://ex.com/a",  10),
                ("Bravo",     "https://ex.com/b",   9),
                ("EventHead", "https://ex.com/h",   8),
                ("EventDup",  "https://ex.com/d",   7),
                ("TailDup",   "https://ex.com/y",   6),  # pool_tail, дубль EventHead
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                )

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 4, "max_items_per_message": 20,
                    "cluster_duplicates": True,
                },
            )

            # Внутренний dedup top-N: EventDup — дубль EventHead (освобождает слот).
            fake_filter = MagicMock()
            def detect_dup_idx(candidates, recent_shown=None):
                return {i for i, it in enumerate(candidates) if it.title == "EventDup"}
            fake_filter.detect_duplicates.side_effect = detect_dup_idx
            p.llm_filter = fake_filter

            # Y дубль EventHead из top-N — но только если EventHead реально
            # присутствует в RECENT (это и есть проверяемый фикс).
            check_calls = []

            def fake_check(candidates, recent_shown):
                check_calls.append([it.title for it in candidates])
                excluded = set()
                for it in candidates:
                    if it.title == "TailDup":
                        for sh in recent_shown:
                            if sh.get("title") == "EventHead":
                                excluded.add(it.url)
                                break
                return excluded

            p._check_against_shown = fake_check

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            p.run_digest(force=True)

            text = "".join(
                call_args[0][0]
                for call_args in delivery.send_text.call_args_list
            )
            # TailDup реально дошёл до backfill-проверки (не отсечён top-N).
            self.assertIn("TailDup", check_calls[0],
                f"TailDup должен дойти до _check_against_shown, calls={check_calls}")

            self.assertIn("Alpha", text)
            self.assertIn("Bravo", text)
            self.assertIn("EventHead", text, "EventHead (top-N) должен быть в digest")
            self.assertNotIn("EventDup", text, "EventDup отсеян внутренним dedup")
            self.assertNotIn("TailDup", text,
                "TailDup из pool_tail НЕ должен быть в digest (дубль EventHead в top-N)")
            p.storage.close()
            p.close()

    def test_skipped_rewrite_filtered_by_guarantee_recall(self):
        """
        Задача 2б: skipped-пересказ X (добран гарантией кино) не добавляется
        повторно. Skipped-item с категорией cinema, score ниже deferred threshold
        (но >= guarantee_min_score). _check_against_shown видит X в top-N.

        NOTE: для простоты тест использует llm_filter без реального LLM.
        Skipped-items добавляются в пул гарантии через _check_against_shown
        с RECENT-контекстом, включающим top-N.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # Top-N: событие X
            _insert_deferred_item(
                db_path, title="CinematicEvent",
                url="https://ex.com/cinemaevent",
                published="2026-01-01 12:00",
                fetched_at="2026-01-01T12:00:00+00:00",
                score=9,
            )
            # Обычный top-2
            for title, url, score in [
                ("TechA", "https://ex.com/techa", 8),
                ("TechB", "https://ex.com/techb", 7),
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published="2026-01-01 12:00",
                    fetched_at="2026-01-01T12:00:00+00:00",
                    score=score,
                )
            # Skipped cinema-item: пересказ CinematicEvent (score < threshold,
            # но >= guarantee_min_score=3). llm_summary есть.
            from unittest.mock import MagicMock as _MM
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT OR IGNORE INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score, llm_summary, tags)
                VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?)
                """,
                (
                    "Deadline",
                    "https://ex.com/cinemarewrite",
                    "CinematicEvent_rewrite",
                    "Summary",
                    "2026-01-01 12:00",
                    "2026-01-01T12:00:00+00:00",
                    "2026-01-01T12:00:00+00:00",
                    5,
                    "Summary: Cinematic rewrite",
                    '["film","movie"]',
                ),
            )
            conn.commit()
            conn.close()

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    "enabled": True, "max_items": 3, "max_items_per_message": 20,
                    "category_min": {"cinema": 1},
                    "cluster_duplicates": True,
                },
            )

            def fake_detect(candidates, recent_shown):
                # CinematicEvent_rewrite дубль CinematicEvent из top-N
                for sh in recent_shown:
                    if sh.get("title") == "CinematicEvent":
                        return {0}
                return set()
            real_filter = getattr(p, "llm_filter", None) or _MM()
            real_filter.detect_duplicates = _MM(side_effect=fake_detect)
            p.llm_filter = real_filter

            delivery = _MM()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            text = "".join(
                call_args[0][0]
                for call_args in delivery.send_text.call_args_list
            )
            # CinematicEvent в digest (top-N лидер)
            self.assertIn("CinematicEvent", text)
            # CinematicEvent_rewrite НЕ в digest (filtered as dup via RECENT)
            self.assertNotIn("CinematicEvent_rewrite", text,
                "Skipped-дубль не должен попасть через guarantee")

            p.storage.close()
            p.close()


# =============================================================================
# Задача 3 (pipeline M2): поздний backfill не возвращает отсечённые дубли
# =============================================================================

class TestLateBackfillNoReclaimedDuplicates(unittest.TestCase):
    """
    Задача 3 (M2): поздний backfill не возвращает отсечённые дубли.
    self._backfill_dup_urls используется для пропуска в позднем backfill.
    """

    def test_early_backfill_dedup_not_reclaimed_by_late_backfill(self):
        """
        Задача 4a (M2): поздний backfill пропускает url из
        self._backfill_dup_urls, отсечённый ранним backfill.

        Схема (5 items, max_items=4):
          - deferrable_post_published = [Alpha(10), Bravo(9), Charlie(8),
            EventDup(7), TailDup(6)]
          - top-N (4): Alpha, Bravo, Charlie, EventDup
          - detect_duplicates отсеивает EventDup → deferred=3 (< max_items)
          - ранний backfill: pool_tail=[TailDup]; fake_check (Charlie в RECENT)
            → TailDup в _backfill_dup_urls, слот остаётся свободным
          - поздний backfill: TailDup пропускается именно по
            self._backfill_dup_urls → digest=[Alpha, Bravo, Charlie] (3)

        БЕЗ M2-фикса (поздний backfill без проверки _backfill_dup_urls):
          TailDup возвращается → digest=[Alpha, Bravo, Charlie, TailDup] →
          тест падает (TailDup в digest, count=4 вместо 3).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta('digest_window_started_at', '2020-01-01T00:00:00+00:00')
            s.close()

            for title, url, score in [
                ('Alpha',    'https://ex.com/a', 10),
                ('Bravo',    'https://ex.com/b',  9),
                ('Charlie',  'https://ex.com/c',  8),
                ('EventDup', 'https://ex.com/d',  7),
                ('TailDup',  'https://ex.com/y',  6),  # pool_tail, дубль Charlie
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published='2026-01-01 12:00',
                    fetched_at='2026-01-01T12:00:00+00:00',
                    score=score)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    'enabled': True, 'max_items': 4,
                    'max_items_per_message': 20,
                    'max_per_location': 10,
                    'cluster_duplicates': True,
                },
            )

            # Внутренний dedup top-N освобождает слот (EventDup — дубль Charlie).
            fake_filter = MagicMock()
            def detect_dup_idx(candidates, recent_shown=None):
                return {i for i, it in enumerate(candidates) if it.title == 'EventDup'}
            fake_filter.detect_duplicates.side_effect = detect_dup_idx
            p.llm_filter = fake_filter

            check_calls = []

            def fake_check(candidates, recent_shown):
                check_calls.append([it.title for it in candidates])
                excluded = set()
                for it in candidates:
                    if it.title == 'TailDup':
                        for sh in recent_shown:
                            if sh.get('title') == 'Charlie':
                                excluded.add(it.url)
                                break
                return excluded

            p._check_against_shown = fake_check

            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            text = ''.join(
                call_args[0][0]
                for call_args in delivery.send_text.call_args_list
            )

            # TailDup реально дошёл до раннего backfill-дедупа.
            self.assertIn('TailDup', check_calls[0],
                f"TailDup должен дойти до _check_against_shown, calls={check_calls}")

            for t in ('Alpha', 'Bravo', 'Charlie'):
                self.assertIn(t, text, f"{t} должен быть в digest")
            self.assertNotIn('EventDup', text, 'EventDup отсеян внутренним dedup')
            self.assertNotIn('TailDup', text,
                'TailDup (в _backfill_dup_urls) НЕ должен попасть в digest')
            self.assertEqual(count, 3,
                f'Ожидалось 3 items (слот не должен заполняться дублем), got {count}')

            p.storage.close()
            p.close()

    def test_late_backfill_works_when_early_backfill_skipped(self):
        """
        Задача 4b (edge case): ранний backfill не добавил ничего
        (deferred == max_items до cap), пул НЕ пуст; после location-cap
        освобождается слот — поздний backfill его добирает.

        Схема: max_items=5, top-5 = Alpha(10), Bravo(9), Charlie(8), Delta(7),
        Echo(6); Charlie/Delta/Echo — одна локация France, max_per_location=2.
        Ранний backfill: len==5==max_items → пропущен (пул [Foxtrot] не тронут).
        Cap: France оставляет Charlie,Delta → Echo отсечён, len=4.
        Поздний backfill: Foxtrot (Germany, свободная локация) добавляется →
        digest=[Alpha,Bravo,Charlie,Delta,Foxtrot] (5), Echo нет.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta('digest_window_started_at', '2020-01-01T00:00:00+00:00')
            s.close()

            france_key = 'France|stem|2026-01-01'
            for title, url, score, cluster_key in [
                ('Alpha',   'https://ex.com/a', 10, None),
                ('Bravo',   'https://ex.com/b',  9, None),
                ('Charlie', 'https://ex.com/c',  8, france_key),
                ('Delta',   'https://ex.com/d',  7, france_key),
                ('Echo',    'https://ex.com/e',  6, france_key),  # отсечётся cap'ом
                ('Foxtrot', 'https://ex.com/f',  5, 'Germany|stem|2026-01-01'),
            ]:
                _insert_deferred_item(
                    db_path, title=title, url=url,
                    published='2026-01-01 12:00',
                    fetched_at='2026-01-01T12:00:00+00:00',
                    score=score, cluster_key=cluster_key)

            p = _mk_pipeline(
                db_path,
                digest_cfg={
                    'enabled': True, 'max_items': 5, 'max_items_per_message': 20,
                    'max_per_location': 2,
                    'cluster_duplicates': False,
                },
            )
            delivery = MagicMock()
            delivery.send_text.return_value = True
            p.delivery = delivery

            count = p.run_digest(force=True)

            text = ''.join(
                call_args[0][0]
                for call_args in delivery.send_text.call_args_list
            )
            for t in ('Alpha', 'Bravo', 'Charlie', 'Delta'):
                self.assertIn(t, text, f"{t} должен быть в digest")
            self.assertNotIn('Echo', text,
                'Echo отсечён location-cap (France=2) и не вернулся backfill-ом')
            # Foxtrot доступен только через поздний backfill (ниже top-5,
            # пул не тронут ранним backfill'ом — deferred был полон).
            self.assertIn('Foxtrot', text,
                'Foxtrot (добавлен поздним backfill) должен попасть в digest')
            self.assertEqual(count, 5, f'Expected 5 items, got {count}')

            p.storage.close()
            p.close()

