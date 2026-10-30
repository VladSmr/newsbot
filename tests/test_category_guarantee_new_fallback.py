"""
Fallback из нескоренных (state='new') для _category_guarantee.

Задача 2: если категория после добора из deferred + skipped всё ещё ниже
category_min — недостающие items берутся из state='new' того же окна
(fetched_at DESC, ровно столько, сколько не хватает).

Шесть тестов:
  1. score=2/3 item теперь попадает в гарантию (порог 2).
  2. llm_reason='prefilter' (score=1) — НЕ попадает никогда.
  3. Категория без скоренных → лучший new по fetched_at DESC → digested, llm_score=NULL.
  4. Скоренные приоритетнее new: 1 скоренный (score=3) + 3 new, category_min=2 →
     скоренный + 1 свежайший new.
  5. Fallback не берёт больше, чем не хватает (category_min=1, есть скоренный →
     new не трогается).
  6. guarantee_min_score из конфига читается и применяется (score=2 в пуле).
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

from newsbot.pipeline import NewsPipeline
from newsbot.storage import Storage


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


def _insert_deferred_item(db_path, source, url, title, published, fetched_at,
                          score=6, tags=None):
    """Вставляет deferred-item напрямую в БД."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at,
         llm_score, tags)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?, ?)
        """,
        (source, url, title, "Test summary", published, fetched_at,
         fetched_at, score, tags),
    )
    conn.commit()
    conn.close()


def _insert_skipped_item(db_path, source, url, title, published, fetched_at,
                         score=5, tags=None, llm_reason=None):
    """Вставляет skipped-item напрямую в БД."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at,
         llm_score, llm_summary, llm_reason, tags)
        VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?, ?)
        """,
        (
            source, url, title,
            "Summary for " + title,
            published, fetched_at, fetched_at,
            score,
            f"Summary: {title}",
            llm_reason,
            tags,
        ),
    )
    conn.commit()
    conn.close()


def _insert_new_item(db_path, source, url, title, published, fetched_at,
                     tags=None, cluster_key=None):
    """Вставляет new-item напрямую в БД (нескоренный)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, tags, cluster_key)
        VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?, ?)
        """,
        (source, url, title, "Summary", published, fetched_at,
         fetched_at, tags, cluster_key),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, digest_cfg):
    """Создаёт Pipeline с временной БД и мок-доставкой."""
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": 15,
        "digest": digest_cfg,
        "llm": {"enabled": False},
    }
    p = NewsPipeline(cfg)
    p.storage.close()
    p.storage = Storage(db_path)
    delivery = MagicMock()
    delivery.send_text.return_value = True
    p.delivery = delivery
    return p


class TestGuaranteeMinScore2(unittest.TestCase):
    """Тест 1: score=2/3 item теперь попадает в гарантию (порог=2)."""

    def test_skipped_score_2_pulled_into_guarantee(self):
        """
        Раньше guarantee_min_score=4 → skipped score=2 отсекался.
        Теперь порог=2 → score=2 попадает в пул гарантии и закрывает минимум.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 1 deferred 'other' (не cinema) с высоким score — займёт слоты
            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=9,
            )
            # skipped cinema, score=2 — ниже старого порога 4, но >= нового 2
            _insert_skipped_item(
                db_path, "Deadline", "https://ex.com/cinema2", "Cinema2",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=2, tags='["film"]',
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
                "guarantee_min_score": 2,
            })

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/cinema2", digested,
                "skipped cinema score=2 должен попасть в гарантию при пороге 2",
            )
            p.storage.close()
            p.close()


class TestPrefilterNeverPulled(unittest.TestCase):
    """Тест 2: llm_reason='prefilter' (score=1) — НЕ попадает никогда."""

    def test_prefilter_item_not_in_guarantee(self):
        """
        prefilter items: score=1, llm_reason='prefilter'.
        Даже при guarantee_min_score=2 они должны быть исключены.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=9,
            )
            # prefilter item: score=1, llm_reason='prefilter'
            _insert_skipped_item(
                db_path, "SomeBlog", "https://ex.com/prefilter", "Prefiltered",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=1, tags='["film"]', llm_reason="prefilter",
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
                "guarantee_min_score": 2,
            })

            count = p.run_digest(force=True)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertNotIn(
                "https://ex.com/prefilter", digested,
                "prefilter item (score=1, reason=prefilter) НЕ должен попадать в гарантию",
            )
            p.storage.close()
            p.close()


class TestNewFallbackGuaranteesCategory(unittest.TestCase):
    """Тест 3: категория без скоренных → лучший new по fetched_at DESC → digested, llm_score=NULL."""

    def test_category_without_scored_items_uses_new_fallback(self):
        """
        Категория cinema не представлена ни в deferred, ни в skipped.
        Fallback берёт лучший (fetched_at DESC) new-item той же категории.
        Он попадает в дайджест, помечается digested, llm_score=NULL не падает.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 1 deferred 'other' — чтобы был непустой дайджест
            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=8,
            )
            # new-item cinema (не скорился) из того же окна
            _insert_new_item(
                db_path, "Deadline", "https://ex.com/new-cinema", "NewCinema",
                "2026-01-01 14:00", "2026-01-01T14:00:00+00:00",
                tags='["film"]', cluster_key="другое|a|2026-01-01",
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
            })

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/new-cinema", digested,
                "new-item cinema должен попасть в гарантию при отсутствии скоренных",
            )

            # llm_score=NULL: проверяем, что item существует и не упал
            all_items = list(p.storage.get_all_by_state("digested"))
            cinema_item = next(
                (i for i in all_items if i.url == "https://ex.com/new-cinema"), None
            )
            self.assertIsNotNone(cinema_item)
            # llm_score должен быть None/null (не падает)
            self.assertIsNone(
                cinema_item.llm_score,
                "new-item имеет llm_score=None — не падает на None",
            )
            p.storage.close()
            p.close()


class TestScoredPriorityOverNew(unittest.TestCase):
    """Тест 4: скоренные приоритетнее new."""

    def test_scored_wins_over_new_for_min_count(self):
        """
        category_min=2 для cinema. В пуле: 1 скоренный (score=3) + 3 new.
        Ожидание: скоренный + 1 свежайший new (нужен ровно 1 new, не все 3).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=9,
            )
            # 1 skipped cinema, score=3 (скоренный)
            _insert_skipped_item(
                db_path, "Deadline", "https://ex.com/cin-scored", "CinScored",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=3, tags='["film"]',
            )
            # 3 new cinema-items: разное fetched_at (DESC = свежие первыми)
            # Самая свежая должна быть выбрана последней
            _insert_new_item(
                db_path, "IGN", "https://ex.com/new-cin-old", "OldNew",
                "2026-01-01 10:00", "2026-01-01T10:00:00+00:00",
                tags='["film"]',
            )
            _insert_new_item(
                db_path, "IGN", "https://ex.com/new-cin-mid", "MidNew",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                tags='["film"]',
            )
            _insert_new_item(
                db_path, "IGN", "https://ex.com/new-cin-fresh", "FreshNew",
                "2026-01-01 14:00", "2026-01-01T14:00:00+00:00",
                tags='["film"]',
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 2},
                "guarantee_min_score": 2,
            })

            count = p.run_digest(force=True)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            # Скоренный обязательно
            self.assertIn(
                "https://ex.com/cin-scored", digested,
                "скоренный cinema (score=3) должен быть в дайджесте",
            )
            # Ровно один new — самый свежий
            new_in_digest = [
                u for u in digested
                if u.startswith("https://ex.com/new-cin-")
            ]
            self.assertEqual(
                len(new_in_digest), 1,
                f"Должен быть ровно 1 new-item, а не {len(new_in_digest)}: {new_in_digest}",
            )
            self.assertIn("https://ex.com/new-cin-fresh", new_in_digest,
                          "Самый свежий new-item должен быть выбран")
            p.storage.close()
            p.close()


class TestFallbackTakesExactDeficit(unittest.TestCase):
    """Тест 5: fallback не берёт больше, чем не хватает."""

    def test_fallback_takes_only_what_is_missing(self):
        """
        category_min=1. Есть скоренный cinema (score=5) → new не нужен.
        Fallback не должен добавлять никакие new-items.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=9,
            )
            # skipped cinema, score=5 — закрывает category_min=1 полностью
            _insert_skipped_item(
                db_path, "Deadline", "https://ex.com/cin-scored", "CinScored",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=5, tags='["film"]',
            )
            # new-item cinema — НЕ должен попасть (скоренный уже закрыл минимум)
            _insert_new_item(
                db_path, "IGN", "https://ex.com/new-cin", "NewCinema",
                "2026-01-01 14:00", "2026-01-01T14:00:00+00:00",
                tags='["film"]',
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
                "guarantee_min_score": 2,
            })

            count = p.run_digest(force=True)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            # Скоренный — да
            self.assertIn(
                "https://ex.com/cin-scored", digested,
                "скоренный cinema должен быть в дайджесте",
            )
            # new-item cinema — нет (не нужен)
            self.assertNotIn(
                "https://ex.com/new-cin", digested,
                "new-item cinema НЕ должен попадать, если скоренный уже закрыл минимум",
            )
            p.storage.close()
            p.close()


class TestGuaranteeMinScoreFromConfig(unittest.TestCase):
    """Тест 6: guarantee_min_score из конфига читается и применяется."""

    def test_score_2_in_pool_selected_score_1_excluded(self):
        """
        guarantee_min_score=2 в конфиге.
        В пуле skipped: score=2 (должен пройти) и score=1 с reason!=prefilter
        (гипотетический путь — должен быть отсечён).
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            _insert_deferred_item(
                db_path, "tass", "https://ex.com/other1", "Other1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=9,
            )
            # skipped, score=2 — проходит порог 2
            _insert_skipped_item(
                db_path, "Deadline", "https://ex.com/cin2", "CinemaScore2",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=2, tags='["film"]',
            )
            # skipped, score=1 с другим reason — отсекается порогом 2
            _insert_skipped_item(
                db_path, "Deadline", "https://ex.com/cin1", "CinemaScore1",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=1, tags='["film"]', llm_reason="some_other_reason",
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 10,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
                "guarantee_min_score": 2,   # из конфига
            })

            count = p.run_digest(force=True)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/cin2", digested,
                "score=2 >= guarantee_min_score(2) → в гарантии",
            )
            self.assertNotIn(
                "https://ex.com/cin1", digested,
                "score=1 < guarantee_min_score(2) → отсечён",
            )
            p.storage.close()
            p.close()


class TestFullDigestWithNewFallbackSurvivesSlice(unittest.TestCase):
    """Тест 4: дайджест заполнен до max_items, категория X без скоренных →
    new-fallback item ПРИСУТСТВУЕТ в финальном списке.

    БЕЗ фикса 1 (резервирование слотов) new-item (llm_score=NULL → 0) попадает
    в хвост после сортировки score DESC и вырезается [:max_items] →
    тест ПАДАЕТ. С фиксом — проходит.
    """

    def test_new_fallback_survives_full_digest_slice(self):
        """
        max_items=5. Заполняем дайджест 5 deferred-items (score 9..5).
        cinema без скоренных, но new-item cinema из того же окна.
        category_min=1, гарантия добавляет new-item (swapped-out не-кино).
        Финальный [:5] → guarantee-reserved слот гарантирует cinema в выдаче.
        """
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 5 deferred items — заполняют дайджест до max_items=5.
            for i in range(1, 6):
                _insert_deferred_item(
                    db_path, "tass", f"https://ex.com/other{i}", f"Other{i}",
                    "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=10 - i,
                )
            # new-item cinema — в том же окне, fetched_at позже всех deferred.
            _insert_new_item(
                db_path, "Deadline", "https://ex.com/new-cinema",
                "NewCinema",
                "2026-01-01 14:00", "2026-01-01T14:00:00+00:00",
                tags='["film"]', cluster_key="другое|a|2026-01-01",
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 5,
                "max_items_per_message": 20,
                "category_min": {"cinema": 1},
            })

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/new-cinema", digested,
                "new-fallback cinema должен остаться в дайджесте "
                "после [:max_items]slice — гарантированный слот",
            )
            p.storage.close()
            p.close()


class TestBothGuaranteedAndOriginalSurviveSlice(unittest.TestCase):
    """Тест 5: исходный deferred + new-fallback, оба защищены финальным слайсом.

    cinema min=2. Исходный deferred cinema (score=3, ниже других) — сам закрывает
    1 из 2 минимума. new-fallback cinema — второй. При len > max_items финальный
    sort+slice может вырезать исходный (score 3) → категория падает до 1.

    БЕЗ фикса (guarantee_added_urls) — тест ПАДАЕТ: исходный score-3 вырезается.
    С фиксом (guarantee_protected_urls) — ОБА cinema-items в финальном списке.
    """

    def test_original_deferred_plus_new_fallback_both_survive(self):
        with _tmp_db() as db_path:
            s = Storage(db_path)
            s.set_meta("digest_window_started_at", "2020-01-01T00:00:00+00:00")
            s.close()

            # 5 deferred: 4 other (score 9..6) — заполняют дайджест.
            # 1 cinema (score=3) — исходный deferred, сам закрывает 1 из 2 минимума.
            # Он ниже всех other-items, уязвим для [:max_items]-среза.
            for i in range(1, 5):
                _insert_deferred_item(
                    db_path, "tass", f"https://ex.com/other{i}", f"Other{i}",
                    "2026-01-01 12:00", "2026-01-01T12:00:00+00:00", score=10 - i,
                )
            _insert_deferred_item(
                db_path, "deadline", "https://ex.com/cinema-deferred",
                "CinemaDeferred",
                "2026-01-01 12:00", "2026-01-01T12:00:00+00:00",
                score=3, tags='["film"]',
            )
            # new-item cinema — закрывает вторую позицию минимума.
            _insert_new_item(
                db_path, "Deadline", "https://ex.com/new-cinema",
                "NewCinema",
                "2026-01-01 14:00", "2026-01-01T14:00:00+00:00",
                tags='["film"]', cluster_key="другое|a|2026-01-01",
            )

            p = _mk_pipeline(db_path, {
                "enabled": True, "max_items": 5,
                "max_items_per_message": 20,
                "category_min": {"cinema": 2},
            })

            count = p.run_digest(force=True)
            self.assertGreaterEqual(count, 1)

            digested = {i.url for i in p.storage.get_all_by_state("digested")}
            self.assertIn(
                "https://ex.com/cinema-deferred", digested,
                "исходный deferred cinema (score=3) должен быть защищён слайсом — "
                "он сам закрывает 1 из category_min=2",
            )
            self.assertIn(
                "https://ex.com/new-cinema", digested,
                "new-fallback cinema должен быть защищён",
            )
            # Проверяем, что в дайджесте ровно 2 cinema-items.
            # Может быть больше если cap позволил — считаем电影院--items в digested.
            cinema_in_digest = [
                i for i in p.storage.get_all_by_state("digested")
                if i.url in {"https://ex.com/cinema-deferred",
                             "https://ex.com/new-cinema"}
            ]
            self.assertEqual(
                len(cinema_in_digest), 2,
                f"Должно быть 2 cinema-items, получили {len(cinema_in_digest)}",
            )
            p.storage.close()
            p.close()


if __name__ == "__main__":
    unittest.main()