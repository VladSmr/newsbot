"""
Тесты оконных операций хранилища (Storage Window).

Покрывает:
  1. get_deferred_items(since_iso): фильтрация по границе окна.
  2. delete_items_before: удаляет items всех состояний старше cutoff,
     возвращает корректный rowcount; второй вызов — 0.

Запуск: python -m pytest tests/test_storage_window.py -v
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.storage import Storage


class TestStorageWindow(unittest.TestCase):
    """Тесты оконной логики: since_iso и delete_items_before."""

    def setUp(self):
        self.db_dir = tempfile.mkdtemp(prefix="nb_test_win_")
        self.db_path = str(Path(self.db_dir) / "test_win.db")
        self.storage = Storage(self.db_path)

    def tearDown(self):
        self.storage.close()
        try:
            os.remove(self.db_path)
        except (OSError, PermissionError):
            pass  # Windows: файл может быть занят
        try:
            os.rmdir(self.db_dir)
        except OSError:
            pass

    def _insert(self, state, hours_ago, score=5.0, source="Test",
                title="Test", url=None):
        """Вставляет item напрямую через SQL (для управления fetched_at и state)."""
        if url is None:
            url = (f"https://example.com/win_"
                   f"{hash((title or '') + state + str(hours_ago))}")
        fetched = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score, llm_summary, llm_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, url, title, "sum", fetched, fetched,
              state, fetched, score, "summary", "reason"))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────
    # get_deferred_items(since_iso=...)
    # ─────────────────────────────────────────────────────────────────

    def test_get_deferred_items_since_iso_25h_skipped(self):
        """Item с fetched_at 25 часов назад НЕ попадает в выборку (since_iso)."""
        # 25 ч назад — за пределами окна
        self._insert("deferred", hours_ago=25, score=7.0, title="Old")
        # 1 ч назад — внутри окна
        self._insert("deferred", hours_ago=1, score=6.0, title="Fresh")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        items = self.storage.get_deferred_items(since_iso=cutoff)
        self.assertEqual(len(items), 1,
            "25-часовой item должен быть за границей окна, 1-часовой — внутри")
        self.assertEqual(items[0].title, "Fresh")

    def test_get_deferred_items_since_iso_1h_returned(self):
        """Item 1 час назад возвращается при since_iso=24h назад."""
        self._insert("deferred", hours_ago=1, score=6.0, title="Fresh")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        items = self.storage.get_deferred_items(since_iso=cutoff)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Fresh")

    def test_get_deferred_items_no_since_iso_backward_compat(self):
        """Без since_iso возвращаются ВСЕ deferred (обратная совместимость)."""
        self._insert("deferred", hours_ago=25, score=7.0, title="Old")
        self._insert("deferred", hours_ago=1, score=6.0, title="Fresh")
        items = self.storage.get_deferred_items()
        self.assertEqual(len(items), 2,
            "Без since_iso возвращаются оба deferred items")

    def test_get_deferred_items_since_iso_only_deferred(self):
        """since_iso фильтрует ТОЛЬКО deferred; other states игнорируются."""
        self._insert("deferred", hours_ago=1, score=7.0, title="Def")
        self._insert("delivered", hours_ago=1, score=8.0, title="Del")
        self._insert("new", hours_ago=1, score=9.0, title="New")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        items = self.storage.get_deferred_items(since_iso=cutoff)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Def")

    def test_get_deferred_items_since_iso_empty_result(self):
        """Все items старше cutoff → пустой результат."""
        self._insert("deferred", hours_ago=100, score=7.0, title="TooOld")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        items = self.storage.get_deferred_items(since_iso=cutoff)
        self.assertEqual(len(items), 0)

    # ─────────────────────────────────────────────────────────────────
    # delete_items_before
    # ─────────────────────────────────────────────────────────────────

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def test_delete_items_before_removes_only_terminal_states(self):
        """Удаляются ТОЛЬКО терминальные states (delivered/skipped/deferred/
        digested/failed); 'new' и 'pending' остаются даже если старше cutoff."""
        self._insert("deferred", hours_ago=48, title="DefOld")
        self._insert("delivered", hours_ago=48, title="DelOld")
        self._insert("skipped", hours_ago=48, title="SkipOld")
        self._insert("digested", hours_ago=48, title="DigOld")
        self._insert("failed", hours_ago=48, title="FailOld")
        self._insert("new", hours_ago=48, title="NewOld")
        self._insert("pending", hours_ago=48, title="PendOld")
        # Свежие — по 1 часу
        self._insert("deferred", hours_ago=1, title="DefFresh")
        self._insert("delivered", hours_ago=1, title="DelFresh")
        self._insert("new", hours_ago=1, title="NewFresh")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        deleted = self.storage.delete_items_before(cutoff)

        self.assertEqual(deleted, 5,
            "Должно быть удалено 5 терминальных items старше cutoff")

        # Проверяем: новые и pending (независимо от возраста) — остались
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        remaining = conn.execute(
            "SELECT title FROM items"
        ).fetchall()
        conn.close()
        remaining_titles = {r["title"] for r in remaining}
        self.assertIn("NewOld", remaining_titles,
            "new-item старше cutoff НЕ должен быть удалён")
        self.assertIn("PendOld", remaining_titles,
            "pending-item старше cutoff НЕ должен быть удалён")
        self.assertIn("NewFresh", remaining_titles)
        self.assertNotIn("DefOld", remaining_titles,
            "deferred старше cutoff должен быть удалён")
        self.assertNotIn("DelOld", remaining_titles,
            "delivered старше cutoff должен быть удалён")

    def test_delete_items_before_returns_correct_rowcount(self):
        """delete_items_before возвращает точное количество удалённых строк."""
        for i in range(5):
            self._insert("delivered", hours_ago=48, title=f"Old{i}", url=f"https://ex.com/old{i}")
        for i in range(3):
            self._insert("delivered", hours_ago=1, title=f"Fresh{i}", url=f"https://ex.com/fresh{i}")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        deleted = self.storage.delete_items_before(cutoff)
        self.assertEqual(deleted, 5, "Должно быть удалено ровно 5 старых items")

    def test_delete_items_before_idempotent_second_call_zero(self):
        """Повторный вызов delete_items_before удаляет 0 (items уже удалены)."""
        self._insert("deferred", hours_ago=48, title="Old")
        self._insert("delivered", hours_ago=48, title="Old2")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        first = self.storage.delete_items_before(cutoff)
        self.assertEqual(first, 2)

        second = self.storage.delete_items_before(cutoff)
        self.assertEqual(second, 0, "Повторный вызов должен удалить 0 items")

    def test_delete_items_before_preserves_feedback_targets(self):
        """delete_items_before НЕ удаляет feedback_targets
        (они живут по своему retention 180 дней и нужны для 👍/👎)."""
        # Вставляем items с явными URL
        self._insert("deferred", hours_ago=48, title="Old",
                     url="https://ex.com/fb_old")
        self._insert("deferred", hours_ago=1, title="Fresh",
                     url="https://ex.com/fb_fresh")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        conn = sqlite3.connect(self.db_path)

        # feedback_targets с теми же URL'ами что у items
        conn.execute("""
            INSERT INTO feedback_targets (url, source, title, kind, created_at)
            VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
        """, (
            "https://ex.com/fb_old", "Test", "Old", "push",
            (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
            "https://ex.com/fb_fresh", "Test", "Fresh", "push",
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        ))
        conn.commit()
        conn.close()

        self.storage.delete_items_before(cutoff)

        # Проверяем: оба feedback_targets сохранены (не зависят от delete_items_before)
        conn2 = sqlite3.connect(self.db_path)
        conn2.row_factory = sqlite3.Row
        remaining = conn2.execute(
            "SELECT title FROM feedback_targets ORDER BY title"
        ).fetchall()
        conn2.close()
        self.assertEqual(len(remaining), 2,
            "Все feedback_targets сохраняются, даже если item был удалён")
        self.assertEqual({r["title"] for r in remaining}, {"Fresh", "Old"})

    def test_delete_items_before_empty_db(self):
        """delete_items_before на пустой БД возвращает 0."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        deleted = self.storage.delete_items_before(cutoff)
        self.assertEqual(deleted, 0)

    # ─────────────────────────────────────────────────────────────────
    # get_oldest_unprocessed_fetched_at
    # ─────────────────────────────────────────────────────────────────

    def test_get_oldest_unprocessed_fetched_at_returns_min_for_new_pending(self):
        """Возвращает MIN(fetched_at) для items в состояниях new/pending."""
        t_old = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        t_mid = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        t_fresh = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score, llm_summary, llm_reason)
            VALUES
                (?, ?, ?, ?, ?, ?, 'new',     ?, ?, ?, ?),
                (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?),
                (?, ?, ?, ?, ?, ?, 'new',     ?, ?, ?, ?)
        """, (
            "T", "https://ex.com/old",   "Old",   "s", t_old,   t_old,   t_old,   5.0, "sum", "r",
            "T", "https://ex.com/mid",   "Mid",   "s", t_mid,   t_mid,   t_mid,   5.0, "sum", "r",
            "T", "https://ex.com/fresh", "Fresh", "s", t_fresh, t_fresh, t_fresh, 5.0, "sum", "r",
        ))
        conn.commit()
        conn.close()

        result = self.storage.get_oldest_unprocessed_fetched_at()
        self.assertEqual(result, t_old,
            "Должен вернуться самый старый fetched_at среди new/pending")

    def test_get_oldest_unprocessed_fetched_at_skips_terminal_states(self):
        """Терминальные states (delivered/skipped/...) НЕ учитываются."""
        t_delivered = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        t_pending  = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score, llm_summary, llm_reason)
            VALUES
                (?, ?, ?, ?, ?, ?, 'delivered', ?, ?, ?, ?),
                (?, ?, ?, ?, ?, ?, 'pending',  ?, ?, ?, ?)
        """, (
            "T", "https://ex.com/d", "Del", "s", t_delivered, t_delivered, t_delivered, 5.0, "sum", "r",
            "T", "https://ex.com/p", "Pen", "s", t_pending,   t_pending,   t_pending,   5.0, "sum", "r",
        ))
        conn.commit()
        conn.close()

        result = self.storage.get_oldest_unprocessed_fetched_at()
        self.assertEqual(result, t_pending,
            "Только pending учитывается; delivered игнорируется")

    def test_get_oldest_unprocessed_fetched_at_returns_none_when_empty(self):
        """Возвращает None, если нет ни одного new/pending item."""
        self._insert("delivered", hours_ago=1, title="OnlyTerminal")
        result = self.storage.get_oldest_unprocessed_fetched_at()
        self.assertIsNone(result,
            "None ожидается, когда new/pending items отсутствуют")

    # ─────────────────────────────────────────────────────────────────
    # purge_items: separate 'new' retention window
    # ─────────────────────────────────────────────────────────────────

    def test_purge_keeps_new_items_within_new_retention(self):
        """'new' item 2 дня назад при retention_days=0 НЕ удаляется
        (new живёт минимум DEFAULT_NEW_RETENTION_DAYS=7 дней)."""
        self._insert("new", hours_ago=48, title="New2days")  # 2 дня назад
        deleted = self.storage.purge_items(retention_days=0)
        self.assertEqual(deleted, 0, "new-item 2 дня назад не должен быть удалён")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        remaining = {r["title"] for r in conn.execute("SELECT title FROM items").fetchall()}
        conn.close()
        self.assertIn("New2days", remaining)

    def test_purge_removes_old_new_items(self):
        """'new' item 8 дней назад удаляется (старше DEFAULT_NEW_RETENTION_DAYS=7)."""
        self._insert("new", hours_ago=8 * 24, title="New8days")
        deleted = self.storage.purge_items(retention_days=0)
        self.assertEqual(deleted, 1, "new-item старше 7 дней должен быть удалён")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        remaining = {r["title"] for r in conn.execute("SELECT title FROM items").fetchall()}
        conn.close()
        self.assertNotIn("New8days", remaining)

    def test_purge_removes_terminal_items_at_retention_1(self):
        """'delivered' item 2 дня назад при retention_days=1 удаляется."""
        self._insert("delivered", hours_ago=48, title="Del2days")
        deleted = self.storage.purge_items(retention_days=1)
        self.assertEqual(deleted, 1, "delivered старше retention_days должен быть удалён")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        remaining = {r["title"] for r in conn.execute("SELECT title FROM items").fetchall()}
        conn.close()
        self.assertNotIn("Del2days", remaining)


if __name__ == "__main__":
    unittest.main()