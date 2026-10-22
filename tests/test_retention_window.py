"""
Тесты окна хранения и выборки по возрасту (Retention Window).
Запускать: python -m pytest tests/test_retention_window.py -v
"""
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Проектный корень → newsbot
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.storage import Storage
from newsbot import config as config_module


class TestRetentionWindow(unittest.TestCase):
    """Тесты: срок хранения — сутки, выборка дайджеста — 24 часа."""

    def setUp(self):
        self.db_dir = tempfile.mkdtemp(prefix="nb_test_ret_")
        self.db_path = str(Path(self.db_dir) / "test_ret.db")
        self.storage = Storage(self.db_path)

    def tearDown(self):
        self.storage.close()
        # Очистка
        import os
        try:
            os.remove(self.db_path)
            os.rmdir(self.db_dir)
        except (OSError, PermissionError):
            pass  # Windows: файл может быть занят

    def _insert(self, state, hours_ago, score=5.0, urgent=False,
                source="Test", title="Test", url=None,
                llm_summary="s", llm_reason="r"):
        """Вставляет item напрямую через SQL (для управления fetched_at и state)."""
        if url is None:
            url = f"https://example.com/ret_{hash((title or '') + state + str(hours_ago))}"
        fetched = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO items
                (source, url, title, summary, published, fetched_at,
                 state, state_at, llm_score, llm_summary, llm_reason, llm_urgent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, url, title, "sum", fetched, fetched,
              state, fetched, score, llm_summary, llm_reason, urgent))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────
    # get_deferred_items
    # ─────────────────────────────────────────────────────────────────

    def test_get_deferred_items_max_age_25h_skipped(self):
        """Item с fetched_at 25 часов назад НЕ попадает в выборку (max_age=24)."""
        self._insert("deferred", hours_ago=25, score=7.0)
        self._insert("deferred", hours_ago=1, score=6.0)
        items = self.storage.get_deferred_items(max_age_hours=24)
        # 25-часовой НЕ должен быть, 1-часовой — да
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].llm_score, 6.0)

    def test_get_deferred_items_max_age_recent_returned(self):
        """Item 1 час назад возвращается при max_age=24."""
        self._insert("deferred", hours_ago=1, score=6.0)
        items = self.storage.get_deferred_items(max_age_hours=24)
        self.assertEqual(len(items), 1)

    def test_get_deferred_items_max_age_order_preserved(self):
        """Order by llm_score DESC сохраняется при max_age_hours."""
        self._insert("deferred", hours_ago=1, score=5.0, title="Low")
        self._insert("deferred", hours_ago=2, score=9.0, title="High")
        self._insert("deferred", hours_ago=3, score=7.0, title="Mid")
        items = self.storage.get_deferred_items(max_age_hours=24)
        titles = [it.title for it in items]
        self.assertEqual(titles, ["High", "Mid", "Low"])

    def test_get_deferred_items_no_param_returns_all(self):
        """Без параметра max_age_hours возвращаются ВСЕ deferred (обратная совместимость)."""
        self._insert("deferred", hours_ago=25, score=7.0)
        self._insert("deferred", hours_ago=1, score=6.0)
        items = self.storage.get_deferred_items()
        self.assertEqual(len(items), 2)

    def test_get_deferred_items_max_age_none_returns_all(self):
        """max_age_hours=None также возвращает все deferred (явный None)."""
        self._insert("deferred", hours_ago=25, score=7.0)
        self._insert("deferred", hours_ago=1, score=6.0)
        items = self.storage.get_deferred_items(max_age_hours=None)
        self.assertEqual(len(items), 2)

    # ─────────────────────────────────────────────────────────────────
    # purge_items
    # ─────────────────────────────────────────────────────────────────

    def test_purge_deferred_25h_removed(self):
        """25-часовой deferred удаляется при deferred_retention_days=1."""
        self._insert("deferred", hours_ago=25, score=6.0)
        self.storage.purge_items(retention_days=1, deferred_retention_days=1)
        items = self.storage.get_deferred_items()
        self.assertEqual(len(items), 0)

    def test_purge_deferred_1h_kept(self):
        """1-часовой deferred сохраняется при deferred_retention_days=1."""
        self._insert("deferred", hours_ago=1, score=6.0)
        self.storage.purge_items(retention_days=1, deferred_retention_days=1)
        items = self.storage.get_deferred_items()
        self.assertEqual(len(items), 1)

    def test_purge_delivered_25h_removed(self):
        """25-часовой delivered удаляется при retention_days=1."""
        self._insert("delivered", hours_ago=25, score=6.0)
        self.storage.purge_items(retention_days=1, deferred_retention_days=1)
        # delivered не в deferred — смотрим через count
        self.assertEqual(self.storage.count_by_state("delivered"), 0)

    def test_purge_1h_delivered_kept(self):
        """1-часовой delivered сохраняется при retention_days=1."""
        self._insert("delivered", hours_ago=1, score=6.0)
        self.storage.purge_items(retention_days=1, deferred_retention_days=1)
        self.assertEqual(self.storage.count_by_state("delivered"), 1)

    # ─────────────────────────────────────────────────────────────────
    # config defaults
    # ─────────────────────────────────────────────────────────────────

    def test_config_digest_max_age_hours_24(self):
        """Дефолт digest.max_age_hours == 24."""
        self.assertEqual(config_module.DEFAULT_CONFIG["digest"].get("max_age_hours"), 24)

    def test_config_retention_days_1(self):
        """Дефолт database.retention_days == 1."""
        self.assertEqual(config_module.DEFAULT_CONFIG["database"]["retention_days"], 1)

    def test_config_deferred_retention_days_3(self):
        """Дефолт database.deferred_retention_days == 3 (v1.1.5)."""
        self.assertEqual(config_module.DEFAULT_CONFIG["database"]["deferred_retention_days"], 3)


if __name__ == "__main__":
    unittest.main()