"""
P1-регресс: защищённый разбор dedup.* из конфига (без parse-guard).

Сценарии:
  1) Опечатка в числовых полях (threshold: high, window_days: 3d) НЕ роняет
     filter_new — иначе цикл обрывался бы до save_many и новости терялись.
  2) cross_source: "false" (строка) НЕ должен трактоваться как truthy.
  3) window_days <= 0 → clamp до минимума + warning (дедуп не выключается молча).
"""
import tempfile
import os
from pathlib import Path

import pytest

from newsbot.collector import NewsItem
from newsbot.dedup import filter_new
from newsbot.storage import Storage


def _mk_item(source="SourceA", title="SpaceX launches new Starship mission",
             url="https://a.com/1") -> NewsItem:
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary="Test summary text",
        published="01.01.2025 12:00 UTC",
        published_parsed=None,
    )


@pytest.fixture()
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


class TestDedupConfigParseGuard:
    def test_typo_in_numeric_fields_does_not_raise(self, db_path):
        """threshold: high / window_days: 3d → дефолты, item остаётся новым."""
        cfg = {
            "cross_source": True,
            "fuzzy": True,
            "threshold": "high",
            "token_threshold": "low",
            "window_days": "3d",
        }
        st = Storage(db_path, dedup_config=cfg)
        try:
            item = _mk_item()
            new_items, total, dup = filter_new([item], st, dedup_config=cfg)
            assert total == 1
            assert dup == 0
            assert len(new_items) == 1
        finally:
            st.close()

    def test_cross_source_string_false_is_disabled(self, db_path):
        """cross_source: "false" (строка) → дедуп выключен, а не truthy."""
        cfg = {"cross_source": "false", "fuzzy": False, "window_days": 3}
        st = Storage(db_path, dedup_config=cfg)
        try:
            first = _mk_item(source="SourceA", url="https://a.com/x")
            assert st.save(first) is True

            second = _mk_item(source="SourceB", url="https://b.com/x")
            new_items, total, dup = filter_new([second], st, dedup_config=cfg)
            assert dup == 0, "cross_source='false' не должен включать дедуп"
            assert len(new_items) == 1
        finally:
            st.close()

    def test_window_days_zero_clamped_and_warns(self, db_path, caplog):
        """window_days=0 → clamp до 1 день + warning, без исключения."""
        cfg = {"cross_source": True, "fuzzy": False, "window_days": 0}
        st = Storage(db_path, dedup_config=cfg)
        try:
            item = _mk_item()
            with caplog.at_level("WARNING", logger="newsbot.dedup"):
                new_items, total, dup = filter_new([item], st, dedup_config=cfg)
            assert len(new_items) == 1
            assert any("window_days" in r.message for r in caplog.records)
        finally:
            st.close()

    def test_bool_typo_falls_back_to_default(self, db_path, caplog):
        """Невалидный bool (например 'maybe') → дефолт + warning."""
        cfg = {"cross_source": "maybe", "fuzzy": "maybe", "window_days": 3}
        st = Storage(db_path, dedup_config=cfg)
        try:
            item = _mk_item()
            with caplog.at_level("WARNING", logger="newsbot.dedup"):
                new_items, total, dup = filter_new([item], st, dedup_config=cfg)
            assert len(new_items) == 1
            assert any("cross_source" in r.message for r in caplog.records)
        finally:
            st.close()

    def test_cross_source_int_zero_disables_without_warning(self, db_path, caplog):
        """Major-регресс: cross_source: 0 (int) → дедуп выключен, без warning."""
        cfg = {"cross_source": 0, "fuzzy": 0, "window_days": 3}
        st = Storage(db_path, dedup_config=cfg)
        try:
            first = _mk_item(source="SourceA", url="https://a.com/int0")
            assert st.save(first) is True

            second = _mk_item(source="SourceB", url="https://b.com/int0")
            with caplog.at_level("WARNING", logger="newsbot.dedup"):
                new_items, total, dup = filter_new([second], st, dedup_config=cfg)

            assert dup == 0, "cross_source=0 не должен включать кросс-дедуп"
            assert len(new_items) == 1
            assert not any(
                "cross_source" in r.message for r in caplog.records
            ), "int 0 — валидное значение, warning не нужен"
        finally:
            st.close()

    def test_cross_source_int_one_enables(self, db_path):
        """cross_source: 1 (int) → дедуп включён."""
        cfg = {"cross_source": 1, "fuzzy": 0, "window_days": 3}
        st = Storage(db_path, dedup_config=cfg)
        try:
            first = _mk_item(source="SourceA", url="https://a.com/int1")
            assert st.save(first) is True

            second = _mk_item(source="SourceB", url="https://b.com/int1")
            new_items, total, dup = filter_new([second], st, dedup_config=cfg)
            assert dup == 1, "cross_source=1 должен включать кросс-дедуп"
            assert len(new_items) == 0
        finally:
            st.close()

    def test_window_days_float_floored_with_warning(self, db_path, caplog):
        """window_days=2.9 → floor до 2 + warning (не молчаливое усечение)."""
        cfg = {"cross_source": True, "fuzzy": False, "window_days": 2.9}
        st = Storage(db_path, dedup_config=cfg)
        try:
            item = _mk_item()
            with caplog.at_level("WARNING", logger="newsbot.dedup"):
                new_items, total, dup = filter_new([item], st, dedup_config=cfg)
            assert len(new_items) == 1
            assert any(
                "window_days" in r.message and "округлено" in r.message
                for r in caplog.records
            )
        finally:
            st.close()

    def test_window_days_upper_clamp_warns(self, db_path, caplog):
        """window_days=300 → clamp до 30 + warning (не сканируем год заголовков)."""
        from newsbot.dedup import MAX_DEDUP_WINDOW_DAYS

        cfg = {"cross_source": True, "fuzzy": False, "window_days": 300}
        st = Storage(db_path, dedup_config=cfg)
        try:
            item = _mk_item()
            with caplog.at_level("WARNING", logger="newsbot.dedup"):
                new_items, total, dup = filter_new([item], st, dedup_config=cfg)
            assert len(new_items) == 1
            assert any(
                "window_days" in r.message
                and str(MAX_DEDUP_WINDOW_DAYS) in r.message
                for r in caplog.records
            )
        finally:
            st.close()
