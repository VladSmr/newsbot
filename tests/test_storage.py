"""
Тесты storage.py — покрывают tags-поля.
"""
import json
import tempfile
from pathlib import Path

import pytest

from newsbot.storage import Storage
from newsbot.collector import NewsItem


@pytest.fixture
def tmp_db(tmp_path):
    """Временная БД для каждого теста."""
    db_path = tmp_path / "test.db"
    storage = Storage(db_path=db_path)
    yield storage
    storage.close()


class TestTagsField:
    """Задача 1.3: тесты на поле tags."""

    def test_save_and_read_tags(self, tmp_db):
        """Сохранить item с tags, прочитать — tags совпадают."""
        item = NewsItem(
            source="TestSource",
            url="https://example.com/tags-test",
            title="Тест тегов",
            summary="Краткое описание",
            published="2025-01-01",
            tags=json.dumps(["News", "Broadway", "Reviews"]),
        )
        saved = tmp_db.save(item)
        assert saved, "Item должен быть сохранён"

        # Чтение через get_all_by_state
        items = tmp_db.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].tags == json.dumps(["News", "Broadway", "Reviews"])

    def test_save_without_tags(self, tmp_db):
        """Item без tags сохраняется и читается как None."""
        item = NewsItem(
            source="TestSource",
            url="https://example.com/no-tags",
            title="Без тегов",
            summary="Описание без тегов",
            published="2025-01-01",
            tags=None,
        )
        saved = tmp_db.save(item)
        assert saved

        items = tmp_db.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].tags is None

    def test_save_many_with_tags(self, tmp_db):
        """save_many: несколько items с разными tags."""
        items = [
            NewsItem(
                source="Source1",
                url=f"https://example.com/many-tags-{i}",
                title=f"Новость {i}",
                summary="Краткое описание",
                published="2025-01-01",
                tags=json.dumps([f"Tag{i}A", f"Tag{i}B"]),
            )
            for i in range(3)
        ]
        inserted = tmp_db.save_many(items)
        assert len(inserted) == 3

        read = tmp_db.get_all_by_state("new")
        assert len(read) == 3
        for i, ni in enumerate(read):
            expected = json.dumps([f"Tag{i}A", f"Tag{i}B"])
            assert ni.tags == expected, f"Item {i}: tags не совпали"

    def test_tags_in_deferred(self, tmp_db):
        """tags читаются из deferred-items."""
        item = NewsItem(
            source="DeferredSource",
            url="https://example.com/deferred-tags",
            title="Отложенная новость",
            summary="Описание",
            published="2025-01-01",
            tags=json.dumps(["DeferredTag"]),
        )
        tmp_db.save(item)
        # Переводим в deferred
        tmp_db.set_state(["https://example.com/deferred-tags"], "deferred")

        deferred = tmp_db.get_deferred_items()
        assert len(deferred) == 1
        assert deferred[0].tags == json.dumps(["DeferredTag"])

    def test_schema_has_tags_column(self, tmp_db):
        """Колонка tags существует в схеме."""
        conn = tmp_db._connect()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
        assert "tags" in cols, "Колонка tags должна существовать в items"