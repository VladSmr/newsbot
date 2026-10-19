"""Тесты стеммера, локаций и кластеризации (Задача 2)."""
import pytest

from newsbot.stemmer import normalize, stem_hash
from newsbot.locations import extract_location, cluster_key, LOCATIONS


class TestStemmer:
    """Тесты stemmer.py."""

    def test_stem_normalize_basic(self):
        """normalize() убирает числа, стоп-слова, короткие слова."""
        result = normalize("The police clash in Paris 2025")
        assert "the" not in result
        assert "police" in result
        assert "paris" in result
        assert "2025" not in result

    def test_stem_hash_deterministic(self):
        """stem_hash() детерминирован: одинаковый вход → одинаковый выход."""
        h1 = stem_hash("France protests: police clash in Paris")
        h2 = stem_hash("France protests: police clash in Paris")
        assert h1 == h2

    def test_stem_hash_similar_titles(self):
        """Синонимичные заголовки дают одинаковый stem_hash."""
        h1 = stem_hash("France protests: police clash in Paris")
        h2 = stem_hash("Paris protest clash: police in France")
        assert h1 == h2

    def test_stem_hash_different_titles(self):
        """Разные заголовки дают разные stem_hash."""
        h1 = stem_hash("France protests in Paris")
        h2 = stem_hash("Japan earthquake in Tokyo")
        assert h1 != h2

    def test_normalize_cyrillic(self):
        """normalize() работает с кириллицей."""
        result = normalize("В Белгородской области прошли события")
        assert "в" not in result
        assert "белгородск" in result or "област" in result

    def test_stem_suffix_stripping(self):
        """Суффиксы корректно убираются."""
        result = normalize("protesting protesters protestation")
        assert "protest" in result


class TestLocations:
    """Тесты locations.py."""

    def test_extract_location_cyrillic(self):
        """Извлечение локации из кириллического заголовка."""
        assert extract_location("В Белгородской области...") == "россия"
        assert extract_location("В Москве прошёл митинг") == "москва"
        assert extract_location("Жители Киева вышли на акцию") == "киев"

    def test_extract_location_english_countries(self):
        """Извлечение локации — страны (EN)."""
        assert extract_location("France protests: police clash in Paris") == "франция"
        assert extract_location("In Ukraine, tensions escalate") == "украина"
        assert extract_location("USA and China trade deal announced") == "сша"

    def test_extract_location_english_cities(self):
        """Извлечение локации — города (EN)."""
        assert extract_location("LONDON - Big news today") == "лондон"
        assert extract_location("BERLIN hosts conference") == "берлин"

    def test_extract_location_no_match(self):
        """Без совпадения возвращается 'other'."""
        assert extract_location("Something happened somewhere") == "other"

    def test_extract_location_first_match(self):
        """extract_location возвращает первую найденную."""
        # "France" — первое слово — должно вернуть "франция"
        result = extract_location("France and Germany sign deal in Berlin")
        assert result == "франция"


class TestClusterKey:
    """Тесты cluster_key()."""

    def test_cluster_key_structure(self):
        """cluster_key содержит локацию, stem, дату."""
        class FakeItem:
            title = "France protests: police clash in Paris"
            published = "2026-10-18T20:00:00+00:00"

        key = cluster_key(FakeItem())
        parts = key.split("|")
        assert len(parts) == 3
        assert parts[0] == "франция"
        assert parts[2] == "2026-10-18"

    def test_cluster_key_stem_consistency(self):
        """Одинаковые title дают одинаковый stem в cluster_key."""
        class Fake1:
            title = "France protests: police clash in Paris"
            published = "2026-10-18T20:00:00+00:00"

        class Fake2:
            title = "Paris protest clash: police in France"
            published = "2026-10-18T20:00:00+00:00"

        k1 = cluster_key(Fake1())
        k2 = cluster_key(Fake2())
        # stem одинаковый, локация и день совпадают
        assert k1 == k2

    def test_cluster_key_different_days(self):
        """Разные дни — разные cluster_key."""
        class Fake1:
            title = "France protests in Paris"
            published = "2026-10-18T20:00:00+00:00"

        class Fake2:
            title = "France protests in Paris"
            published = "2026-10-19T20:00:00+00:00"

        k1 = cluster_key(Fake1())
        k2 = cluster_key(Fake2())
        assert k1 != k2
        assert "2026-10-18" in k1
        assert "2026-10-19" in k2


class TestCollectorClusterKey:
    """Тесты cluster_key в NewsItem (Задача 2.3)."""

    def test_newsitem_cluster_key_field(self):
        """NewsItem имеет поле cluster_key."""
        from newsbot.collector import NewsItem

        item = NewsItem(
            source="Test",
            url="https://example.com/test",
            title="Test title",
            summary="summary",
            published="2026-10-18",
            cluster_key="россия|abc12345|2026-10-18",
        )
        assert item.cluster_key == "россия|abc12345|2026-10-18"

    def test_newsitem_cluster_key_default_none(self):
        """cluster_key по умолчанию None."""
        from newsbot.collector import NewsItem

        item = NewsItem(
            source="Test",
            url="https://example.com/test",
            title="Test title",
            summary="summary",
            published="2026-10-18",
        )
        assert item.cluster_key is None

    def test_storage_saves_and_reads_cluster_key(self, tmp_path=None):
        """Storage сохраняет и читает cluster_key."""
        from newsbot.storage import Storage
        from newsbot.collector import NewsItem
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            storage = Storage(db_path=db_path)

            item = NewsItem(
                source="Test",
                url="https://example.com/cluster-test",
                title="France protests in Paris",
                summary="summary",
                published="18.10.2026 20:00 UTC",
                cluster_key="франция|abc12345|2026-10-18",
            )
            saved = storage.save(item)
            assert saved, "Item должен быть сохранён"

            # Чтение
            items = storage.get_all_by_state("new")
            assert len(items) == 1
            assert items[0].cluster_key == "франция|abc12345|2026-10-18"
            storage.close()

    def test_storage_save_many_with_cluster_key(self, tmp_path=None):
        """save_many() сохраняет cluster_key."""
        from newsbot.storage import Storage
        from newsbot.collector import NewsItem
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            storage = Storage(db_path=db_path)

            items = [
                NewsItem(
                    source="Test",
                    url="https://example.com/c1",
                    title="Test 1",
                    summary="s1",
                    published="2026-10-18",
                    cluster_key="россия|aaa|2026-10-18",
                ),
                NewsItem(
                    source="Test",
                    url="https://example.com/c2",
                    title="Test 2",
                    summary="s2",
                    published="2026-10-18",
                    cluster_key="сша|bbb|2026-10-18",
                ),
            ]
            inserted = storage.save_many(items)
            assert len(inserted) == 2

            # Проверяем cluster_key при чтении
            all_items = storage.get_all_by_state("new")
            keys = {i.cluster_key for i in all_items}
            assert "россия|aaa|2026-10-18" in keys
            assert "сша|bbb|2026-10-18" in keys
            storage.close()