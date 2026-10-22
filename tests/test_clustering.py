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

    def test_normalize_russian_drone_headline(self):
        """Русский заголовок → не пустой hash (не md5(''))."""
        result = normalize("Сбито 80 БПЛА в Крыму")
        assert result.strip() != "", (
            "normalize() не должен возвращать пустую строку для русского заголовка. "
            "Это указывает на потерю кириллицы при нормализации."
        )
        assert len(result) > 0

    def test_normalize_russian_vs_other_different(self):
        """Два разных русских заголовка → разные строки."""
        r1 = normalize("Сбито 80 БПЛА в Крыму")
        r2 = normalize("Другая новость в Крыму")
        assert r1 != r2, "Разные заголовки должны давать разные стемы"

    def test_stem_hash_cyrillic_nonempty(self):
        """stem_hash() для русского заголовка — не md5('')."""
        h = stem_hash("Сбито 80 БПЛА в Крыму")
        MD5_EMPTY = "d41d8cd98f00b204e9800998ecf8427e"
        assert h != MD5_EMPTY, (
            f"stem_hash вернул md5('') = {h}. Кириллица теряется при нормализации!"
        )
        assert len(h) == 8, "stem_hash должен быть 8-символьным"

    def test_normalize_emoji_headline(self):
        """Emoji/Unicode в заголовке — нормализация не падает, hash не пустой."""
        result = normalize("Breaking 🚨 News 🔥 in Tokyo")
        assert result.strip() != "", "Emoji-заголовок не должен давать пустой normalize()"
        h = stem_hash("Breaking 🚨 News 🔥 in Tokyo")
        assert h != "d41d8cd9", "Emoji-заголовок не должен давать md5('')"

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

    def test_extract_location_two_word(self):
        """2-словная локация: 'South Africa' → 'юар' (n-грамма 2)."""
        assert extract_location("Mining boom in South Africa") == "юар"
        assert extract_location("South Africa elections: what to expect") == "юар"

    def test_extract_location_us_not_helps_us(self):
        """'helps us understand' → НЕ локация US (строчная 'us')."""
        assert extract_location("Study helps us understand the brain") == "other"
        # Но заглавная US как отдельное слово → сша
        assert extract_location("US announces new tariffs on China") == "сша"
        assert extract_location("The US president signed the bill") == "сша"

    def test_extract_location_no_punct_glue(self):
        """Пунктуация не склеивает токены: 'Moscow,' — москва."""
        assert extract_location("Moscow, rain expected today") == "москва"
        # Точка внутри слова — не матчится
        assert extract_location("U.S. sanctions announced") == "other"

    def test_extract_location_cyrillic_ngram(self):
        """3-грамма 'белгородская область' → россия."""
        assert extract_location("В Белгородской области прошли события") == "россия"
        assert extract_location(
            "В Белгородской области прошли события и обстрелы"
        ) == "россия"
        """extract_location возвращает первую найденную."""
        # "France" — первое слово — должно вернуть "франция"
        result = extract_location("France and Germany sign deal in Berlin")
        assert result == "франция"

    def test_extract_location_typographic_quotes(self):
        """NIT: «ёлочки»/многоточие/тире снимаются с краёв токена."""
        assert extract_location("«Москва» — новости дня") == "москва"
        assert extract_location("Москва…") == "москва"
        assert extract_location("Москва — новости дня") == "москва"
        assert extract_location("«Киев»: подробности") == "киев"

    def test_locations_no_duplicate_kiev(self):
        """
        NIT: в исходнике LOCATIONS нет дублирующихся строковых ключей.

        Проверка через импортированный dict бесполезна: в dict-литерале
        дубликат молча перекрывается ещё при создании объекта, и тест
        `LOCATIONS["kiev"] == "киев"` проходит даже при наличии мёртвой
        записи. Поэтому парсим исходник locations.py и ищем повторы ключей
        в присваивании LOCATIONS.
        """
        import ast
        import inspect
        import textwrap

        import newsbot.locations as locations_mod

        tree = ast.parse(textwrap.dedent(inspect.getsource(locations_mod)))
        key_counts = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "LOCATIONS" not in targets or not isinstance(node.value, ast.Dict):
                continue
            for key_node in node.value.keys:
                if isinstance(key_node, ast.Constant) and isinstance(
                    key_node.value, str
                ):
                    key_counts[key_node.value] = (
                        key_counts.get(key_node.value, 0) + 1
                    )

        duplicates = sorted(k for k, n in key_counts.items() if n > 1)
        assert not duplicates, (
            f"в LOCATIONS дубли ключей (перекрываются молча): {duplicates}"
        )

        # Семантика: 'kiev' — город, 'kyiv' — страна (разные ключи).
        assert LOCATIONS["kiev"] == "киев"
        assert LOCATIONS["kyiv"] == "украина"


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

    def test_cluster_key_unknown_day(self):
        """Без published → day='unknown'."""
        class FakeNoPub:
            title = "Breaking news"
            published = None

        key = cluster_key(FakeNoPub())
        parts = key.split("|")
        assert len(parts) == 3
        assert parts[2] == "unknown"

    def test_cluster_key_nostem_per_title_unique(self):
        """
        CRITICAL 1 Regression: два разных nostem-заголовка (empty stem)
        в одной локации и в один день → РАЗНЫЕ cluster_key.
        Раньше: md5(location) для пустого stem → все nostem items сливались
        в один cluster_key → domain_recent_duplicate блокировал 2-й+ item.
        Теперь: per-title hash по исходному title.
        """
        class Fake1:
            title = "🛑"  # pure emoji → empty stem → nostem per-title
            published = "2026-10-21T12:00:00+00:00"

        class Fake2:
            title = "🚨"  # different pure emoji → different per-title hash
            published = "2026-10-21T14:00:00+00:00"

        k1 = cluster_key(Fake1())
        k2 = cluster_key(Fake2())

        # Оба → nostem (пустой stem_hash для этих заголовков)
        # Но РАЗНЫЕ title → РАЗНЫЕ per-title hash
        assert k1 != k2, (
            f"Разные nostem-заголовки должны давать РАЗНЫЕ cluster_key. "
            f"k1={k1!r}, k2={k2!r}"
        )
        # Оба содержат nostem:xxx (per-title hash) и день
        assert "nostem:" in k1, f"Expected nostem: in k1={k1!r}"
        assert "nostem:" in k2, f"Expected nostem: in k2={k2!r}"
        assert "2026-10-21" in k1
        assert "2026-10-21" in k2

    def test_cluster_key_iso_date_format(self):
        """cluster_key содержит ISO-дату YYYY-MM-DD из published."""
        class FakeItem:
            title = "Test headline"
            published = "2026-10-21T14:30:00+00:00"

        key = cluster_key(FakeItem())
        parts = key.split("|")
        assert parts[2] == "2026-10-21", (
            f"day должен быть ISO YYYY-MM-DD, получил: {parts[2]}"
        )

    def test_cluster_key_date_from_published_not_fetched(self):
        """cluster_key использует published, не fetched_at."""
        class Fake1:
            title = "Same news"
            published = "2026-10-21T08:00:00+00:00"

        class Fake2:
            title = "Same news"   # тот же title — тот же stem
            published = "2026-10-21T08:00:00+00:00"   # тот же published

        k1 = cluster_key(Fake1())
        k2 = cluster_key(Fake2())
        assert k1 == k2, "Один title + published → идентичный cluster_key"


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