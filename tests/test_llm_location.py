"""
Тесты: LLM-локация вместо словаря (Задача: заменить locations.py на поле location из LLM).

Покрывает:
  1. Парсинг location в FilterResult (ok=True с location, ok=False без location).
  2. Cap-группировка по LLM-локациям (normalize, lowercase, empty→other).
  3. rebuild_cluster_key с LLM-локацией.
  4. Отсутствие ссылок на locations.py.
  5. Регресс: FilterResult без поля location → пустая строка.
  6. Регресс: фильтр без ответа location продолжает работать.
"""
import json
import unittest
from unittest.mock import MagicMock

import sys
from pathlib import Path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.pipeline import rebuild_cluster_key, _cap_by_location, build_cluster_key, normalize_location


# ─────────────────────────────────────────────────────────────────
# Fixtures — мок HTTP-клиент для filter.py
# ─────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self,
            )


_module_httpx_saved = None


def _mock_httpx(mock_cls):
    global _module_httpx_saved
    import newsbot.llm.client as cm
    if _module_httpx_saved is None:
        _module_httpx_saved = cm.httpx.Client
    cm.httpx.Client = mock_cls


def _unmock():
    global _module_httpx_saved
    if _module_httpx_saved is not None:
        import newsbot.llm.client as cm
        cm.httpx.Client = _module_httpx_saved
        _module_httpx_saved = None


# ─────────────────────────────────────────────────────────────────
# 1. Парсинг location в FilterResult
# ─────────────────────────────────────────────────────────────────

class TestLocationParsing(unittest.TestCase):
    """Тесты парсинга поля location в _parse_response."""

    def _parse(self, raw: dict):
        from newsbot.llm.client import DeepSeekClient
        from newsbot.llm.filter import LLMFilter

        resp_str = json.dumps(raw)

        class _Mock:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return FakeResponse(
                    {"choices": [{"message": {"content": resp_str}}]},
                    status_code=200,
                )

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        _mock_httpx(_Mock)
        client = DeepSeekClient("key", "https://api.deepseek.com", "model")
        cfg = {
            "enabled": True,
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "mock-key",
            "push_threshold": 7,
            "digest_threshold": 5,
            "system_prompt": "test",
        }
        flt = LLMFilter(cfg, client=client)
        item = MagicMock()
        item.title = "Test"
        item.summary = "Test"
        item.source = "Test"

        try:
            return flt.evaluate(item)
        finally:
            _unmock()

    def test_location_present(self):
        """LLM возвращает location → FilterResult.location заполнен."""
        r = self._parse({
            "score": 8,
            "summary": "ok",
            "reason": "r",
            "urgent": False,
            "location": "China",
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.location, "China")

    def test_location_cyrillic_input(self):
        """LLM распознаёт кириллическую новость → location 'China'."""
        r = self._parse({
            "score": 7,
            "summary": "ok",
            "reason": "r",
            "urgent": False,
            "location": "China",
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.location, "China")

    def test_location_empty_string(self):
        """Пустая строка location → пустая строка в FilterResult."""
        r = self._parse({
            "score": 8,
            "summary": "ok",
            "reason": "r",
            "urgent": False,
            "location": "",
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.location, "")

    def test_location_missing_defaults_empty(self):
        """Отсутствующее поле location → пустая строка (регресс)."""
        r = self._parse({
            "score": 8,
            "summary": "ok",
            "reason": "r",
            "urgent": False,
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.location, "")

    def test_location_null_treated_as_empty(self):
        """null location → пустая строка."""
        r = self._parse({
            "score": 8,
            "summary": "ok",
            "reason": "r",
            "urgent": False,
            "location": None,
        })
        self.assertTrue(r.ok)
        self.assertEqual(r.location, "")


# ─────────────────────────────────────────────────────────────────
# 2. Cap-группировка по LLM-локациям (lowercase normalize)
# ─────────────────────────────────────────────────────────────────

class TestCapByLocationLLM(unittest.TestCase):
    """_cap_by_location группирует по нормализованной (lowercase) LLM-локации."""

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_cap_groups_by_normalized_location(self):
        """
        Cap группирует по lowercase-нормализованной локации:
        'China', 'CHINA', 'china' — одна группа cap=2.
        """
        items = [
            self.FakeItem("u1", "china|stem1|2026-10-20"),
            self.FakeItem("u2", "CHINA|stem2|2026-10-20"),
            self.FakeItem("u3", "China|stem3|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # Cap=2 на группу 'china' — должно остаться 2 из 3
        self.assertEqual(len(result), 2)

    def test_cap_cyrillic_title_china(self):
        """
        Реальный кейс: кириллическая новость про Китай.
        LLM возвращает 'China', cluster_key после rebuild → 'china|...'.
        Cap должен кэпировать эти items.
        """
        # Симулируем 5 новостей про Китай, все с локацией 'china'
        items = [self.FakeItem(f"u{i}", f"china|stem{i}|2026-10-20") for i in range(5)]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_cap_different_locations_not_grouped(self):
        """Разные LLM-локации — разные группы cap."""
        items = [
            self.FakeItem("u1", "china|stem1|2026-10-20"),
            self.FakeItem("u2", "russia|stem2|2026-10-20"),
            self.FakeItem("u3", "taiwan|stem3|2026-10-20"),
            self.FakeItem("u4", "china|stem4|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # china: cap=2 (u1,u4), russia: cap=2 (u2), taiwan: cap=2 (u3) → все 4
        self.assertEqual(len(result), 4)

    def test_cap_empty_location_becomes_other(self):
        """Пустая/other-локация НЕ кэпируется (как раньше)."""
        items = [
            self.FakeItem("u1", "other|stem1|2026-10-20"),
            self.FakeItem("u2", "other|stem2|2026-10-20"),
            self.FakeItem("u3", "other|stem3|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # 'other' не кэпируется — все 3 проходят
        self.assertEqual(len(result), 3)

    def test_cap_mixed_llm_and_placeholder(self):
        """Mixed: LLM-локации и stub 'other' — разные группы."""
        items = [
            self.FakeItem("u1", "china|stem1|2026-10-20"),   # LLM-локация
            self.FakeItem("u2", "china|stem2|2026-10-20"),
            self.FakeItem("u3", "other|stem3|2026-10-20"),   # stub 'other'
        ]
        result = _cap_by_location(items, max_per_location=1)
        # china cap=1 → 1 item; other не кэпируется → 1 item; итого 2
        self.assertEqual(len(result), 2)

    def test_cap_per_day_groups(self):
        """Cap считается отдельно по дням."""
        items = [
            self.FakeItem("u1", "china|stem1|2026-10-20"),
            self.FakeItem("u2", "china|stem2|2026-10-20"),
            self.FakeItem("u3", "china|stem3|2026-10-19"),
            self.FakeItem("u4", "china|stem4|2026-10-19"),
            self.FakeItem("u5", "china|stem5|2026-10-19"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        # 2026-10-20: 2 items; 2026-10-19: 2 items; итого 4
        self.assertEqual(len(result), 4)


# ─────────────────────────────────────────────────────────────────
# 3. rebuild_cluster_key
# ─────────────────────────────────────────────────────────────────

class TestRebuildClusterKeyLLM(unittest.TestCase):
    """rebuild_cluster_key подставляет LLM-локацию в cluster_key."""

    def test_rebuild_with_stem(self):
        """Old cluster_key с stem → stem сохраняется, location заменяется."""
        old = "other|abc123|2026-10-20"
        new = rebuild_cluster_key("Tibet", old)
        parts = new.split("|")
        self.assertEqual(parts[0], "tibet")
        self.assertEqual(parts[1], "abc123")
        self.assertEqual(parts[2], "2026-10-20")

    def test_rebuild_normalizes_location(self):
        """LLM-локация нормализуется: "China" → "china"."""
        old = "other|abc|2026-10-20"
        new = rebuild_cluster_key("CHINA", old)
        self.assertTrue(new.startswith("china|"))

    def test_rebuild_empty_location_becomes_other(self):
        """Пустая LLM-локация → 'other'."""
        old = "other|abc|2026-10-20"
        new = rebuild_cluster_key("", old)
        self.assertTrue(new.startswith("other|"))

    def test_rebuild_no_stem(self):
        """Old key без stem → nostem с пустым title hash."""
        old = "other||2026-10-20"  # пустой stem
        new = rebuild_cluster_key("Russia", old)
        parts = new.split("|")
        self.assertEqual(parts[0], "russia")
        self.assertTrue(parts[1].startswith("nostem:"))

    def test_rebuild_no_old_key(self):
        """Нет old_cluster_key → nostem с пустым title hash."""
        new = rebuild_cluster_key("France", None)
        parts = new.split("|")
        self.assertEqual(parts[0], "france")
        self.assertTrue(parts[1].startswith("nostem:"))

    def test_rebuild_same_old_key_different_title_different_keys(self):
        """
        Minor-фикс: два вызова с одинаковым old_cluster_key (без stem),
        но РАЗНЫМИ title → РАЗНЫЕ cluster_key.
        Это ключевое свойство nostem-per-title hash: разные новости
        без stem не должны склеиваться в один cluster.
        """
        old_key = "other||2026-10-20"  # пустой stem
        k1 = rebuild_cluster_key("Russia", old_key, title="AI 2024")
        k2 = rebuild_cluster_key("Russia", old_key, title="AI 2025")
        self.assertNotEqual(
            k1, k2,
            "Одинаковый old_key, разные title → разные cluster_key"
        )
        # Оба начинаются с 'russia' и содержат nostem:
        self.assertTrue(k1.startswith("russia|nostem:"))
        self.assertTrue(k2.startswith("russia|nostem:"))
        # Хеши разные
        self.assertNotEqual(k1.split("|")[1], k2.split("|")[1])

    def test_rebuild_title_none_old_key_nostem_uses_old_key_stem(self):
        """
        При old_cluster_key с nostem и title=None: nostem-ветка использует
        пустой title → md5(b"") — детерминированный, но неуникальный hash.
        Ключ не пустой (содержит 'nostem:da39a3ee5e6b').
        Это крайний случай: реальные items всегда имеют title или url
        (url fallback обрабатывается вызывающей стороной в _process_batch).
        """
        old_key = "other||2026-10-20"  # пустой stem
        k = rebuild_cluster_key("Germany", old_key, title=None)
        parts = k.split("|")
        self.assertEqual(parts[0], "germany")
        self.assertTrue(parts[1].startswith("nostem:"))
        # md5(b"").hexdigest()[:12] = da39a3ee5e6b
        self.assertEqual(parts[1], "nostem:d41d8cd98f00")
        self.assertNotEqual(k, "")  # не пустой ключ

    def test_rebuild_title_provided_used_for_nostem_hash(self):
        """
        title используется для вычисления nostem-hash.
        Два вызова с РАЗНЫМИ title и пустым stem → разные keys.
        """
        old_key = "other||2026-10-20"
        import hashlib
        h_a = hashlib.md5("Headline Alpha".encode()).hexdigest()[:12]
        h_b = hashlib.md5("Headline Beta".encode()).hexdigest()[:12]
        k_a = rebuild_cluster_key("France", old_key, title="Headline Alpha")
        k_b = rebuild_cluster_key("France", old_key, title="Headline Beta")
        self.assertTrue(k_a.endswith(f"nostem:{h_a}|2026-10-20"), f"Got: {k_a}")
        self.assertTrue(k_b.endswith(f"nostem:{h_b}|2026-10-20"), f"Got: {k_b}")

    def test_rebuild_empty_string_title_vs_none_same_hash(self):
        """
        title='' и title=None дают одинаковый md5(b"")-based hash
        (крайний случай, оба означают «нет данных для хеша»).
        """
        old_key = "other||2026-10-20"
        k_none = rebuild_cluster_key("Spain", old_key, title=None)
        k_empty = rebuild_cluster_key("Spain", old_key, title="")
        self.assertEqual(k_none, k_empty,
                         "title=None и title='' должны давать одинаковый hash")


# ─────────────────────────────────────────────────────────────────
# 4. locations.py нигде не импортируется
# ─────────────────────────────────────────────────────────────────

class TestNoLocationsImport(unittest.TestCase):
    """Grep-check: locations.py не импортируется."""

    def test_no_locations_module_import(self):
        """Импорт newsbot.locations должен вызвать ImportError."""
        with self.assertRaises(ImportError):
            import newsbot.locations  # noqa: F401


# ─────────────────────────────────────────────────────────────────
# 5. Регресс: FilterResult без поля location → пустая строка
# ─────────────────────────────────────────────────────────────────

class TestLocationRegression(unittest.TestCase):
    """Регресс: старый формат ответа LLM (без location) не роняет парсинг."""

    def test_old_response_format_no_location(self):
        """Ответ LLM без поля 'location' (старый формат) → пустая location."""
        class _Mock:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                # Старый формат: без location
                return FakeResponse(
                    {"choices": [{"message": {
                        "content": '{"score": 8, "summary": "ok", "reason": "r"}'
                    }}]},
                    status_code=200,
                )

            def close(s):
                pass

            def __enter__(s):
                return s

            def __exit__(s, *args):
                pass

        _mock_httpx(_Mock)
        from newsbot.llm.client import DeepSeekClient
        from newsbot.llm.filter import LLMFilter

        client = DeepSeekClient("key", "https://api.deepseek.com", "model")
        cfg = {
            "enabled": True,
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "mock-key",
            "push_threshold": 7,
            "digest_threshold": 5,
            "system_prompt": "test",
        }
        flt = LLMFilter(cfg, client=client)
        item = MagicMock()
        item.title = "Test"
        item.summary = "Test"
        item.source = "Test"

        try:
            res = flt.evaluate(item)
            self.assertTrue(res.ok)
            self.assertEqual(res.location, "")
        finally:
            _unmock()


# ─────────────────────────────────────────────────────────────────
# 6. build_cluster_key API (ранее locations.build_cluster_key)
# ─────────────────────────────────────────────────────────────────

class TestBuildClusterKeyAPI(unittest.TestCase):
    """API build_cluster_key работает как раньше (для обратной совместимости)."""

    def test_nostem_different_titles_different_keys(self):
        """Per-title hash для nostem: разные title → разные keys."""
        k1 = build_cluster_key("france", None, "2026-01-01", title="AI 2024")
        k2 = build_cluster_key("france", None, "2026-01-01", title="AI 2025")
        self.assertNotEqual(k1, k2)

    def test_same_title_same_key(self):
        """Идентичный title → идентичный cluster_key."""
        k1 = build_cluster_key("france", None, "2026-01-01", title="Same")
        k2 = build_cluster_key("france", None, "2026-01-01", title="Same")
        self.assertEqual(k1, k2)

    def test_with_stem(self):
        """Со stem: cluster_key содержит stem и день."""
        k = build_cluster_key("china", "abc123", "2026-10-20")
        parts = k.split("|")
        self.assertEqual(parts[0], "china")
        self.assertEqual(parts[1], "abc123")
        self.assertEqual(parts[2], "2026-10-20")


# ─────────────────────────────────────────────────────────────────
# 7. MAJOR 5: normalize_location и канонизация cap-группировки
# ─────────────────────────────────────────────────────────────────

class TestNormalizeLocation(unittest.TestCase):
    """Тесты normalize_location() — лёгкая канонизация БЕЗ словаря."""

    def _norm(self, loc):
        from newsbot.pipeline import normalize_location
        return normalize_location(loc)

    def test_strip_and_lowercase(self):
        """strip + lowercase."""
        self.assertEqual(self._norm("  Moscow, Russia  "), "moscow, russia")
        self.assertEqual(self._norm("KRASNODAR, RUSSIA"), "krasnodar, russia")

    def test_double_space_collapsed(self):
        """Двойной пробел → одинарный. Без запятой в исходной строке — запятая не добавляется (без словаря)."""
        # "Moscow  Russia" (без запятой) → space collapse → "Moscow Russia" → "moscow russia"
        self.assertEqual(self._norm("Moscow  Russia"), "moscow russia")
        self.assertEqual(self._norm("Moscow   Russia  "), "moscow russia")

    def test_comma_space_normalized(self):
        """Comma + space already present → stays."""
        self.assertEqual(self._norm("Moscow, Russia"), "moscow, russia")

    def test_trailing_comma_stripped(self):
        """Trailing comma (с пробелами) убирается; comma+space в середине сохраняется."""
        # "moscow , russia," → strip(" ,") → "moscow , russia"
        #   → punct norm: " , " → ", " → "moscow, russia"
        self.assertEqual(self._norm("moscow , russia,"), "moscow, russia")

    def test_trailing_dot_stripped(self):
        """Trailing точка убирается."""
        self.assertEqual(self._norm("Krasnodar, Russia."), "krasnodar, russia")

    def test_leading_the_stripped(self):
        """Leading 'The' убирается."""
        self.assertEqual(self._norm("The Moscow, Russia"), "moscow, russia")
        self.assertEqual(self._norm("THE Russia"), "russia")

    def test_empty_and_none(self):
        """None и пустая/blank строка → пустая."""
        self.assertEqual(self._norm(""), "")
        self.assertEqual(self._norm("   "), "")
        self.assertEqual(self._norm(None), "")

    def test_semicolon_collapsed(self):
        """Точка с запятой → запятая."""
        self.assertEqual(self._norm("Moscow ; Russia"), "moscow, russia")


class TestCapByLocationCanonicalization(unittest.TestCase):
    """
    MAJOR 5-тест: канонизация группировки cap.

    Схема нормализации: normalize_location() (strip/lower/collapse spaces/
    punct / leading-the-strip). LLM-инструкция в промпте гарантирует,
    что ОДНО место возвращается как ОДНА и та же строка. Нормализация
    снимает вариации капитализации и trailing punctuation.

    Ключевое: «Moscow» и «Moscow, Russia» — РАЗНЫЕ строки после нормализации.
    Это осознанное решение: если LLM вернула «Moscow» (без страны) — значит
    она не уверена, что событие в Москве, а не в другом городе с таким же
    названием. Cap группирует по нормализованной ПОЛНОЙ строке; если LLM
    дала «Moscow» для одних новостей и «Moscow, Russia» для других — это
    разные намерения модели, cap не должен их склеивать принудительно.
    Принудительное склеивание по последней части (без словаря) привело бы
    к ложным срабатываниям (cap-фрагментация через «City, Country» vs «Country»).
    """

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def _norm(self, loc):
        from newsbot.pipeline import normalize_location
        return normalize_location(loc)

    def test_cap_same_variants_grouped(self):
        """
        Одинаковые написания с разной капитализацией — ОДИН бакет:
        'China', 'CHINA', 'china' → cap=2 → 2 items.
        """
        items = [
            self.FakeItem("u1", "china|stem1|2026-10-20"),
            self.FakeItem("u2", "CHINA|stem2|2026-10-20"),
            self.FakeItem("u3", "China|stem3|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_cap_comma_normalization_same_bucket(self):
        """
        «Moscow, Russia» и «Moscow  Russia» → normalize_location даёт
        одинаковую строку → ОДИН бакет cap.
        """
        k1 = "moscow, russia|stem1|2026-10-20"
        k2 = "moscow, russia|stem2|2026-10-20"  # double-space collapsed by rebuild
        items = [
            self.FakeItem("u1", k1),
            self.FakeItem("u2", k2),
            self.FakeItem("u3", "moscow, russia|stem3|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_cap_trailing_punct_stripped_same_bucket(self):
        """
        «Krasnodar, Russia.» и «Krasnodar, Russia» → trailing dot убран →
        одинаковая нормализованная строка → ОДИН бакет.
        """
        # normalize_location: «Krasnodar, Russia.» → «krasnodar, russia»
        # normalize_location: «Krasnodar, Russia» → «krasnodar, russia»
        k1 = "krasnodar, russia|stem1|2026-10-20"
        k2 = "krasnodar, russia|stem2|2026-10-20"
        items = [self.FakeItem("u1", k1), self.FakeItem("u2", k2)]
        result = _cap_by_location(items, max_per_location=1)
        self.assertEqual(len(result), 1)

    def test_cap_moscow_vs_moscow_russia_different_buckets(self):
        """
        «Moscow» vs «Moscow, Russia» — РАЗНЫЕ нормализованные строки → разные
        баке cap. Это осознанное: LLM-инструкция формата «City, Country» means
        if model says just 'Moscow' — it's not asserting the country.
        Cap should NOT force-merge these without dictionary.
        """
        # Нормализация: "Moscow" → "moscow"; "Moscow, Russia" → "moscow, russia"
        self.assertNotEqual(self._norm("Moscow"), self._norm("Moscow, Russia"))
        k_moscow = "moscow|stem1|2026-10-20"
        k_moscow_russia = "moscow, russia|stem2|2026-10-20"
        items = [
            self.FakeItem("u1", k_moscow),
            self.FakeItem("u2", k_moscow_russia),
        ]
        # Два разных бакета → cap=2 на каждый → оба проходят
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_cap_leading_the_different_bucket(self):
        """
        «The Moscow, Russia» и «Moscow, Russia» → leading 'The' убран →
        одинаковая строка → ОДИН бакет.
        """
        k1 = "moscow, russia|stem1|2026-10-20"
        k2 = "moscow, russia|stem2|2026-10-20"
        items = [self.FakeItem("u1", k1), self.FakeItem("u2", k2)]
        result = _cap_by_location(items, max_per_location=1)
        self.assertEqual(len(result), 1)


class TestCapLocationCyrillicChina(unittest.TestCase):
    """
    Интеграционный тест: кириллическая новость → LLM вернула 'China'
    → rebuild_cluster_key → normalize_location → _cap_by_location → ≤2 items.
    """

    class FakeItem:
        def __init__(self, url, cluster_key, llm_score=8):
            self.url = url
            self.cluster_key = cluster_key
            self.llm_score = llm_score

    def test_cyrillic_title_china_cluster_key_and_cap(self):
        """
        Симулируем цепочку: кириллический title → LLM «China» →
        rebuild_cluster_key → cluster_key с 'china' → cap оставляет ≤2.
        """
        from newsbot.pipeline import rebuild_cluster_key, _cap_by_location

        # Исходные cluster_key без LLM-локации (ингест)
        old_keys = [f"other|stem{i}|2026-10-20" for i in range(5)]

        # LLM вернула 'China' для всех 5 новостей с кириллическими заголовками
        rebuilt_keys = [rebuild_cluster_key("China", old) for old in old_keys]

        # Все rebuilt keys имеют локацию 'china'
        for k in rebuilt_keys:
            self.assertTrue(k.startswith("china|"), f"Expected china prefix: {k}")

        # Cap=2: 5 items проходят rebuild → 2 остаются
        items = [self.FakeItem(f"https://x.com/{i}", k) for i, k in enumerate(rebuilt_keys)]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 2)

    def test_mixed_locations_cap_respected(self):
        """Mixed: China + Russia → cap считается отдельно по локациям."""
        from newsbot.pipeline import rebuild_cluster_key, _cap_by_location

        old = "other|aaa|2026-10-20"
        china_keys = [rebuild_cluster_key("China", old) for _ in range(3)]
        russia_keys = [rebuild_cluster_key("Russia", old) for _ in range(3)]

        items = [
            self.FakeItem(f"china_{i}", k)
            for i, k in enumerate(china_keys)
        ] + [
            self.FakeItem(f"russia_{i}", k)
            for i, k in enumerate(russia_keys)
        ]

        result = _cap_by_location(items, max_per_location=2)
        # china: 2, russia: 2 → всего 4
        self.assertEqual(len(result), 4)


class TestCapEmptyLocationOtherNotCapped(unittest.TestCase):
    """
    Регресс: пустая location → 'other' → cap НЕ ограничивает.
    """

    class FakeItem:
        def __init__(self, url, cluster_key):
            self.url = url
            self.cluster_key = cluster_key

    def test_empty_location_becomes_other_not_capped(self):
        """Пустая LLM-локация → rebuild → 'other' → cap пропускает все."""
        from newsbot.pipeline import rebuild_cluster_key, _cap_by_location

        old = "other|aaa|2026-10-20"
        # Пустая локация → 'other'
        k = rebuild_cluster_key("", old)
        self.assertTrue(k.startswith("other|"), f"Expected other prefix: {k}")

        # 5 items с 'other' — все проходят cap=2
        items = [
            self.FakeItem(f"u{i}", f"other|stem{i}|2026-10-20")
            for i in range(5)
        ]
        result = _cap_by_location(items, max_per_location=2)
        self.assertEqual(len(result), 5)  # 'other' не кэпируется

    def test_other_bucket_explicitly_not_capped(self):
        """'other' бакет — cap не ограничивает его (регрессия)."""
        items = [
            self.FakeItem("u1", "other|stem1|2026-10-20"),
            self.FakeItem("u2", "other|stem2|2026-10-20"),
            self.FakeItem("u3", "other|stem3|2026-10-20"),
        ]
        result = _cap_by_location(items, max_per_location=1)
        # 'other' не кэпируется — все 3 проходят
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()