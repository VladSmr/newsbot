"""
test_urlnorm.py — тесты нормализации URL для дедупликации.

Покрывает:
  - lowercase scheme и host
  - удаление fragment
  - удаление трекинговых параметров (utm_*, traffic_source, fbclid, …)
  - сортировка query-параметров
  - удаление завершающего слэша (кроме корня)
  - защита от невалидных / не-HTTP(S) URL
  - идемпотентность
  - None / пустая строка
"""
import pytest

from newsbot.urlnorm import normalize_url, TRACKING_PARAMS


class TestNormalizeUrl:
    """Базовые случаи нормализации."""

    def test_lowercase_scheme_host(self):
        """Scheme и host приводятся к lowercase."""
        assert normalize_url("HTTPS://WWW.EXAMPLE.COM/path") == "https://www.example.com/path"
        assert normalize_url("HTTP://API.TEST.ORG/endpoint") == "http://api.test.org/endpoint"

    def test_fragment_removed(self):
        """Fragment (#...) удаляется."""
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"
        assert normalize_url("https://example.com/a#") == "https://example.com/a"
        assert normalize_url("https://example.com/#top") == "https://example.com/"

    def test_trailing_slash_stripped(self):
        """Завершающий слэш убирается для не-корневого пути."""
        assert normalize_url("https://example.com/path/") == "https://example.com/path"
        assert normalize_url("https://example.com/a/b/c/") == "https://example.com/a/b/c"
        # Корень "/" сохраняется
        assert normalize_url("https://example.com/") == "https://example.com/"

    def test_query_params_sorted(self):
        """Query-параметры сортируются по имени для детерминированности."""
        result = normalize_url("https://example.com/page?z=1&a=2&m=3")
        assert result == "https://example.com/page?a=2&m=3&z=1"


class TestTrackingParams:
    """Удаление трекинговых параметров."""

    def test_utm_removed(self):
        """Все utm_* параметры удаляются."""
        url = (
            "https://example.com/article?"
            "utm_source=rss&utm_medium=email&utm_campaign=weekly"
            "&utm_content=link&id=123"
        )
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "utm_campaign" not in result
        assert "id=123" in result

    def test_traffic_source_removed(self):
        """Параметр traffic_source удаляется (Al Jazeera и др. новостные фиды)."""
        url = "https://www.aljazeera.com/news/2025/10/23/article?traffic_source=rss&id=42"
        result = normalize_url(url)
        assert "traffic_source" not in result
        assert "id=42" in result

    def test_fbclid_gclid_mc_removed(self):
        """Рекламные ID и email-трекеры удаляются."""
        url = (
            "https://example.com/page?"
            "fbclid=abc123&gclid=xyz&mc_cid=mid&mc_eid=eid&"
            "referrer=newsletter&id=99"
        )
        result = normalize_url(url)
        assert "fbclid" not in result
        assert "gclid" not in result
        assert "mc_cid" not in result
        assert "mc_eid" not in result
        assert "referrer=" not in result
        assert "id=99" in result

    def test_spam_referrer_removed(self):
        """Параметры spm, referrer удаляются."""
        url = "https://example.com/p?spm=alibaba&referrer=email&id=5"
        result = normalize_url(url)
        assert "spm" not in result
        assert "referrer" not in result
        assert "id=5" in result

    def test_utm_with_legit_params_kept(self):
        """Легитимные параметры сохраняются рядом с utm."""
        url = "https://example.com/view?p=12345&category=tech&ref=section&s=menu"
        result = normalize_url(url)
        assert "p=12345" in result
        assert "category=tech" in result
        # ref, s — НЕ в TRACKING_PARAMS, сохраняются
        assert "ref=section" in result
        assert "s=menu" in result
        assert "utm_source" not in result

    def test_empty_tracking_params_removed(self):
        """Пустые трекинговые параметры (utm_source=) тоже удаляются."""
        url = "https://example.com/p?utm_source=&fbclid=&id=1"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "fbclid" not in result
        assert "id=1" in result

    def test_empty_non_tracking_params_kept(self):
        """Пустые НЕ-трекинговые параметры (key=) остаются."""
        url = "https://example.com/p?empty=&id=1"
        result = normalize_url(url)
        assert "empty=" in result  # empty value — param name is not tracking
        assert "id=1" in result


class TestUtmCollapsing:
    """Варианты с utm/без utm склеиваются в один нормализованный ключ."""

    def test_utm_param_order_different(self):
        """URL с utm-параметрами в разном порядке нормализуются одинаково."""
        u1 = "https://example.com/p?utm_source=a&utm_medium=b&p=1"
        u2 = "https://example.com/p?p=1&utm_medium=b&utm_source=a"
        assert normalize_url(u1) == normalize_url(u2)

    def test_same_url_with_and_without_utm(self):
        """Статья с utm и без — один и тот же нормализованный ключ."""
        base = "https://www.bbc.com/news/technology-12345"
        with_utm = f"{base}?utm_source=rss&utm_medium=email"
        assert normalize_url(with_utm) == normalize_url(base)

    def test_traffic_source_vs_bare(self):
        """Al Jazeera / LiveMint: traffic_source=rss vs bare URL."""
        base = "https://www.aljazeera.com/news/politics/article"
        with_tsource = f"{base}?traffic_source=rss"
        assert normalize_url(with_tsource) == normalize_url(base)

    def test_fragment_ignored(self):
        """URL с fragment и без — одинаковый ключ."""
        base = "https://example.com/article"
        with_frag = f"{base}#comments"
        assert normalize_url(with_frag) == normalize_url(base)


class TestEdgeCases:
    """Edge cases и защита."""

    def test_none_input(self):
        """None на входе — None на выходе."""
        assert normalize_url(None) is None

    def test_empty_string(self):
        """Пустая строка — на выходе пустая строка."""
        assert normalize_url("") == ""

    def test_whitespace_only(self):
        """Whitespace — возвращается как есть."""
        assert normalize_url("   ") == "   "

    def test_non_http_scheme_preserved(self):
        """Не-http(s) схемы возвращаются без изменений (защита)."""
        # urlparse от mailto: даёт scheme=mailto, netloc=user,
        # и мы возвращаем оригинальный URL (не-HTTP(S) guard).
        assert normalize_url("mailto:user@example.com") == "mailto:user@example.com"
        assert normalize_url("file:///path/to/file") == "file:///path/to/file"
        assert normalize_url("tel:+1234567890") == "tel:+1234567890"

    def test_invalid_url_preserved(self):
        """Невалидный URL возвращается как есть, без исключений."""
        bad_urls = [
            "http://",
            "://example.com",
            "not a url",
            "https://",
            "http://[invalid",
        ]
        for url in bad_urls:
            result = normalize_url(url)
            # Должен вернуть что-то, не упасть
            assert result is not None

    def test_port_preserved(self):
        """Порт сохраняется."""
        assert normalize_url("https://example.com:8443/path") == "https://example.com:8443/path"

    def test_userinfo_preserved(self):
        """Userinfo (если есть) сохраняется."""
        assert normalize_url("https://user:pass@example.com/path") == "https://user:pass@example.com/path"

    def test_empty_query_after_removal(self):
        """Все query-параметры удалены — ? не остаётся."""
        url = "https://example.com/p?utm_source=rss"
        assert normalize_url(url) == "https://example.com/p"
        assert "?" not in normalize_url(url)

    def test_only_question_mark(self):
        """URL с голым ? без параметров — без ? на выходе."""
        assert normalize_url("https://example.com/p?") == "https://example.com/p"

    def test_unicode_in_path(self):
        """Unicode в path сохраняется (idna-кодирование не применяем)."""
        result = normalize_url("https://example.com/путь?lang=ru")
        # urlparse оставляет path как есть для не-IDN хостов
        assert "example.com" in result
        assert "путь" in result or "path" in result  # зависит от парсера


class TestIdempotency:
    """Нормализация идемпотентна: normalize(normalize(url)) == normalize(url)."""

    @pytest.mark.parametrize("url", [
        "https://EXAMPLE.COM/p?utm_source=rss&p=1",
        "https://www.aljazeera.com/news?traffic_source=rss",
        "https://host.com/page/?utm_source=a&ref=b#section",
        "https://a.com/",
        "HTTP://API.ORG/v1?KEY=secret",
    ])
    def test_idempotent(self, url):
        first = normalize_url(url)
        second = normalize_url(first)
        assert first == second, f"Idempotency failed for {url!r}: {first!r} → {second!r}"

    def test_idempotent_on_tracked_url(self):
        """Многократное применение не меняет результат."""
        url = "https://EXAMPLE.COM/p?utm_source=feed&fbclid=abc123&p=42#top/"
        for _ in range(5):
            url = normalize_url(url)
        assert url == "https://example.com/p?p=42"


class TestUrlnormNitFixes:
    """
    NIT-фиксы urlnorm:
      1. Путь НЕ lowercased (case-sensitive CMS → 404 по ссылке из дайджеста).
      2. Пустой path → "/".
      3. Query сортируется по (имя, значение).
    """

    def test_path_case_preserved(self):
        """Path сохраняет регистр (case-sensitive)."""
        # HOST uppercase, path MixedCase → host lowercase, path MixedCase
        assert normalize_url("HTTPS://EXAMPLE.COM/Some/Path") == "https://example.com/Some/Path"
        assert normalize_url("HTTP://API.TEST.ORG/v1/Users/Profile") == "http://api.test.org/v1/Users/Profile"

    def test_empty_path_normalized_to_slash(self):
        """Пустой path → "/" (https://x.com и https://x.com/ склеиваются)."""
        assert normalize_url("https://x.com") == "https://x.com/"
        assert normalize_url("https://x.com?a=1") == "https://x.com/?a=1"
        assert normalize_url("https://x.com/") == "https://x.com/"

    def test_x_com_with_without_slash_dedup(self):
        """x.com и x.com/ — дубликаты после нормализации."""
        assert normalize_url("https://x.com") == normalize_url("https://x.com/")

    def test_query_params_sorted_by_name_and_value(self):
        """Query-параметры сортируются по (имя, значение) для детерминированности."""
        # a=1&a=2 vs a=2&a=1 → оба дают a=1&a=2
        r1 = normalize_url("https://example.com/p?a=1&a=2")
        r2 = normalize_url("https://example.com/p?a=2&a=1")
        assert r1 == r2, f"{r1} != {r2}"

        # a=1&b=2 vs b=2&a=1 → оба дают a=1&b=2
        r3 = normalize_url("https://example.com/p?a=1&b=2")
        r4 = normalize_url("https://example.com/p?b=2&a=1")
        assert r3 == r4, f"{r3} != {r4}"

    def test_multi_value_sort_stable(self):
        """Мультизначения с разными значениями сортируются стабильно."""
        r = normalize_url("https://example.com/p?a=1&a=2&a=3")
        # Должен быть: ?a=1&a=2&a=3 (не перемешан)
        assert r == "https://example.com/p?a=1&a=2&a=3", f"Got: {r}"