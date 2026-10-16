"""
Фиксы по ревью T3: тесты для config.py, env.py и collector.py.
Запускать: python -m pytest tests/test_config_env_collector.py -v
"""
import pytest
import sys
from pathlib import Path

# ── env.py: m2-фикс — unquoted values with spaces ────────────────────────────


class TestEnvUnquotedWithSpaces:
    """
    m2-фикс (env.py): shlex.split() для unquoted "KEY=a b c" возвращал ['a','b','c'],
    parts[0]='a' — значение обрезалось. Фикс: для unquoted значений
    используем rest.strip() целиком, shlex.split() только для кавыченных.
    """

    def test_unquoted_value_with_spaces(self, tmp_path, monkeypatch):
        """KEY=value with spaces → значение сохраняет пробелы."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=hello world today\n", encoding="utf-8")

        # Переопределяем _set_env_project_root на tmp_path
        from newsbot.env import _set_env_project_root, _cache
        _set_env_project_root(tmp_path)
        _cache.clear() if hasattr(_cache, 'clear') else None

        # Перезагружаем модуль чтобы сбросить _cache через reload
        import importlib, newsbot.env
        _old_cache = newsbot.env._cache
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("TEST_KEY") == "hello world today", \
                f"Expected 'hello world today', got {result.get('TEST_KEY')!r}"
        finally:
            newsbot.env._cache = _old_cache
            newsbot.env._ENV_PATH = None

    def test_quoted_value_with_spaces(self, tmp_path):
        """KEY="hello world" → кавычки снимаются, пробелы сохраняются."""
        env_file = tmp_path / ".env"
        env_file.write_text('TEST_KEY="hello world today"\n', encoding="utf-8")

        from newsbot.env import _set_env_project_root, _cache
        _set_env_project_root(tmp_path)
        import newsbot.env
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("TEST_KEY") == "hello world today", \
                f"Expected 'hello world today', got {result.get('TEST_KEY')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_simple_unquoted_no_spaces(self, tmp_path):
        """KEY=simple → без изменений."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=simplevalue\n", encoding="utf-8")

        from newsbot.env import _set_env_project_root, _cache
        _set_env_project_root(tmp_path)
        import newsbot.env
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("TEST_KEY") == "simplevalue"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None


# ── config.py: DEFAULT_CONFIG содержит новые ключи ───────────────────────────


class TestDefaultConfigKeys:
    """m2-фикс: DEFAULT_CONFIG содержит llm.push_urgent_only и database.max_retries."""

    def test_database_has_max_retries(self):
        from newsbot.config import DEFAULT_CONFIG
        assert "database" in DEFAULT_CONFIG
        assert "max_retries" in DEFAULT_CONFIG["database"]
        assert DEFAULT_CONFIG["database"]["max_retries"] == 8

    def test_llm_has_push_urgent_only(self):
        from newsbot.config import DEFAULT_CONFIG
        assert "llm" in DEFAULT_CONFIG
        assert "push_urgent_only" in DEFAULT_CONFIG["llm"]
        assert DEFAULT_CONFIG["llm"]["push_urgent_only"] is True

    def test_digest_times_correct(self):
        from newsbot.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["digest"]["times"] == ["08:00", "20:00"]


# ── collector.py: лимит размера ответа (M1) ─────────────────────────────────


class TestCollectorMaxFeedBytes:
    """M1-фикс: collector ограничивает размер ответа, защита от gzip-бомбы."""

    def test_max_feed_bytes_defined(self):
        from newsbot.collector import MAX_FEED_BYTES
        assert MAX_FEED_BYTES == 10 * 1024 * 1024, "MAX_FEED_BYTES должен быть 10 MB"

    def test_feed_collector_init(self):
        """FeedCollector инициализируется с правильным конфигом."""
        from newsbot.collector import FeedCollector, FETCH_TIMEOUT, USER_AGENT
        collector = FeedCollector(sources=[{"name": "test", "url": "http://x.ru/rss"}])
        assert collector._client.timeout.read == FETCH_TIMEOUT
        assert "user-agent" in dict(collector._client.headers)
        collector.close()


# ── collector.py: httpx.MockTransport-тесты (happy-path / overflow / gzip) ───


_MIN_RSS = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Test Feed</title>
  <link>http://example.com</link>
  <item>
    <title>Test News Item</title>
    <link>http://example.com/news/1</link>
    <description>A test news item for unit testing.</description>
    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Second News</title>
    <link>http://example.com/news/2</link>
    <description>Another test item.</description>
    <pubDate>Tue, 02 Jan 2024 14:00:00 GMT</pubDate>
  </item>
</channel>
</rss>'''


def _make_transport(rss_body: bytes, content_encoding: str = ""):
    """Возвращает (httpx.MockTransport, int) — транспорт и число вызовов."""
    from unittest.mock import Mock
    import httpx

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        headers = {"content-type": "application/rss+xml; charset=utf-8"}
        if content_encoding:
            headers["content-encoding"] = content_encoding
        return httpx.Response(200, content=rss_body, headers=headers)

    return httpx.MockTransport(handler), lambda: call_count


# ── collector.py: M3 — SSRF-защита (scheme-валидация) ──────────────────────────


class TestCollectorUrlValidation:
    """M3-фикс: collector валидирует URL до HTTP-запроса (только http/https + непустой хост)."""

    def test_validate_url_http_https(self):
        from newsbot.collector import FeedCollector

        c = FeedCollector([{"name": "x", "url": "http://example.com/rss"}])
        assert c._validate_url("http://example.com/rss") is True
        assert c._validate_url("https://example.com/rss") is True
        c.close()

    def test_validate_url_rejects_file_ftp_mailto(self):
        from newsbot.collector import FeedCollector

        c = FeedCollector([{"name": "x", "url": "http://x"}])
        assert c._validate_url("file:///etc/passwd") is False
        assert c._validate_url("ftp://example.com/rss") is False
        assert c._validate_url("mailto://user@example.com") is False
        assert c._validate_url("javascript:alert(1)") is False
        c.close()

    def test_validate_url_rejects_empty_host(self):
        from newsbot.collector import FeedCollector

        c = FeedCollector([{"name": "x", "url": "http://x"}])
        assert c._validate_url("http://") is False
        assert c._validate_url("https://:8080") is False
        assert c._validate_url("http:///path") is False
        assert c._validate_url("https:///") is False
        c.close()

    def test_collect_rejects_bad_scheme(self):
        from newsbot.collector import FeedCollector

        collector = FeedCollector([
            {"name": "bad_feed", "url": "file:///etc/passwd"},
            {"name": "good_feed", "url": "http://example.com/rss"},
        ])
        try:
            results = list(collector.collect())
            assert len(results) == 1
            assert results[0][0] == "good_feed"
        finally:
            collector.close()


# ── env.py: B9 — inline comment (# внутри значения) ────────────────────────────


class TestEnvInlineComment:
    """B9-фикс (rev2): # — комментарий только если пробел перед ним (или начало строки),
    внутри кавычек не трогается."""

    def test_hash_inside_double_quotes_preserved(self, tmp_path):
        """KEY="value#with#hashes" → значение сохраняет все #."""
        env_file = tmp_path / ".env"
        env_file.write_text('SECRET_WITH_HASH="abc#123#xyz"\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("SECRET_WITH_HASH") == "abc#123#xyz", \
                f"Expected 'abc#123#xyz', got {result.get('SECRET_WITH_HASH')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_hash_inside_single_quotes_preserved(self, tmp_path):
        """KEY='x#y#z' → значение сохраняет все #."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET_WITH_HASH='x#y#z'\n", encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("SECRET_WITH_HASH") == "x#y#z", \
                f"Expected 'x#y#z', got {result.get('SECRET_WITH_HASH')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_hash_outside_quotes_with_space_is_comment(self, tmp_path):
        """KEY=value # comment → comment отрезается, KEY="value"."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value # comment\n", encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("KEY") == "value", f"Got {result.get('KEY')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_hash_in_value_with_trailing_comment(self, tmp_path):
        """KEY="a#b" # comment → VALUE="a#b", comment отрезан."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="a#b" # this is a comment\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("KEY") == "a#b", f"Got {result.get('KEY')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None


# ── env.py: M2 — ровно одна пара кавычек ──────────────────────────────────────


class TestEnvExactOneQuotePair:
    """M2-фикс: снимается ровно ОДНА парная кавычка, не все по краям."""

    def test_trailing_quote_not_stripped(self, tmp_path):
        """KEY=VALUE" → VALUE" (не VALUE); VALUE" — валидный secret."""
        env_file = tmp_path / ".env"
        env_file.write_text('API_TOKEN=secret123"\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("API_TOKEN") == 'secret123"', \
                f"Expected 'secret123\"', got {result.get('API_TOKEN')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_leading_quote_not_stripped(self, tmp_path):
        """KEY="VALUE → "VALUE (не VALUE)."""
        env_file = tmp_path / ".env"
        env_file.write_text('API_TOKEN="VALUE\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("API_TOKEN") == '"VALUE', \
                f"Expected '\"VALUE', got {result.get('API_TOKEN')!r}"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_double_quotes_exact_pair_removed(self, tmp_path):
        """KEY="VALUE" → VALUE."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="VALUE"\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("KEY") == "VALUE"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_single_quotes_exact_pair_removed(self, tmp_path):
        """KEY='VALUE' → VALUE."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY='VALUE'\n", encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("KEY") == "VALUE"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None

    def test_double_quoted_with_spaces(self, tmp_path):
        """KEY="hello world" → hello world."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="hello world"\n', encoding="utf-8")

        import newsbot.env
        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file
        try:
            result = newsbot.env.load_env()
            assert result.get("KEY") == "hello world"
        finally:
            newsbot.env._cache = None
            newsbot.env._ENV_PATH = None


# ── collector.py: httpx.MockTransport-тесты (happy-path / overflow / gzip) ───


class TestCollectorMockTransport:
    """httpx.MockTransport: happy-path, overflow, gzip-overflow."""

    def test_happy_path_parses_rss_and_returns_items(self):
        """Валидный RSS → ok=True, items непустой."""
        import httpx
        from newsbot.collector import FeedCollector, MAX_FEED_BYTES

        transport, get_calls = _make_transport(_MIN_RSS)
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "test_feed", "url": "http://test/feed"}])
        collector._client.close()   # закрываем оригинальный
        collector._client = client  # подменяем на мок

        try:
            name, result = next(collector.collect())
            assert result.ok is True, f"expected ok=True, got {result.ok}"
            assert len(result.items) == 2, f"expected 2 items, got {len(result.items)}"
            assert result.items[0].title == "Test News Item"
            assert result.items[1].title == "Second News"
            assert get_calls() == 1, "Должен быть ровно 1 HTTP-запрос"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)  # восстанавливаем

    def test_body_exceeds_max_feed_bytes_returns_ok_false(self):
        """Тело больше MAX_FEED_BYTES → ok=False, items пуст."""
        import httpx
        from newsbot.collector import FeedCollector, MAX_FEED_BYTES

        large_body = b"<rss><channel><item><title>Big</title>"
        large_body += b"X" * (MAX_FEED_BYTES + 1024)
        large_body += b"</item></channel></rss>"

        transport, get_calls = _make_transport(large_body)
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "big_feed", "url": "http://test/big"}])
        collector._client.close()
        collector._client = client

        try:
            name, result = next(collector.collect())
            assert result.ok is False, f"expected ok=False for large feed, got {result.ok}"
            assert result.items == [], f"expected empty items, got {result.items}"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_gzip_decompressed_exceeds_max_feed_bytes_returns_ok_false(self):
        """Gzip-тело, распакованное > MAX_FEED_BYTES → ok=False (считаем распакованное)."""
        import gzip
        import httpx
        from newsbot.collector import FeedCollector, MAX_FEED_BYTES

        decompressed = b"<rss><channel><item><title>GzipTest</title>"
        decompressed += b"Y" * (MAX_FEED_BYTES + 4096)
        decompressed += b"</item></channel></rss>"
        compressed = gzip.compress(decompressed)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                content=compressed,
                headers={
                    "content-type": "application/rss+xml",
                    "content-encoding": "gzip",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "gzip_feed", "url": "http://test/gzip"}])
        collector._client.close()
        collector._client = client

        try:
            name, result = next(collector.collect())
            assert result.ok is False, (
                f"expected ok=False for gzip > MAX_FEED_BYTES, got {result.ok}"
            )
            assert result.items == [], f"expected empty items, got {result.items}"
            assert call_count == 1
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)