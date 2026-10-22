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


# ── collector.py: FEED_DEADLINE_SEC и MAX_ITEMS_PER_FEED ──────────────────────


class TestCollectorFeedLimits:
    """Лимиты: wall-clock deadline + MAX_ITEMS_PER_FEED."""

    def test_feed_deadline_defined(self):
        from newsbot.collector import FEED_DEADLINE_SEC
        assert FEED_DEADLINE_SEC == 30

    def test_max_items_per_feed_defined(self):
        from newsbot.collector import MAX_ITEMS_PER_FEED
        assert MAX_ITEMS_PER_FEED == 50

    def test_slow_feed_respects_deadline(self):
        """FEED_DEADLINE_SEC превышен → собирается часть items, не весь фид."""
        import httpx
        from newsbot.collector import FeedCollector, FEED_DEADLINE_SEC

        entry_count = 200
        slow_body = b'''<?xml version="1.0"?><rss><channel>'''
        for i in range(entry_count):
            slow_body += (
                f'<item><title>Slow {i}</title>'
                f'<link>http://ex.com/{i}</link></item>'
            ).encode()
        slow_body += b'</channel></rss>'

        def slow_handler(req: httpx.Request) -> httpx.Response:
            import time
            time.sleep(0.05)   # каждый entry = 50ms → 200 шт. > 30 с дедлайна
            return httpx.Response(200, content=slow_body)

        transport = httpx.MockTransport(slow_handler)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "slow", "url": "http://slow/feed"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True
            # Собрано меньше всех 200 entries (прервались по дедлайну)
            assert 0 < len(result.items) < entry_count, (
                f"Ожидалось <{entry_count} items, получено {len(result.items)}"
            )
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_max_items_per_feed_caps(self):
        """>MAX_ITEMS_PER_FEED entries → собирается ровно MAX_ITEMS_PER_FEED."""
        import httpx
        from newsbot.collector import FeedCollector, MAX_ITEMS_PER_FEED

        big_body = b'''<?xml version="1.0"?><rss><channel>'''
        for i in range(MAX_ITEMS_PER_FEED + 100):
            big_body += (
                f'<item><title>Item {i}</title>'
                f'<link>http://ex.com/{i}</link></item>'
            ).encode()
        big_body += b'</channel></rss>'

        transport, _ = _make_transport(big_body)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "big", "url": "http://big/feed"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True
            assert len(result.items) == MAX_ITEMS_PER_FEED, (
                f"Ожидалось {MAX_ITEMS_PER_FEED} items, получено {len(result.items)}"
            )
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)


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


# -- config.py: m1 - None-секции не роняют .get-цепочки ----------------------


class TestConfigNullSections:
    """YAML-секция без значения (`llm:`) не должна затирать дефолтный dict."""

    def test_null_llm_section_keeps_defaults(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\ntelegram:\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg["llm"], dict)
        assert isinstance(cfg["telegram"], dict)
        assert cfg.get("llm", {}).get("model") == "deepseek-flash"
        assert cfg.get("llm", {}).get("daily_cost_limit") == 0.50

    def test_null_nested_section_keeps_default(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  balance:\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg["llm"]["balance"], dict)
        assert cfg["llm"]["balance"].get("enabled") is True


class TestConfigNonDictSections:
    """
    Minor: секция, заданная скаляром (`llm: deepseek`), не должна затирать
    дефолтный dict — иначе cfg.get("llm", {}).get(...) падает AttributeError.
    """

    def test_scalar_llm_section_falls_back_to_defaults(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm: deepseek\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert isinstance(cfg["llm"], dict)
        assert cfg["llm"]["model"] == "deepseek-flash"
        # Именно эта цепочка падала в pipeline.py.
        assert cfg.get("llm", {}).get("push_threshold") == 7
        assert any("llm" in r.getMessage() for r in caplog.records)

    def test_scalar_telegram_section_falls_back_to_defaults(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("telegram: broken\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg["telegram"], dict)
        assert "token" in cfg["telegram"]

    def test_scalar_balance_section_falls_back_to_defaults(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  balance: true\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg["llm"]["balance"], dict)
        assert cfg["llm"]["balance"].get("enabled") is True


# -- Minor 5 / Major 2(a): гварды config.py -----------------------------------


class TestConfigHardening:
    """sources: null, не-dict корень, валидация числовых llm-полей."""

    def test_sources_null_does_not_crash(self, tmp_path):
        """Minor 5: sources: null → len(None) не должен ронять загрузку."""
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("sources:\nllm:\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, dict)
        assert cfg["llm"]["model"] == "deepseek-flash"

    def test_non_dict_root_falls_back_to_defaults(self, tmp_path):
        """Minor 5: YAML-корень не словарь → defaults + warning, без падения."""
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("- a\n- b\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, dict)
        assert cfg["llm"]["push_threshold"] == 7

    def test_invalid_llm_numeric_falls_back_to_default(self, tmp_path, caplog):
        """Major 2(a): push_threshold: "high" → warning + дефолт (единая точка)."""
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  push_threshold: high\n"
            "  digest_threshold: null\n"
            "  max_retries: 2.5\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["llm"]["push_threshold"] == 7
        assert cfg["llm"]["digest_threshold"] == 5
        assert cfg["llm"]["max_retries"] == 3
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "push_threshold" in messages
        assert "max_retries" in messages

    def test_numeric_string_accepted(self, tmp_path):
        """Числовая строка "9" — валидна (не считается опечаткой)."""
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('llm:\n  push_threshold: "9"\n', encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg["llm"]["push_threshold"] == 9

    def test_digest_cluster_duplicates_in_defaults(self, tmp_path):
        """Minor 5: digest.cluster_duplicates есть в DEFAULT_CONFIG (True)."""
        from newsbot.config import DEFAULT_CONFIG, load_config

        assert DEFAULT_CONFIG["digest"]["cluster_duplicates"] is True

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        cfg = load_config(str(cfg_file))
        assert cfg["digest"]["cluster_duplicates"] is True

    def test_broken_yaml_fails_fast(self, tmp_path, capsys):
        """
        P1-13: синтаксически битый YAML → одна понятная строка в stderr
        и SystemExit(1) ДО setup_logging (не необработанный YAMLError).
        """
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("feeds: [unclosed\n", encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            load_config(str(cfg_file))

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "config.yaml" in err
        assert "синтаксическая ошибка" in err
        assert "Traceback" not in err

    def test_dedup_max_seen_titles_in_defaults(self):
        """dedup.max_seen_titles есть в DEFAULT_CONFIG и равен 2000."""
        from newsbot.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["dedup"]["max_seen_titles"] == 2000


# ── P2: llm-числа — ±inf и отрицательные не проходят ─────────────────────────


class TestConfigLLMNumericRanges:
    """
    P2-фикс: llm.timeout_sec=.inf (валидный YAML float) раньше проходил
    валидацию → OverflowError на каждом LLM-вызове; batch_size/max_tokens=-1
    тоже проходили. Теперь isfinite + range-check → warning + дефолт.
    """

    def test_inf_timeout_falls_back_to_default(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  timeout_sec: .inf\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["llm"]["timeout_sec"] == 60
        assert any("timeout_sec" in r.getMessage() for r in caplog.records)

    def test_negative_nan_inf_values_fall_back(self, tmp_path, caplog):
        import logging
        import math

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  batch_size: -1\n"
            "  max_tokens: -1\n"
            "  temperature: .nan\n"
            "  daily_cost_limit: -.inf\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["llm"]["batch_size"] == 10
        assert cfg["llm"]["max_tokens"] == 4096
        assert cfg["llm"]["temperature"] == 0
        assert cfg["llm"]["daily_cost_limit"] == 0.50
        assert all(
            not isinstance(cfg["llm"][k], float) or math.isfinite(cfg["llm"][k])
            for k in ("temperature", "daily_cost_limit")
        )

    def test_out_of_range_upper_bound_falls_back(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  max_tokens: 999999999\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg["llm"]["max_tokens"] == 4096


# ── P1: telegram.poll_timeout — тип, диапазон, кламп ─────────────────────────


class TestConfigTelegramPollTimeout:
    """
    P1-фикс: poll_timeout влияет на join poller'а при shutdown
    (join = poll_timeout + 5). Значение >30 клампится до 30, невалидное —
    дефолт 25, чтобы бюджет graceful shutdown оставался жёстким.
    """

    def test_valid_value_kept_as_int(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('telegram:\n  poll_timeout: "10"\n', encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg["telegram"]["poll_timeout"] == 10

    def test_too_large_value_is_clamped(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("telegram:\n  poll_timeout: 120\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["telegram"]["poll_timeout"] == 30
        assert any("poll_timeout" in r.getMessage() for r in caplog.records)

    def test_invalid_value_falls_back_to_default(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("telegram:\n  poll_timeout: soon\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["telegram"]["poll_timeout"] == 25
        assert any("poll_timeout" in r.getMessage() for r in caplog.records)

    def test_negative_or_zero_falls_back_to_default(self, tmp_path):
        from newsbot.config import load_config

        for bad in ("0", "-5", "1.5", "true"):
            cfg_file = tmp_path / "config.yaml"
            cfg_file.write_text(
                f"telegram:\n  poll_timeout: {bad}\n", encoding="utf-8"
            )
            cfg = load_config(str(cfg_file))
            assert cfg["telegram"]["poll_timeout"] == 25, bad


# ── poll_interval_minutes: тип/диапазон/кламп (MAJOR: рестарт-луп демона) ────


class TestConfigPollInterval:
    """
    MAJOR-фикс: poll_interval_minutes валидируется при загрузке конфига.

    Раньше строка "15" или null доезжали до демона, где `slept >= interval`
    падал TypeError'ом ВНЕ try → процесс умирал → restart: unless-stopped →
    вечный рестарт-луп.
    """

    def test_string_is_coerced_to_int(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('poll_interval_minutes: "15"\n', encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["poll_interval_minutes"] == 15
        assert isinstance(cfg["poll_interval_minutes"], int)
        assert not [r for r in caplog.records if "poll_interval" in r.getMessage()]

    def test_null_falls_back_to_default_with_warning(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("poll_interval_minutes: null\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["poll_interval_minutes"] == 15
        assert any(
            "poll_interval_minutes" in r.getMessage() for r in caplog.records
        )

    def test_non_numeric_falls_back_to_default(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "poll_interval_minutes: often\n", encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["poll_interval_minutes"] == 15
        assert any(
            "poll_interval_minutes" in r.getMessage() for r in caplog.records
        )

    def test_zero_and_negative_are_clamped_to_one(self, tmp_path):
        from newsbot.config import load_config

        for bad in ("0", "-5", 0, -5):
            cfg_file = tmp_path / "config.yaml"
            cfg_file.write_text(
                f"poll_interval_minutes: {bad}\n", encoding="utf-8"
            )
            cfg = load_config(str(cfg_file))
            assert cfg["poll_interval_minutes"] == 1, bad

    def test_too_large_is_clamped_to_max(self, tmp_path, caplog):
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "poll_interval_minutes: 1500\n", encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["poll_interval_minutes"] == 240
        assert any(
            "poll_interval_minutes" in r.getMessage() for r in caplog.records
        )

    def test_bool_is_rejected(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("poll_interval_minutes: true\n", encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg["poll_interval_minutes"] == 15


# --
