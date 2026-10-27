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

    def test_slow_feed_respects_deadline(self, caplog):
        """
        FEED_DEADLINE_SEC превышен → собирается часть items, не весь фид.

        MockTransport отвечает мгновенно. collector.py проверяет
        FEED_DEADLINE_SEC в entries-loop (line 406: elapsed > FEED_DEADLINE_SEC).
        deadline срабатывает только если истекло >30 с. Имитируем это патчем
        collector.time.monotonic: возвращаем 1.0 для HTTP-фазы (elapsed≈0<30),
        FEED_DEADLINE_SEC+1 для entries-loop (elapsed=31>30 → deadline превышен).
        """
        import logging
        import httpx
        from unittest import mock
        import newsbot.collector as col
        from newsbot.collector import FeedCollector, FEED_DEADLINE_SEC

        _calls = [0]

        def fake_monotonic():
            """
            Deterministic fake: returns the current call index as fake time.
            collector.py sets deadline = feed_start + FEED_DEADLINE_SEC (= 30).
            The entry-loop check (line 406) fires when elapsed > 30.
            We return the raw call counter n, so after n calls elapsed = n.
            With 31 items, the entry loop alone makes 31 checks (n=0..30),
            which is enough for the deadline to fire during entry ~30.
            """
            n = _calls[0]
            _calls[0] = n + 1
            return float(n)

        # Build a feed with 31 items so the entry loop makes enough
        # monotonic() calls to exceed the 30-second deadline.
        body = b'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        for i in range(31):
            body += (f'<item><title>Slow {i}</title>'
                     f'<link>http://ex.com/{i}</link></item>').encode()
        body += b'</channel></rss>'
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))

        # collector.py имеет 'import time'; коллектор вызывает time.monotonic.
        # Патчим модульный атрибут напрямую (через тот же module object).
        with mock.patch.object(col.time, 'monotonic', fake_monotonic):
            collector = FeedCollector([{"name": "slow", "url": "http://slow/feed"}])
            collector._client.close()
            collector._client = httpx.Client(transport=transport)

            with caplog.at_level(logging.WARNING, logger="newsbot.collector"):
                _, result = next(collector.collect())

            assert result.ok is True
            # After 31 monotonic() calls (from 31-item entry loop + HTTP phase),
            # deadline fires: elapsed > FEED_DEADLINE_SEC=30 → fewer than 31 items
            assert len(result.items) < 31, (
                f"Expected fewer than 31 items (deadline exceeded), got {len(result.items)}"
            )
            # WARNING о превышении дедлайна
            assert any("FEED_DEADLINE_SEC" in r.getMessage() for r in caplog.records), \
                f"Ожидался WARNING о FEED_DEADLINE_SEC, получено: {[r.getMessage() for r in caplog.records]}"

            collector._client.close()
        # restore happens automatically via mock.patch

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


class TestConfigPricingCacheHit:
    """Major #12-фикс: pricing.*.cache_hit присутствует и валидируется."""

    def test_pricing_cache_hit_in_defaults(self):
        """DEFAULT_CONFIG содержит cache_hit в pricing."""
        from newsbot.config import DEFAULT_CONFIG

        flash_pricing = DEFAULT_CONFIG["llm"]["pricing"].get("deepseek-flash", {})
        assert "cache_hit" in flash_pricing, (
            f"cache_hit должен быть в pricing.deepseek-flash: {flash_pricing}"
        )
        assert flash_pricing["cache_hit"] == 0.003, (
            f"cache_hit deepseek-flash = 0.003: got {flash_pricing['cache_hit']}"
        )

    def test_cache_hit_from_yaml_loaded(self, tmp_path):
        """pricing.cache_hit загружается из YAML."""
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  pricing:\n"
            "    my-model:\n"
            "      input: 0.10\n"
            "      output: 0.40\n"
            "      cache_hit: 0.002\n",
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_file))
        pricing = cfg["llm"]["pricing"]["my-model"]
        assert pricing["cache_hit"] == 0.002, f"cache_hit должен быть 0.002: {pricing}"

    def test_invalid_cache_hit_falls_back_to_default(self, tmp_path, caplog):
        """cache_hit: "bad" → warning + дефолт."""
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  pricing:\n"
            "    deepseek-flash:\n"
            "      input: 0.15\n"
            "      output: 0.60\n"
            "      cache_hit: not-a-number\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        pricing = cfg["llm"]["pricing"]["deepseek-flash"]
        assert pricing["cache_hit"] == 0.003, (
            f"cache_hit должен fallback в 0.003: {pricing}"
        )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "cache_hit" in messages or "cache" in messages, (
            f"Expected warning about cache_hit, got: {messages}"
        )

    def test_negative_cache_hit_falls_back(self, tmp_path, caplog):
        """cache_hit: -0.001 → warning + дефолт."""
        import logging

        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  pricing:\n"
            "    deepseek-flash:\n"
            "      cache_hit: -0.001\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        pricing = cfg["llm"]["pricing"]["deepseek-flash"]
        assert pricing["cache_hit"] == 0.003, (
            f"negative cache_hit должен fallback в 0.003: {pricing}"
        )


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
        assert cfg["llm"]["max_tokens"] == 4608
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
        assert cfg["llm"]["max_tokens"] == 4608


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


# ── Task 1: NameError в logger.warning (feed_url → url) + defense-in-depth ──


class TestCollectorInvalidLinkIsolation:
    """
    Task 1: entry с невалидной ссылкой (javascript:, ftp:, пустой host)
    не роняет весь цикл сбора. Фиксим: feed_url → url.
    """

    def _rss_with_invalid_entry(self, invalid_link: str, valid_link: str = None):
        """RSS-фид: один entry с invalid_link, один с valid_link."""
        if valid_link is None:
            valid_link = "https://example.com/news/1"
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel>'
            '<item>'
            f'<title>Bad Link Item</title>'
            f'<link>{invalid_link}</link>'
            '<description>A news item with a bad link.</description>'
            '</item>'
            '<item>'
            '<title>Good Link Item</title>'
            f'<link>{valid_link}</link>'
            '<description>Normal news item.</description>'
            '</item>'
            '</channel></rss>'
        )
        return body.encode()

    def _rss_with_multiple_feeds(self, bad_url: str, good_url: str):
        """RSS-фид для второго источника (good)."""
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel>'
            '<item>'
            '<title>Good Feed Item</title>'
            f'<link>{good_url}</link>'
            '<description>Item from the second feed.</description>'
            '</item>'
            '</channel></rss>'
        )
        return body.encode()

    def test_invalid_link_javascript_no_nameerror(self):
        """Entry с javascript: — не падает с NameError, остальные entries обработаны."""
        import httpx
        from newsbot.collector import FeedCollector

        body = self._rss_with_invalid_entry("javascript:alert(1)")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "test_feed", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            name, result = next(collector.collect())
            # НЕ NameError → collect завершился
            assert result.ok is True, f"expected ok=True, got {result.ok}"
            # Валидный entry обработан
            good_items = [it for it in result.items if "Good Link" in it.title]
            assert len(good_items) == 1, (
                f"Expected 1 good item, got {len(good_items)}. "
                f"Items: {[it.title for it in result.items]}"
            )
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_invalid_link_ftp_no_nameerror(self):
        """Entry с ftp: — не падает, остальные entries обработаны."""
        import httpx
        from newsbot.collector import FeedCollector

        body = self._rss_with_invalid_entry("ftp://example.com/file")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "ftp_feed", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            name, result = next(collector.collect())
            assert result.ok is True
            good_items = [it for it in result.items if "Good Link" in it.title]
            assert len(good_items) == 1
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_invalid_link_empty_host_no_nameerror(self):
        """Entry с пустым хостом — не падает, остальные entries обработаны."""
        import httpx
        from newsbot.collector import FeedCollector

        body = self._rss_with_invalid_entry("https://")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport, follow_redirects=True)
        collector = FeedCollector([{"name": "bad_host_feed", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            name, result = next(collector.collect())
            assert result.ok is True
            good_items = [it for it in result.items if "Good Link" in it.title]
            assert len(good_items) == 1
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_one_bad_feed_does_not_break_collect_cycle(self):
        """
        Task 1б: defense-in-depth. Один фид с exception в _fetch_feed
        не прерывает обработку других фидов.
        """
        import httpx
        from newsbot.collector import FeedCollector, FeedResult
        from unittest import mock

        # Мокаем _fetch_feed напрямую: bad1/bad2 бросают, good — штатно.
        def fake_fetch(name, url, lang=None):
            if name in ("bad_feed_1", "bad_feed_2"):
                raise RuntimeError("Simulated feed failure")
            # Штатный результат
            body = self._rss_with_multiple_feeds(
                "https://example.com/good1", "https://example.com/good2"
            )
            transport = httpx.MockTransport(
                lambda req: httpx.Response(200, content=body)
            )
            client = httpx.Client(transport=transport)
            tmp_collector = FeedCollector([{"name": name, "url": url}])
            tmp_collector._client.close()
            tmp_collector._client = client
            try:
                _, result = next(tmp_collector.collect())
            finally:
                client.close()
                tmp_collector._client = httpx.Client(timeout=15)
            return result

        collector = FeedCollector([
            {"name": "bad_feed_1", "url": "http://test/bad1"},
            {"name": "good_feed", "url": "http://test/good"},
            {"name": "bad_feed_2", "url": "http://test/bad2"},
        ])

        with mock.patch.object(collector, "_fetch_feed", fake_fetch):
            results = list(collector.collect())

        # 3 результата: ok=False для bad, ok=True для good
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        bad1 = next(r for r in results if r[0] == "bad_feed_1")
        good = next(r for r in results if r[0] == "good_feed")
        bad2 = next(r for r in results if r[0] == "bad_feed_2")

        assert bad1[1].ok is False, "bad_feed_1 should have ok=False"
        assert good[1].ok is True, "good_feed should have ok=True"
        assert len(good[1].items) == 1, f"Expected 1 item, got {len(good[1].items)}"
        assert bad2[1].ok is False, "bad_feed_2 should have ok=False"

        collector.close()


# --
# ── Major-1: broken <link> (ValueError from urljoin) — entire feed NOT lost ──


class TestCollectorBrokenLinkIsolation:
    """
    Major-1-фикс: urljoin на битом <link> бросает ValueError из urlsplit.
    Раньше: исключение вылетало из _fetch_feed в collect(), терялся весь фид.
    Теперь: try/except начинается ПЕРЕД urljoin, ValueError ловится,
    entry пропускается с warning, остальные entries обрабатываются.
    """

    def test_broken_urljoin_does_not_lose_entire_feed(self, caplog):
        """Фид из 3 entries, у одного битый <link> → 2 собраны, warning есть."""
        import logging
        import httpx
        from newsbot.collector import FeedCollector

        # RSS: entry1 (ok), entry2 (broken link → ValueError in urljoin),
        # entry3 (ok).  Broken link: urljoin("http://feed/", "http://[::1")
        # → ValueError: netloc incorrectly formatted with IP literal.
        body = (
            b'<?xml version="1.0"?><rss version="2.0"><channel>'
            b'<item><title>Item OK 1</title><link>http://example.com/news/1</link></item>'
            b'<item><title>Item BROKEN</title><link>http://[::1</link></item>'
            b'<item><title>Item OK 2</title><link>http://example.com/news/2</link></item>'
            b'</channel></rss>'
        )
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "test", "url": "http://feed/rss"}])
        collector._client.close()
        collector._client = client

        try:
            with caplog.at_level(logging.WARNING, logger="newsbot.collector"):
                _, result = next(collector.collect())

            # Фид обработан, не потерян
            assert result.ok is True, f"expected ok=True, got {result.ok}"
            # 2 из 3 entries собраны
            assert len(result.items) == 2, (
                f"Expected 2 items (broken skipped), got {len(result.items)}: "
                f"{[it.title for it in result.items]}"
            )
            titles = {it.title for it in result.items}
            assert "Item OK 1" in titles
            assert "Item OK 2" in titles
            # Warning о пропущенном entry
            assert any("пропущен entry" in r.getMessage() for r in caplog.records), (
                f"Expected warning about skipped entry, got: "
                f"{[r.getMessage() for r in caplog.records]}"
            )
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_broken_link_urljoin_valueerror_caught(self):
        """ValueError из urljoin именно ловится (не вылетает из collect())."""
        import httpx
        from newsbot.collector import FeedCollector

        # urljoin("http://feed/", "http://[::1") → ValueError: netloc incorrectly
        # formatted with IP literal (Python 3.8).
        body = (
            b'<?xml version="1.0"?><rss version="2.0"><channel>'
            b'<item><title>Only This</title><link>http://[::1</link></item>'
            b'</channel></rss>'
        )
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "broken", "url": "http://feed/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            # НЕ исключение — collect() вернул результат
            assert result.ok is True, "broken link feed should return ok=True"
            assert result.items == [], "broken entry should be skipped, items empty"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)


# ── Minor-1: retry logic — timeout + 502/503/504 ───────────────────────────────


class TestCollectorRetry:
    """
    Minor-1-фикс: один retry с backoff для идемпотентного GET.
    Retry только на таймаутах и 502/503/504, бюджет не превышает
    FEED_DEADLINE_SEC. НЕ retry'ятся: 404, 403, 500, etc.
    """

    def test_timeout_then_retry_succeeds(self):
        """Таймаут первого запроса → retry → успех."""
        import httpx
        import time
        from newsbot.collector import FeedCollector

        body = (
            b'<?xml version="1.0"?><rss version="2.0"><channel>'
            b'<item><title>Retry Success</title>'
            b'<link>http://example.com/news/1</link></item>'
            b'</channel></rss>'
        )

        attempts = []

        def handler(req: httpx.Request) -> httpx.Response:
            attempts.append(time.time())
            if len(attempts) == 1:
                # Первый запрос — имитируем таймаут
                raise httpx.TimeoutException("simulated timeout")
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "retry_feed", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True, f"expected ok=True after retry, got {result.ok}"
            assert len(result.items) == 1
            assert len(attempts) == 2, f"Expected 2 attempts, got {len(attempts)}"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_502_then_retry_succeeds(self):
        """502 на первом запросе → retry → успех."""
        import httpx
        from newsbot.collector import FeedCollector

        body = (
            b'<?xml version="1.0"?><rss version="2.0"><channel>'
            b'<item><title>Retry 502</title>'
            b'<link>http://example.com/news/1</link></item>'
            b'</channel></rss>'
        )

        attempt_count = [0]

        def handler(req: httpx.Request) -> httpx.Response:
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return httpx.Response(502, content=b"Bad Gateway")
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "retry_502", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True, f"expected ok=True after 502-retry, got {result.ok}"
            assert attempt_count[0] == 2
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_404_not_retried(self):
        """404 — не retry'ится, сразу ok=False."""
        import httpx
        from newsbot.collector import FeedCollector

        attempt_count = [0]

        def handler(req: httpx.Request) -> httpx.Response:
            attempt_count[0] += 1
            return httpx.Response(404, content=b"Not Found")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "not_found", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is False, f"expected ok=False for 404, got {result.ok}"
            assert attempt_count[0] == 1, "404 should NOT be retried"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_retry_exhausted_timeout_returns_ok_false(self):
        """Оба таймаута неуспешны → ok=False."""
        import httpx
        from newsbot.collector import FeedCollector

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("always timeout")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "timeout_feed", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is False
            assert result.items == []
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)


# ── Minor-3: tags caps ([:50], each [:64]) ────────────────────────────────────


class TestCollectorTagsCap:
    """Minor-3-фикс: tags_list[:50], каждый term[:64]."""

    def test_tags_capped_to_50(self):
        """Более 50 тегов — обрезаются до 50."""
        import httpx
        import json
        from newsbot.collector import FeedCollector

        items = "".join(
            f"<category>term_{i}</category>"
            for i in range(60)
        )
        body = (
            f'<?xml version="1.0"?><rss version="2.0"><channel>'
            f'<item><title>Tag Test</title>'
            f'<link>http://example.com/news/1</link>'
            f'{items}'
            f'</item></channel></rss>'
        ).encode()

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "tags_test", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True
            assert len(result.items) == 1
            tags = json.loads(result.items[0].tags)
            assert len(tags) == 50, f"Expected 50 tags, got {len(tags)}"
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)

    def test_each_tag_capped_to_64(self):
        """Каждый тег длиннее 64 симв. — обрезается."""
        import httpx
        import json
        from newsbot.collector import FeedCollector

        long_tag = "X" * 100
        body = (
            f'<?xml version="1.0"?><rss version="2.0"><channel>'
            f'<item><title>Long Tag Test</title>'
            f'<link>http://example.com/news/1</link>'
            f'<category>{long_tag}</category>'
            f'</item></channel></rss>'
        ).encode()

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=body)
        )
        client = httpx.Client(transport=transport)
        collector = FeedCollector([{"name": "long_tag_test", "url": "http://test/rss"}])
        collector._client.close()
        collector._client = client

        try:
            _, result = next(collector.collect())
            assert result.ok is True
            assert len(result.items) == 1
            tags = json.loads(result.items[0].tags)
            assert len(tags) == 1
            assert len(tags[0]) == 64, f"Expected 64 chars, got {len(tags[0])}"
            assert tags[0] == "X" * 64
        finally:
            client.close()
            collector._client = httpx.Client(timeout=15)


# ── Major-2: database.retry_threshold_sec string coercion ────────────────────


class TestConfigDatabaseRetryThreshold:
    """Major-2-фикс: retry_threshold_sec как строка "60" → приводится к int."""

    def test_retry_threshold_string_coerced(self, tmp_path):
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('database:\n  retry_threshold_sec: "60"\n', encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg["database"]["retry_threshold_sec"] == 60
        assert isinstance(cfg["database"]["retry_threshold_sec"], int)

    def test_retry_threshold_abc_falls_back(self, tmp_path, caplog):
        import logging
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("database:\n  retry_threshold_sec: abc\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            cfg = load_config(str(cfg_file))

        assert cfg["database"]["retry_threshold_sec"] == 60
        assert any("retry_threshold_sec" in r.getMessage() for r in caplog.records)

    def test_database_numeric_fields_coerced(self, tmp_path):
        """max_retries, recovery_new_limit, window_days, max_per_location — int."""
        from newsbot.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "database:\n"
            "  max_retries: '5'\n"
            "  recovery_new_limit: '300'\n"
            "dedup:\n"
            "  window_days: '2'\n"
            "  max_seen_titles: '1000'\n"
            "  llm_recent_limit: '15'\n"
            "digest:\n"
            "  max_per_location: '3'\n",
            encoding="utf-8",
        )

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg["database"]["max_retries"], int)
        assert isinstance(cfg["database"]["recovery_new_limit"], int)
        assert isinstance(cfg["dedup"]["window_days"], int)
        assert isinstance(cfg["dedup"]["max_seen_titles"], int)
        assert isinstance(cfg["dedup"]["llm_recent_limit"], int)
        assert isinstance(cfg["digest"]["max_per_location"], int)
        assert cfg["database"]["max_retries"] == 5
        assert cfg["dedup"]["window_days"] == 2


# ── Major-3: logging.level validation ─────────────────────────────────────────


class TestSetupLoggingLevelValidation:
    """Major-3-фикс: level: null/int/INF0 → warning + INFO, не AttributeError."""

    def test_level_null_no_crash_with_warning(self, tmp_path, capsys):
        """level: null → warning + INFO (не AttributeError)."""
        import logging
        from newsbot.log import setup_logging

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            # Не падаем с AttributeError
            setup_logging({"level": None})
            # Уровень установлен в INFO (дефолт)
            assert root.level == logging.INFO
            # Warning записан в stderr
            captured = capsys.readouterr().err
            assert "level" in captured.lower() or "INFO" in captured
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
            root.handlers = saved_handlers

    def test_level_int_no_crash_with_warning(self, tmp_path, capsys):
        """level: 123 (int) → warning + INFO."""
        import logging
        from newsbot.log import setup_logging

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging({"level": 123})
            assert root.level == logging.INFO
            captured = capsys.readouterr().err
            assert "level" in captured.lower() or "INFO" in captured
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
            root.handlers = saved_handlers

    def test_level_typo_inf0_with_warning(self, tmp_path, capsys):
        """level: "INF0" (опечатка) → warning + INFO (не тихий fallback)."""
        import logging
        from newsbot.log import setup_logging

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging({"level": "INF0"})
            assert root.level == logging.INFO
            captured = capsys.readouterr().err
            assert "INF0" in captured, f"Expected warning about INF0 in stderr: {captured}"
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
            root.handlers = saved_handlers

    def test_level_debug_lowercase_works(self, tmp_path):
        """level: "debug" (нижний регистр) → DEBUG."""
        import logging
        from newsbot.log import setup_logging

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging({"level": "debug"})
            assert root.level == logging.DEBUG
        finally:
            for h in root.handlers[:]:
                root.removeHandler(h)
            root.handlers = saved_handlers


# ── Minor-4: userinfo in URL sanitization ─────────────────────────────────────


class TestSanitizerUserinfo:
    """Minor-4-фикс: userinfo (scheme://user:pass@host) маскируется."""

    def test_userinfo_in_url_masked(self):
        """scheme://user:pass@host → scheme://***@host."""
        from newsbot.log import _sanitize

        result = _sanitize("URL: http://admin:secret123@example.com/api")
        assert "admin" not in result
        assert "secret123" not in result
        assert "***" in result
        assert "example.com" in result

    def test_userinfo_without_port_masked(self):
        """<URL_WITH_CREDS_5> → http://***@host."""
        from newsbot.log import _sanitize

        result = _sanitize("Feed: https://bot111:Tok22222222@example.com/rss")
        assert "bot111" not in result
        assert "Tok22222222" not in result
        assert "***" in result

    def test_telegram_token_still_masked(self):
        """Telegram-токен вырезается (регресс-тест)."""
        from newsbot.log import _sanitize

        token = "bot123456:ABCDefgh1234567890abcdefgh"
        result = _sanitize(f"Token: {token}")
        assert "ABCDefgh1234567890abcdefgh" not in result
        assert "***REDACTED***" in result

    def test_deepseek_key_still_masked(self):
        """DeepSeek-api-key вырезается (регресс-тест)."""
        from newsbot.log import _sanitize

        key = "sk-" + "a" * 30
        result = _sanitize(f"Key: {key}")
        assert key not in result
        assert "***REDACTED***" in result


# ── Minor-7: atomic env write ─────────────────────────────────────────────────


class TestEnvAtomicWrite:
    """Minor-7-фикс: _write_env пишет в .env.tmp + os.replace()."""

    def test_write_env_creates_file(self, tmp_path):
        import newsbot.env

        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = tmp_path / ".env"

        try:
            newsbot.env._write_env("TEST_KEY", "test_value")
            content = (tmp_path / ".env").read_text(encoding="utf-8")
            assert "TEST_KEY=test_value" in content
            assert (tmp_path / ".env.tmp").exists() is False, ".env.tmp should not remain"
        finally:
            newsbot.env._ENV_PATH = None
            newsbot.env._cache = None

    def test_write_env_overwrites_existing_key(self, tmp_path):
        import newsbot.env

        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=old_value\n", encoding="utf-8")

        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = env_file

        try:
            newsbot.env._write_env("EXISTING", "new_value")
            content = env_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            matching = [l for l in lines if l.startswith("EXISTING=")]
            assert len(matching) == 1, f"Expected 1 EXISTING line, got {matching}"
            assert "new_value" in matching[0]
            assert "old_value" not in content
        finally:
            newsbot.env._ENV_PATH = None
            newsbot.env._cache = None

    def test_tmp_file_cleaned_up(self, tmp_path):
        """После успешной записи .env.tmp не остаётся."""
        import newsbot.env

        newsbot.env._set_env_project_root(tmp_path)
        newsbot.env._cache = None
        newsbot.env._ENV_PATH = tmp_path / ".env"

        try:
            newsbot.env._write_env("CLEANUP", "check")
            assert not (tmp_path / ".env.tmp").exists()
        finally:
            newsbot.env._ENV_PATH = None
            newsbot.env._cache = None


# ── Minor-8: naive datetime → ValueError ──────────────────────────────────────


class TestOffpeakNaiveDatetime:
    """Minor-8-фикс: naive datetime → ValueError (явный, не молчаливый)."""

    def test_is_deepseek_peak_raises_on_naive(self):
        """is_deepseek_peak с naive datetime → ValueError."""
        from datetime import datetime
        from newsbot.offpeak import is_deepseek_peak

        naive_dt = datetime(2026, 1, 5, 7, 30, 0)  # пн 07:30 без tz
        with pytest.raises(ValueError, match="tz-aware"):
            is_deepseek_peak(naive_dt)

    def test_is_llm_allowed_raises_on_naive(self):
        """is_llm_allowed_now с naive datetime → ValueError."""
        from datetime import datetime
        from newsbot.offpeak import is_llm_allowed_now

        naive_dt = datetime(2026, 1, 5, 12, 0, 0)  # пн 12:00 без tz
        with pytest.raises(ValueError, match="tz-aware"):
            is_llm_allowed_now(off_peak_only=True, now_utc=naive_dt)

    def test_tz_aware_still_works(self):
        """tz-aware datetime — работает как прежде (регресс-тест)."""
        from datetime import datetime, timezone
        from newsbot.offpeak import is_deepseek_peak

        # Пн 01:30 UTC → пик
        peak_dt = datetime(2026, 1, 5, 1, 30, 0, tzinfo=timezone.utc)
        assert is_deepseek_peak(peak_dt) is True

        # Пн 13:00 UTC → off-peak
        offpeak_dt = datetime(2026, 1, 5, 13, 0, 0, tzinfo=timezone.utc)
        assert is_deepseek_peak(offpeak_dt) is False
